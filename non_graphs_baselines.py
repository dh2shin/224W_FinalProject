import lightgbm as lgb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from flaml import AutoML
from matplotlib import pyplot as plt
from load_data import DataLoader
from evaluate_model import ModelEvaluator
import pickle
import warnings
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ValueWarning)
from sklearn.cluster import SpectralClustering

class NonGraphBaseline:
    def __init__(self, model_type, load_data=False):
        # Supports "historical_averages", "automl", "timeseries"
        self.model_type = model_type
        train_functions = {"historical_averages": self.train_historical_averages,
                           "automl": self.train_automl, "timeseries": self.train_ARIMA}
        predict_functions = {"historical_averages": self.predict_historical_averages,
                           "automl": self.predict_automl, "timeseries": self.predict_ARIMA}
        self.train_function = train_functions[model_type]
        self.predict_function = predict_functions[model_type]
        if not load_data:
            train_df = pd.read_csv("data/preprocessed/train.csv")
            test_df = pd.read_csv("data/preprocessed/test.csv")
            graph_structure = np.load("data/pems_adj_mat.npy")
        else:
            data_loader = DataLoader(baseline=False)
            data_loader.load_data()
            train_df = data_loader.train
            test_df = data_loader.test
            graph_structure = data_loader.graph_structure
        # Compute clusters
        n_clusters = 5
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
        labels = sc.fit_predict((graph_structure + graph_structure.T)/2)
        self.graph_clustering = {i: labels[i] for i in range(len(labels))}
        self.features = ["time_index", "node_index", "normalized_time"]
        self.target = "speed"
        self.train = train_df
        self.test = test_df
        self.X_train, self.y_train = train_df[self.features], train_df[self.target]
        self.X_test, self.y_test = test_df[self.features], test_df[self.target]
        self.SHORT_TERM_PREDICTION_PERIOD = 12

    def train_historical_averages(self):
        self.model = self.train.groupby(["node_index", "normalized_time"])["speed"].mean().reset_index()

    def predict_historical_averages(self, mode):
        if mode == "long_term":
            predictions_df = self.X_test.merge(self.model, on=["node_index", "normalized_time"], how="left")
            return predictions_df["speed"]
        else:
            test_start = self.X_test["time_index"].min()
            test_end = self.X_test["time_index"].max()
            n_nodes = len(pd.unique(self.X_train["node_index"]))
            node_preds = {j: np.zeros(test_end-test_start+1) for j in range(n_nodes)}
            initial_obs_df = self.X_train[(self.X_train["time_index"] >= test_start - self.SHORT_TERM_PREDICTION_PERIOD) &
                                         (self.X_train["time_index"] < test_start)]
            for j in range(n_nodes):
                new_obs_indices = initial_obs_df[initial_obs_df["node_index"] == j].index
                new_pred = self.y_train.loc[new_obs_indices].mean()
                node_preds[j][0:self.SHORT_TERM_PREDICTION_PERIOD] = new_pred*np.ones(self.SHORT_TERM_PREDICTION_PERIOD)
            for i in range(test_start + self.SHORT_TERM_PREDICTION_PERIOD, test_end, self.SHORT_TERM_PREDICTION_PERIOD):
                print(i)
                new_obs_df = self.X_test[(self.X_test["time_index"] >= i - self.SHORT_TERM_PREDICTION_PERIOD) &
                                         (self.X_test["time_index"] < i)]
                for j in range(n_nodes):
                    new_obs_indices = new_obs_df[new_obs_df["node_index"] == j].index
                    new_pred = self.y_test.loc[new_obs_indices].mean()
                    if i <= test_end - self.SHORT_TERM_PREDICTION_PERIOD:
                        node_preds[j][i-test_start:i-test_start+self.SHORT_TERM_PREDICTION_PERIOD] = new_pred*np.ones(self.SHORT_TERM_PREDICTION_PERIOD)
                    else:
                        node_preds[j][i-test_start:] = new_pred * np.ones(len(node_preds[j])-i+test_start)
            node_preds = list(node_preds.values())
            return np.column_stack(node_preds).ravel()

    def train_automl(self):
        X_train = self.X_train.copy()
        X_train["cluster"] = X_train["node_index"].apply(lambda x: self.graph_clustering[int(x)])
        X_train["cos_time"] = np.cos(X_train["normalized_time"])
        X_train["sin_time"] = np.sin(X_train["normalized_time"])
        X_train = X_train.drop(columns="normalized_time")
        automl = AutoML()
        automl_settings = {
            "time_budget": 300,
            "metric": "rmse",
            "task": "regression",
            "log_file_name": "logs/automl.log",
            "estimator_list": ["lgbm"]
        } 
        automl.fit(X_train=X_train, y_train=self.y_train, **automl_settings)
        self.model = automl.model
        best_model_name = "models/automl.pkl"
        with open(best_model_name, 'wb') as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    def predict_automl(self, mode):
        X_test = self.X_test.copy()
        X_test["cluster"] = X_test["node_index"].apply(lambda x: self.graph_clustering[int(x)])
        X_test["cos_time"] = np.cos(X_test["normalized_time"])
        X_test["sin_time"] = np.sin(X_test["normalized_time"])
        X_test = X_test.drop(columns="normalized_time")
        if mode == "long_term":
            return self.model.predict(X_test)

    def train_ARIMA(self):
        n_nodes = len(pd.unique(self.X_train["node_index"]))
        chosen_nodes = np.random.randint(0, n_nodes, size=5)
        np.save("data/chosen_nodes.npy", chosen_nodes)
        self.model = []
        for i in chosen_nodes:
            print(i)
            ts = self.train[self.train["node_index"] == i]["speed"].copy()
            order = (0, 1, 1)
            model = sm.tsa.arima.ARIMA(ts, order=order)
            self.model.append(model.fit())
    
    def predict_ARIMA(self, mode):
        chosen_nodes = np.random.randint(325, size=5)
        test_start = self.X_test["time_index"].min()
        test_end = self.X_test["time_index"].max()
        n_nodes = len(pd.unique(self.X_train["node_index"]))
        if mode == "long_term":
            node_preds = []
            for i in range(n_nodes):
                predictions = self.model[i].predict(start=test_start, end=test_end)
                node_preds.append(predictions)
            return np.column_stack(node_preds).ravel()
        elif mode == "short_term":
            chosen_nodes = np.load("data/chosen_nodes.npy", allow_pickle=True)
            node_preds = {j: np.zeros(test_end-test_start+1) for j in range(len(chosen_nodes))}
            for i in range(test_start, test_end, self.SHORT_TERM_PREDICTION_PERIOD):
                print(i)
                new_obs_df = self.X_test[(self.X_test["time_index"] >= i - self.SHORT_TERM_PREDICTION_PERIOD) &
                                         (self.X_test["time_index"] < i)]
                for j, node in enumerate(chosen_nodes):
                    if i > test_start:
                        new_obs_indices = new_obs_df[new_obs_df["node_index"] == node].index
                        new_obs = self.y_test.loc[new_obs_indices].to_list()
                        fit_new_obs = self.model[j].append(new_obs, refit=False)
                    else:
                        fit_new_obs = self.model[j]
                    predictions = fit_new_obs.forecast(self.SHORT_TERM_PREDICTION_PERIOD).to_numpy()
                    if i <= test_end - self.SHORT_TERM_PREDICTION_PERIOD:
                        node_preds[j][i-test_start:i-test_start+self.SHORT_TERM_PREDICTION_PERIOD] = predictions
                    else:
                        node_preds[j][i-test_start:] = predictions[:len(node_preds[j])-i+test_start]
            node_preds = list(node_preds.values())
            node_preds = np.column_stack(node_preds).ravel()
            np.save("data/node_preds.npy", node_preds)
            return node_preds
        

if __name__ == "__main__":
    model_type = "timeseries"
    mode = "short_term"
    model = NonGraphBaseline(model_type)
    model.train_function()
    y_pred = model.predict_function(mode)
    evaluator = ModelEvaluator(y_pred, model.y_test, model.X_test, model_type)
    evaluator.evaluate()