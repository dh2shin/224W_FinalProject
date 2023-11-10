from torch_geometric_temporal.dataset import PemsBayDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, baseline=True, train_test_split=0.8):
        self.baseline = baseline
        self.train_test_split = train_test_split

    def load_data(self):
        if self.baseline:
            A = np.load("data/pems_adj_mat.npy")
            X = np.load("data/pems_node_values.npy")
            speeds = X[:, :, 0]
            times = X[:, :, 1]
            time_indices = np.array(range(X.shape[0])).repeat(X.shape[1])
            node_indices = np.array(list(range(X.shape[1])) * X.shape[0])
            data_df = pd.DataFrame({"time_index": time_indices, "node_index": node_indices, "speed": speeds.flatten(), "normalized_time": times.flatten()})
            # Split into train and test
            train_df = data_df[data_df["time_index"] <= self.train_test_split * X.shape[0]]
            test_df = data_df[data_df["time_index"] > self.train_test_split * X.shape[0]]
            train_df.to_csv("data/preprocessed/train.csv", index=False)
            test_df.to_csv("data/preprocessed/test.csv", index=False)
            self.graph_structure = A
            self.train = train_df
            self.test = test_df
        else:
            loader = PemsBayDatasetLoader()
            # dataset has attributes edge_index, edge_weight, features, and targets (both 325x2x12)
            dataset = loader.get_dataset()
            train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=self.train_test_split)
            self.train = train_dataset
            self.test = test_dataset