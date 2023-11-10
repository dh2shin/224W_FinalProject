import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Normalization constants on speed
mean = 61.77375
std = 9.293026

def mean_absolute_percentage_error(y_true, y_pred, epsilon=0.01):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

class ModelEvaluator:
    def __init__(self, y_pred, y_test, X_test, model_name, unnormalize=False):
        if unnormalize:
            y_pred = y_pred*std + mean
            y_true = y_true*std + mean
        self.y_pred = y_pred
        self.y_test = y_test
        self.X_test = X_test
        self.model_name = model_name

    def evaluate(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        with open("results/" + self.model_name + ".txt", "w+") as f:
            f.write(f"Root Mean Squared Error: {np.sqrt(mse):.3f} \n")
            f.write(f"Mean Absolute Error: {mae:.3f} \n")
            f.write(f"Mean Absolute Percentage Error: {mape:.3f}% \n")