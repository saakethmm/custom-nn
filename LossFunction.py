import numpy as np

def L2_norm(self, y_true, y_pred):
    return np.sum(np.power(y_true - y_pred, 2))

def L2_norm_prime(self, y_true, y_pred):
    return 2*(y_pred - y_true)

def mse(self, y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(self, y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
