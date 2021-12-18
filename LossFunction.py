import numpy as np

def L2_norm(y_true, y_pred):
    return np.sum(np.power(y_true - y_pred, 2))

def L2_norm_prime(y_true, y_pred):
    return 2*(y_pred - y_true)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    p = np.exp(y_pred)/np.sum(np.exp(y_pred))
    log_likelihood = -np.sum(y_true * np.log(p))
    return log_likelihood

def cross_entropy_prime(y_true, y_pred):
    p = np.exp(y_pred)/np.sum(np.exp(y_pred))
    return p - y_true




