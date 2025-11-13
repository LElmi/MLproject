import numpy as np

def compute_loss(Y_pred, Y_true, lam, W1, W2, W3, W4):
    mse = np.mean((Y_pred - Y_true) ** 2)
    reg = lam * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    return mse + reg
