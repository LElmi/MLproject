import numpy as np

def load_data(filename):
    data = np.loadtxt(filename, delimiter=",", comments="#")
    X = data[:, 1:-4]
    Y = data[:, -4:]
    return X, Y

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std
