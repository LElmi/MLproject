import numpy as np
from models.relu import relu_deriv

def backward(Y_pred, Y, cache, W1, W2, W3, W4, lam):
    # Handle accidental double-wrapped cache ((Y_pred, cache))
    if len(cache) == 2 and isinstance(cache[1], tuple):
        cache = cache[1]

    X, z1, a1, mask1, z2, a2, mask2, z3, a3, mask3 = cache
    m = X.shape[0]

    dZ4 = (Y_pred - Y) * (2 / m)
    dW4 = a3.T @ dZ4 + 2 * lam * W4
    db4 = np.sum(dZ4, axis=0, keepdims=True)

    dA3 = dZ4 @ W4.T * mask3
    dZ3 = dA3 * relu_deriv(z3)
    dW3 = a2.T @ dZ3 + 2 * lam * W3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T * mask2
    dZ2 = dA2 * relu_deriv(z2)
    dW2 = a1.T @ dZ2 + 2 * lam * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T * mask1
    dZ1 = dA1 * relu_deriv(z1)
    dW1 = X.T @ dZ1 + 2 * lam * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
            "W4": dW4, "b4": db4}
