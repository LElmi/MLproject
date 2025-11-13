import itertools
import numpy as np
from models.deepnet_reg import DeepNetReg
from utils.data_utils import load_data, normalize

def grid_search(X, Y, param_grid, epochs=500, batch_size=64):
    """
    Perform a simple grid search over hyperparameters.

    param_grid: dict
        Example: {
            "lr": [0.001, 0.005],
            "lam": [1e-4, 1e-3],
            "dropout_rate": [0.1, 0.2],
            "h1": [64, 128],
            "h2": [32, 64],
            "h3": [16, 32]
        }
    """
    keys = list(param_grid.keys())
    best_score = float("inf")
    best_params = None
    results = []

    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        print("\nTesting combination:", params)

        model = DeepNetReg(
            input_dim=X.shape[1],
            h1=params["h1"],
            h2=params["h2"],
            h3=params["h3"],
            output_dim=Y.shape[1],
            lr=params["lr"],
            lam=params["lam"],
            dropout_rate=params["dropout_rate"]
        )
        model.train(X, Y, epochs=epochs, batch_size=batch_size, verbose=False)
        preds, _ = model.forward(X, training=False)
        mse = np.mean((preds - Y) ** 2)
        print(f" -> MSE: {mse:.6f}")

        results.append((params, mse))
        if mse < best_score:
            best_score = mse
            best_params = params

    print("\nBest Parameters:")
    print(best_params)
    print(f"Best MSE: {best_score:.6f}")

    return best_params, results
