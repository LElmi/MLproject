import numpy as np
import itertools
from utils.data_utils import load_data, normalize
from utils.plotting_utils import plot_loss_curve, plot_predictions
from models.deepnet_reg import DeepNetReg
from models.adam_optimizer import AdamOptimizer
from training import train_with_early_stopping
from models.forward_pass import forward


def grid_search(X, Y, param_grid, epochs=300, batch_size=64):
    """Run a basic grid search on the model."""
    keys = list(param_grid.keys())
    best_score = float("inf")
    best_params = None

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
        optimizer = AdamOptimizer(lr=params["lr"])
        train_with_early_stopping(model, optimizer, X, Y, X, Y, epochs=epochs, batch_size=batch_size, patience=50, verbose=False)

        preds, _ = forward(
            X,
            model.W1, model.b1,
            model.W2, model.b2,
            model.W3, model.b3,
            model.W4, model.b4,
            model.dropout_rate,
            training=False
        )
        mse = np.mean((preds - Y) ** 2)
        print(f" -> MSE: {mse:.6f}")

        if mse < best_score:
            best_score = mse
            best_params = params

    print("\nBest configuration found:")
    print(best_params)
    print(f"Best MSE: {best_score:.6f}")
    return best_params


if __name__ == "__main__":
    # --------------------------
    # Load and normalize data
    # --------------------------
    X, Y = load_data("data/ML-CUP25-TR.csv")
    X, X_mean, X_std = normalize(X)
    Y, Y_mean, Y_std = normalize(Y)

    # Train/validation split
    val_ratio = 0.1
    n_samples = X.shape[0]
    n_val = int(n_samples * val_ratio)
    perm = np.random.permutation(n_samples)
    X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]
    X_train, Y_train = X[perm[n_val:]], Y[perm[n_val:]]

    # --------------------------
    # User prompts
    # --------------------------
    use_grid_search = input("Use grid search for hyperparameters? [y/N]: ").strip().lower() == "y"
    use_early_stopping = input("Use early stopping during training? [y/N]: ").strip().lower() == "y"

    # --------------------------
    # Get hyperparameters
    # --------------------------
    if use_grid_search:
        param_grid = {
            "lr": [0.001, 0.0025, 0.005],
            "lam": [0.0001,0.0005, 0.001],
            "dropout_rate": [0.1, 0.2],
            "h1": [128, 96, 64],
            "h2": [64, 32],
            "h3": [32, 16, 8]
        }
        best_params = grid_search(X_train, Y_train, param_grid)
        lr = best_params["lr"]
        lam = best_params["lam"]
        dropout_rate = best_params["dropout_rate"]
        h1, h2, h3 = best_params["h1"], best_params["h2"], best_params["h3"]
    else:
        lr = float(input("Learning rate (e.g. 0.001): ") or 0.001)
        lam = float(input("Regularization λ (e.g. 0.001): ") or 0.001)
        dropout_rate = float(input("Dropout rate (0–1): ") or 0.1)
        h1 = int(input("Hidden layer 1 size (e.g. 128): ") or 128)
        h2 = int(input("Hidden layer 2 size (e.g. 64): ") or 64)
        h3 = int(input("Hidden layer 3 size (e.g. 32): ") or 32)

    epochs = 2000
    batch_size = 64

    # --------------------------
    # Initialize model + optimizer
    # --------------------------
    model = DeepNetReg(
        input_dim=X.shape[1], h1=h1, h2=h2, h3=h3,
        output_dim=Y.shape[1], lr=lr, lam=lam, dropout_rate=dropout_rate
    )
    optimizer = AdamOptimizer(lr=lr)

    # --------------------------
    # Training
    # --------------------------
    if use_early_stopping:
        patience = max(100, epochs // 5)  # less aggressive
        min_delta = 1e-4
        train_with_early_stopping(
            model, optimizer,
            X_train, Y_train, X_val, Y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience, min_delta=min_delta,
            verbose=True
        )
    else:
        model.train(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=True)

    # --------------------------
    # Evaluation
    # --------------------------
    preds_norm, _ = forward(
        X,
        model.W1, model.b1,
        model.W2, model.b2,
        model.W3, model.b3,
        model.W4, model.b4,
        model.dropout_rate,
        training=False
    )
    preds = preds_norm * Y_std + Y_mean
    true_vals = Y * Y_std + Y_mean

    per_target_mse = np.mean((preds - true_vals) ** 2, axis=0)
    mae = np.mean(np.abs(preds - true_vals))
    mean_abs_true = np.mean(np.abs(true_vals))
    accuracy = 100 * (1 - mae / (mean_abs_true + 1e-8))

    print("\n=== Evaluation Metrics ===")
    print("Per-target MSE:", per_target_mse)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Regression Accuracy: {accuracy:.2f}%")

    # --------------------------
    # Plots
    # --------------------------
    plot_loss_curve(model.loss_history)
    plot_predictions(true_vals, preds)
