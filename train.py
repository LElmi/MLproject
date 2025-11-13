import sys
import numpy as np
from utils.data_utils import load_data, normalize
from utils.plotting_utils import plot_loss_curve, plot_predictions
from models.deepnet_reg import DeepNetReg

def get_manual_hyperparameters():
    print("\n=== Neural Network Hyperparameter Setup ===")

    def get_input(prompt, default, cast_func):
        try:
            val = input(f"{prompt} [{default}]: ")
            if val.strip() == "":
                return default
            return cast_func(val)
        except (ValueError, EOFError):
            print(f"Invalid input or input not supported, using default ({default}).")
            return default

    lr = get_input("Learning rate", 0.001, float)
    lam = get_input("L2 regularization (lambda)", 1e-4, float)
    dropout_rate = get_input("Dropout rate", 0.1, float)
    epochs = get_input("Epochs", 2000, int)
    batch_size = get_input("Batch size", 64, int)
    h1 = get_input("Hidden layer 1 size", 128, int)
    h2 = get_input("Hidden layer 2 size", 64, int)
    h3 = get_input("Hidden layer 3 size", 32, int)

    print("\nConfiguration:")
    print(f"  Learning rate: {lr}")
    print(f"  L2 lambda:     {lam}")
    print(f"  Dropout:       {dropout_rate}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Layers:        [{h1}, {h2}, {h3}]\n")

    return lr, lam, dropout_rate, epochs, batch_size, h1, h2, h3


def grid_search(X, Y):
    print("\n=== Running Grid Search ===")
    best_loss = float("inf")
    best_params = None

    lrs = [0.001, 0.0005]
    lams = [1e-4, 1e-3]
    dropouts = [0.1, 0.2]
    hidden_sizes = [(128, 64, 32), (64, 64, 32)]

    for lr in lrs:
        for lam in lams:
            for dropout_rate in dropouts:
                for h1, h2, h3 in hidden_sizes:
                    print(f"Testing lr={lr}, lam={lam}, dropout={dropout_rate}, layers=[{h1},{h2},{h3}]")
                    model = DeepNetReg(
                        input_dim=X.shape[1],
                        h1=h1, h2=h2, h3=h3,
                        output_dim=Y.shape[1],
                        lr=lr, lam=lam,
                        dropout_rate=dropout_rate
                    )
                    model.train(X, Y, epochs=500, batch_size=64, verbose=False)
                    loss = model.loss_history[-1]
                    print(f"  -> Loss: {loss:.6f}")
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (lr, lam, dropout_rate, 500, 64, h1, h2, h3)

    print(f"\nBest params found: lr={best_params[0]}, lam={best_params[1]}, dropout={best_params[2]}, "
          f"layers=[{best_params[5]}, {best_params[6]}, {best_params[7]}], Loss={best_loss:.6f}\n")
    return best_params


def main():
    X, Y = load_data("data/ML-CUP25-TR.csv")
    X, X_mean, X_std = normalize(X)
    Y, Y_mean, Y_std = normalize(Y)

    use_grid_search = False
    try:
        ans = input("Do you want to perform a grid search? [y/N]: ").strip().lower()
        if ans == "y":
            use_grid_search = True
    except EOFError:
        print("Input not supported, using manual hyperparameters.")

    if use_grid_search:
        lr, lam, dropout_rate, epochs, batch_size, h1, h2, h3 = grid_search(X, Y)
    else:
        lr, lam, dropout_rate, epochs, batch_size, h1, h2, h3 = get_manual_hyperparameters()

    model = DeepNetReg(
        input_dim=X.shape[1],
        h1=h1, h2=h2, h3=h3,
        output_dim=Y.shape[1],
        lr=lr, lam=lam,
        dropout_rate=dropout_rate
    )

    model.train(X, Y, epochs=epochs, batch_size=batch_size)

    preds_norm, _ = model.forward(X, training=False)
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

    plot_loss_curve(model.loss_history)
    plot_predictions(true_vals, preds)


if __name__ == "__main__":
    main()
