from utils.data_utils import load_data, normalize
from utils.plotting_utils import plot_loss_curve, plot_predictions
from models.deepnet_reg import DeepNetReg
import numpy as np


def get_hyperparameters():
    print("\n=== Neural Network Hyperparameter Setup ===")

    def get_input(prompt, default, cast_func):
        val = input(f"{prompt} [{default}]: ")
        if val.strip() == "":
            return default
        try:
            return cast_func(val)
        except ValueError:
            print(f"Invalid input, using default ({default}).")
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


if __name__ == "__main__":
    X, Y = load_data("data/ML-CUP25-TR.csv")
    X, X_mean, X_std = normalize(X)
    Y, Y_mean, Y_std = normalize(Y)

    lr, lam, dropout_rate, epochs, batch_size, h1, h2, h3 = get_hyperparameters()

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
