import numpy as np
from models.deepnet_reg import DeepNetReg
from utils.data_utils import load_data, normalize
from utils.hyperparam_search import grid_search
from utils.plotting_utils import plot_results  # assuming you have this

# ===============================
# Load and normalize data
# ===============================
X, Y = load_data("data/ML-CUP25-TR.csv")
X, X_mean, X_std = normalize(X)
Y, Y_mean, Y_std = normalize(Y)

# ===============================
# Choose mode
# ===============================
print("Select training mode:")
print("1. Manual hyperparameter input")
print("2. Automatic grid search")
mode = input("Enter 1 or 2: ").strip()

if mode == "1":
    # Manual input mode with suggested defaults
    lr = float(input("Learning rate [default=0.001]: ") or 0.001)
    lam = float(input("L2 regularization λ [default=1e-4]: ") or 1e-4)
    dropout_rate = float(input("Dropout rate [default=0.1]: ") or 0.1)
    h1 = int(input("Hidden layer 1 size [default=128]: ") or 128)
    h2 = int(input("Hidden layer 2 size [default=64]: ") or 64)
    h3 = int(input("Hidden layer 3 size [default=32]: ") or 32)
    epochs = int(input("Epochs [default=2000]: ") or 2000)
    batch_size = int(input("Batch size [default=64]: ") or 64)

    model = DeepNetReg(X.shape[1], h1, h2, h3, Y.shape[1],
                       lr=lr, lam=lam, dropout_rate=dropout_rate)

    model.train(X, Y, epochs=epochs, batch_size=batch_size, verbose=True)
    preds, _ = model.forward(X, training=False)
    mse = np.mean((preds - Y) ** 2)
    print(f"Final training MSE: {mse:.6f}")

else:
    # Grid search mode
    param_grid = {
        "lr": [0.001, 0.003, 0.01],
        "lam": [1e-4, 1e-3],
        "dropout_rate": [0.1, 0.2],
        "h1": [64, 128],
        "h2": [32, 64],
        "h3": [16, 32],
    }
    best_params, results = grid_search(X, Y, param_grid, epochs=800, batch_size=64)
    print("\nGrid search complete.")
    print("Best hyperparameters found:", best_params)
