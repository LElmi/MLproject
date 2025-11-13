def get_hyperparameters(grid_search=False):
    """
    Prompt user for hyperparameters or return a grid for grid search.
    """
    if grid_search:
        return {
            "lr": [0.001, 0.0005],
            "lam": [1e-4, 1e-3],
            "dropout_rate": [0.1, 0.2],
            "epochs": [2000],
            "batch_size": [64],
            "h1": [128, 64],
            "h2": [64, 32],
            "h3": [32, 16]
        }

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

    return lr, lam, dropout_rate, epochs, batch_size, h1, h2, h3
