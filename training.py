import numpy as np
from models.forward_pass import forward
from models.backward_pass import backward


def train_with_early_stopping(
    model, optimizer,
    X_train, Y_train, X_val, Y_val,
    epochs=1000, batch_size=64,
    patience=200, min_delta=1e-4,
    verbose=True
):
    """Training loop with early stopping."""

    best_val_loss = float("inf")
    best_weights = {}
    patience_counter = 0
    model.loss_history = []

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[perm], Y_train[perm]

        for i in range(0, len(X_train), batch_size):
            Xb = X_train[i:i + batch_size]
            Yb = Y_train[i:i + batch_size]

            Y_pred, cache = forward(
                Xb,
                model.W1, model.b1,
                model.W2, model.b2,
                model.W3, model.b3,
                model.W4, model.b4,
                model.dropout_rate,
                training=True
            )

            grads = backward(Y_pred, Yb, cache,
                             model.W1, model.W2, model.W3, model.W4,
                             model.lam)
            optimizer.update(model, grads)

        # Compute losses
        Y_val_pred, _ = forward(
            X_val,
            model.W1, model.b1,
            model.W2, model.b2,
            model.W3, model.b3,
            model.W4, model.b4,
            model.dropout_rate,
            training=False
        )
        val_loss = np.mean((Y_val_pred - Y_val) ** 2)
        model.loss_history.append(val_loss)

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} - Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {p: getattr(model, p).copy() for p in
                            ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    for p, v in best_weights.items():
        setattr(model, p, v)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    return model
