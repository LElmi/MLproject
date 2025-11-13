import numpy as np
from models.forward_pass import forward
from models.compute_loss import compute_loss
from models.backward_pass import backward

def train(model, X, Y, epochs=2000, batch_size=64, verbose=True):
    n = X.shape[0]
    for e in range(epochs):
        idx = np.random.permutation(n)
        X, Y = X[idx], Y[idx]
        batch_losses = []

        for i in range(0, n, batch_size):
            Xb, Yb = X[i:i+batch_size], Y[i:i+batch_size]
            Yp, cache = forward(Xb, model.W1, model.b1,
                                model.W2, model.b2,
                                model.W3, model.b3,
                                model.W4, model.b4,
                                model.dropout_rate, training=True)
            loss = compute_loss(Yp, Yb, model.lam,
                                model.W1, model.W2, model.W3, model.W4)
            grads = backward(Yp, Yb, cache,
                             model.W1, model.W2, model.W3, model.W4,
                             model.lam)
            model.optimizer.update(model, grads)
            batch_losses.append(loss)

        model.loss_history.append(np.mean(batch_losses))
        if verbose and (e % 100 == 0 or e == epochs - 1):
            print(f"Epoch {e}/{epochs} - Loss: {model.loss_history[-1]:.6f}")
