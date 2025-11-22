import numpy as np
from src.staticnn.activationf.sigm import sigmaf

# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray

def forward_hidden(x: Array1D, w: Array2D) -> Array1D:
    """
    Calcola l'output del hidden layer con act. func. sigma
    
    Args: 
        X: vettore input (n_features)
        W: matrice pesi (n_features, n_hidden)
    
    Ritorna: 
        vettore attivazioni hidden layer (n_hidden)
    """
    n_hidden = w.shape[1]
    x_1 = np.zeros(n_hidden)

    for i in range(n_hidden):


        x_1[i] = sigmaf(np.dot(x, w[:, i]))

    print(x_1)
    return x_1


def forward_output(x_1: Array1D, k: Array2D) -> Array1D:
    """
    Calcola l'output layer (lineare).
    
    Args:
        X1: vettore attivazioni hidden layer (n_hidden)
        K: matrice pesi output (n_hidden, n_outputs)
    
    Ritorna:
        vettore predizioni
    """
    n_outputs = k.shape[1]
    out = np.zeros(n_outputs)

    for i in range(n_outputs):
        out[i] = np.dot(x_1, k[:, i])

    return out
