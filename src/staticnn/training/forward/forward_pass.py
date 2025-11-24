import numpy as np
from src.staticnn.activationf.sigm import sigmaf

# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray

def forward_hidden(x_i: Array1D, w_ji: Array2D) -> Array1D:
    """
    Calcola l'output del hidden layer con act. func. sigma
    
    Args: 
        X: vettore input (n_features)
        W: matrice pesi (n_features, n_hidden)
    
    Ritorna: 
        vettore attivazioni hidden layer (n_hidden)
    """
    n_hidden_units = w_ji.shape[1]
    x_j = np.zeros(n_hidden_units)

    for i in range(n_hidden_units):
        x_j[i] = sigmaf(np.dot(x_i, w_ji[:, i]))

    return x_j


def forward_output(x_j: Array1D, w_kj: Array2D) -> Array1D:
    """
    Calcola l'output layer (lineare).
    
    Args:
        X1: vettore attivazioni hidden layer (n_hidden)
        K: matrice pesi output (n_hidden, n_outputs)
    
    Ritorna:
        vettore predizioni
    """
    n_outputs = w_kj.shape[1]
    x_k = np.zeros(n_outputs)

    for i in range(n_outputs):
        x_k[i] = np.dot(x_j, w_kj[:, i])

    return x_k
