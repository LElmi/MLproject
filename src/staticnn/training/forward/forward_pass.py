import numpy as np
from src.staticnn.activationf.sigm import sigmaf
from src.staticnn.activationf.relu import *
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
    """"
    n_hidden_units = w_ji.shape[1]
    x_j = np.zeros(n_hidden_units)
    #print("inside forward hidden, x_j.size: ", x_j.shape, "w_ji.shape: ", w_ji.shape)
    for junit in range(n_hidden_units):
        x_j[junit] = relu(np.dot(x_i, w_ji[:, junit]))
        #x_j[i] = relu(np.dot(x_i, w_ji[:, i]))
    """
    z_j = np.dot(w_ji.T, x_i)  # (n_hidden, n_features) @ (n_features,) -> (n_hidden,)
    #z_j = np.clip(z_j, -100, 100)
    x_j = relu(z_j)  # Applica la sigmoide a tutti gli elementi
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
    """
    n_outputs = w_kj.shape[1]
    x_k = np.zeros(n_outputs)

    for i in range(n_outputs):
        x_k[i] = np.dot(x_j, w_kj[:, i])
    """
    x_k = np.dot(w_kj.T, x_j)
    return x_k

    
    
def forward_all_layers(x_i: Array1D, w_j1i: Array2D,w_j2j1: Array2D, w_kj2: Array2D) -> tuple[Array1D, Array1D, Array1D]:

    x_j1 = forward_hidden(x_i, w_j1i)
    x_j2 = forward_hidden(x_j1, w_j2j1)
    return forward_output(x_j2, w_kj2), x_j2,x_j1