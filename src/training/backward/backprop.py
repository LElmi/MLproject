import numpy as np
from typing import Callable


Array2D = np.ndarray
Array1D = np.ndarray



def delta_k(d: Array1D, x_k: Array1D) -> Array1D: 
    """
    Funzione che calcola il delta k vettore, quindi una componente fondamentale per calcolare la backprop sull'hidden layer

    Args:
        d: vettore target
        x_k: vettore output layer
        (opzionale) f_deactiv: derivata della funzione di attivazione, nel caso dell'output layer è linaere quindi ritorno al x_k
        x_j: vettore hidden layer

    Ritorna: 
        dk: vettore dei delta k 
    """
    # Inizializza vettore delta k vuoto
    dk = np.zeros(x_k.size)
    
    for kunit in range(x_k.size):
        # Aggiorna il vettore deltak con la differenza tra target e previsione * la derivata della funzione lineare quindi 1
        dk[kunit] = (d[kunit] - x_k[kunit]) * 1

    #print(f"-----  dk[kunit] = (d[kunit] - x_k[kunit]) * x_k[kunit]) = {d[kunit]} - {x_k[kunit]} * {x_k[kunit]}")
    return dk



def delta_j(x_i: Array1D, w_ji: Array2D, x_j: Array1D, w_kj: Array2D, delta_k: Array1D, x_k: Array1D, fd: Callable) -> Array1D:
    """
    Funzione che calcola il vettore delta_j corrispondente all'hidden layer
    
    Args: 
        x_j: vettore hidden layer
        w_kj: matrice dei pesi tra l'ultimo hidden layer e l'otput layer
        delta_k: vettore dei delta k 
        x_k: vettore output layer
        fd: derivata della funzione di attivazione dell'hidden layer da passare

    Ritorna:
        dj: vettore dei delta j
    """
    # Inizializza vettore delta j vuoto

    n_junits = x_j.size
    dj = np.zeros(n_junits)

    for junit in range(n_junits):

        net_j = np.dot(x_i, w_ji[1:, junit]) + w_ji[0][junit] # <- Aggiunge il bias
        sommatoria_sui_dk = np.dot(delta_k, w_kj[junit + 1, :]) # <- Salta la prima riga dei bias

        #for kunit in range(delta_k.size):
        #    sum_parz += delta_k[kunit] * w_kj[junit, kunit]

        dj[junit] = sommatoria_sui_dk * fd(net_j)

    return dj
    
    



def compute_delta_all_layers(d: Array1D, x_k: Array1D, w_kj2: Array2D, x_j1: Array1D, w_j2j1: Array2D, x_j2: Array1D, w_j1i: Array2D, x_i: Array1D, fd: Callable) -> tuple[Array1D, Array1D, Array1D]:
    dk = delta_k(d, x_k)

    dj2 = delta_j(x_j1, w_j2j1, x_j2, w_kj2, dk, x_k, fd)


    dj1 = delta_j(x_i, w_j1i, x_j1, w_j2j1, dj2, x_j2, fd)


    return dj1, dj2, dk        
