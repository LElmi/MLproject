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
    return dk

def delta_j(x_i: Array1D, w_ji: Array2D, x_j: Array1D, w_kj: Array2D, delta_k: Array1D, x_k: Array1D, fd: Callable) -> Array1D:
    """
    Funzione che calcola il vettore delta_j corrispondente all'hiddent layer

    Args:
        x_j: vettore hidden layer
        w_kj: matrice dei pesi tra l'ultimo hidden layer e l'otput layer
        delta_k: vettore dei delta k
        x_k: vettore output layer
        fd: derivata della funzione di attivazione dell'hidden layer da passare
    Ritorna:
        dj: vettore dei delta j
    """
    dj = np.zeros(x_j.size)
    for junit in range(x_j.size):
        net_j = np.dot(x_i, w_ji[:, junit])
        sum_parz = 0
        for kunit in range(delta_k.size):
            sum_parz += delta_k[kunit] * w_kj[junit, kunit]
        dj[junit] = sum_parz * fd(net_j)
    return dj

def compute_delta_all_layers(d: Array1D, x_k: Array1D, w_kj2: Array2D, x_j1: Array1D, w_j2j1: Array2D, x_j2: Array1D, w_j1i: Array2D, x_i: Array1D, fd: Callable) -> tuple[Array1D, Array1D, Array1D]:
    """
    Funzione che calcola i vettori delta per tutti i layer

    Args:
        d: vettore target
        x_k: vettore output layer
        w_kj2: matrice dei pesi tra il secondo hidden layer e l'output layer
        x_j1: vettore del primo hidden layer
        w_j2j1: matrice dei pesi tra il primo hidden layer e il secondo hidden layer
        x_j2: vettore del secondo hidden layer
        w_j1i: matrice dei pesi tra l'input layer e il primo hidden layer
        x_i: vettore input layer
        fd: derivata della funzione di attivazione dell'hidden layer da passare
    Ritorna:
        dj1: vettore dei delta per il primo hidden layer
        dj2: vettore dei delta per il secondo hidden layer
        dk: vettore dei delta per l'output layer
    """
    dk = delta_k(d, x_k)
    dj2 = delta_j(x_j1, w_j2j1, x_j2, w_kj2, dk, x_k, fd)
    dj1 = delta_j(x_i, w_j1i, x_j1, w_j2j1, dj2, x_j2, fd)
    return dj1, dj2, dk

        
#def backprop_output(d: Array1D, x_k: Array1D, f_prime: function, w_kj: Array2D, x_j: Array1D,  ):
    """
    Calcola per ogni k che appartengono all'output layer su ogni nodo appartenente all'hidden layer precedente

    Args:
        x_k: vettore output layer
        w_kj: matrice dei pesi tra l'ultimo hidden layer e l'otput layer
        x_j: vettore hidden layer

    Ritorna:
        matrice k aggiornata
    """

#def backprop_hidden()