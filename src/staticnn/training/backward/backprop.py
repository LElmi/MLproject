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
        # Aggiorna il vettore deltak con la differenza tra target e previsione * la derivata della funzione lineare quindi l'elemento x_k
        dk[kunit] = (d[kunit] - x_k[kunit]) * 1

    #print(f"-----  dk[kunit] = (d[kunit] - x_k[kunit]) * x_k[kunit]) = {d[kunit]} - {x_k[kunit]} * {x_k[kunit]}")
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
    # Inizializza vettore delta j vuoto
    dj = np.zeros(x_j.size)

    
    for junit in range(x_j.size):

        sum_parz_w_kj = 0
        # Somma tra tutti i delta[k] * e il peso corrispondente della matrice w_kj, fissato il nodo j (=junit) di destinazione 
        for k in range(x_k.size):
            sum_parz_w_kj += (delta_k[k] * w_kj[junit][k])
        
        # La Net del nodo j = x_j[junit]
        dj[junit] = sum_parz_w_kj * fd(np.dot(x_i, w_ji[:, junit]))

    return dj
    
    
def compute_delta_all_layers(d: Array1D, x_k: Array1D, w_kj: Array2D, x_j: Array1D, x_i: Array1D, w_ji: Array2D, fd: Callable) -> tuple[Array1D, Array1D]: 

    dk = delta_k(d, x_k)
    return delta_j(x_i, w_ji, x_j, w_kj, dk, x_k, fd), dk
        
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