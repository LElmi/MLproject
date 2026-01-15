import numpy as np
import math as math
from typing import Callable, List, Optional


Array2D = np.ndarray
Array1D = np.ndarray


def delta_k(d: Array1D, x_k: Array1D, x_prev: Array1D, w_prev: Array2D, f_act: Callable = None) -> Array1D: 
    """
    Funzione che calcola il delta k vettore di un pattern, quindi una componente fondamentale per calcolare la backprop sull'hidden layer

    Args:
        d: vettore target
        x_k: vettore output layer
        (opzionale) f_deactiv: derivata della funzione di attivazione, nel caso dell'output layer è linaere quindi ritorno al x_k
        x_j: vettore hidden layer

    Ritorna: 
        dk: vettore dei delta k 
    """
    # Inizializza vettore delta k vuoto
    #dk = np.zeros(x_k.size)

    # Aggiorna il vettore deltak con la differenza tra target e previsione * la derivata della funzione lineare quindi 1
    #dk = (d - x_k) # * 1 # <- Versione vettorializzata

    """
    for kunit in range(x_k.size):
        # Aggiorna il vettore deltak con la differenza tra target e previsione * la derivata della funzione lineare quindi 1
        dk[kunit] = (d[kunit] - x_k[kunit]) * 1

    #print(f"-----  dk[kunit] = (d[kunit] - x_k[kunit]) * x_k[kunit]) = {d[kunit]} - {x_k[kunit]} * {x_k[kunit]}")
    """

    if f_act == None: 
        return d - x_k
    else: 
        net_k = np.dot(x_prev, w_prev[1:]) + w_prev[0]
        return d - x_k * f_act(net_k, True)



def delta_j(w_ji: Array2D, x_prev: Array1D, w_next: Array2D, 
            delta_next: Array1D, f_act: Callable) -> Array1D:
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

    #n_units = w_ji.shape[1]
    #dj = np.zeros(n_units)

    error_signal = np.dot(w_next[1:, :], delta_next)
    net_j = np.dot(x_prev, w_ji[1:]) + w_ji[0]
    derivative = f_act(net_j, True)
    return error_signal * derivative

"""    for unit in range(n_units):

        net_j = np.dot(w_ji[1:, unit], x_prev) + w_ji[0][unit] # <- Aggiunge il bias
        sommatoria = np.dot(w_next[unit + 1, :], delta_next) # <- Salta la prima riga dei bias

        for kunit in range(delta_k.size):
            sum_parz += delta_k[kunit] * w_kj[junit, kunit]

        # La f_act ha un parametro aggiuntivo, se True, ritorna la derivata della funzione
        dj[unit] = sommatoria * df_act(net_j, True)"""


    #return dj
    
    
def compute_delta_weights(delta: Array1D, x_prev: Array1D) -> Array2D:
    """
    Calcola la matrice dei delta per i pesi dato il delta e l'input.
    
    Args:
        delta: vettore delta del layer corrente
        x_prev: vettore del layer precedente
    
    Returns:
        delta_w: matrice dei delta pesi (include bias alla riga 0)
    """
    # Aggiungo un valore che consiste nel bias, maggiori informazioni in appunti_utili/ appunti_gestione_batch
    x_prev_biased = np.append(1, x_prev)
    # Outer product tra x_prev e delta
    delta_w = np.outer(x_prev_biased, delta)

    return delta_w

def compute_delta_all_layers_list(
    d: Array1D,                         # Target
    layer_results_list: List[Array1D],  # Output di ogni layer dal forward
    weights_matrix_list: List[Array2D], # Pesi attuali
    x_pattern: Array1D,                 # Input originale della rete
    f_act_hidden: Callable,                   # Derivata funzione attivazione
    f_act_output: Callable,
    old_deltas: Optional[List[Array2D]] = None, # Per il momentum
    alpha_momentum: float = 0.0,
    max_norm_gradient_for_clipping: float = 5
) -> tuple[List[Array2D], float]:
    
    """
    Esegue la backpropagation su una lista arbitraria di layer.
    Dentro il ciclo for prende il risultato dell'operazione forward su tutta la rete: layer_result_list,
    considerando che l'ultimo elemento è il risultato dell'output finale, nel caso della backpropagation
        - questo array viene letto al contrario (1)
        - scorriamo l'array calcolando il delta con la funzione "delta_j" (2)
        - se siamo al primo elemento (quindi alla prima iterazione) utilizziamo l'x_pattern (3) calcolando il delta con "delta_k"

    """
    
    # Liste per salvare i delta (errori) e i delta_w (gradienti pesi)
    # Li riempiremo all'inverso e poi faremo il reverse alla fine
    deltas_list = []
    weight_gradients_list = []
    
    num_layers = len(weights_matrix_list)


    for i in range(num_layers - 1, -1, -1): # (1)

        curr_output = layer_results_list[i]
        
        if i == 0:
            prev_output = x_pattern
        else:
            prev_output = layer_results_list[i - 1]
            
        w_curr = weights_matrix_list[i]
                
        if i == num_layers - 1: 
            delta = delta_k(d, curr_output, prev_output, w_curr, f_act_output) # (3)

        else:
            w_next = weights_matrix_list[i + 1]
            delta_next = deltas_list[-1]
            
            #print("w_next: ", w_next, "\n\n\n\n", "delta_next: ", delta_next, "\n\n\n\n")
            delta = delta_j(w_curr, prev_output, w_next, delta_next, f_act_hidden) # (2)
            
        # Salviamo il delta (servirà per il layer precedente nel prossimo giro del ciclo)
        deltas_list.append(delta)
        
        
        grad = compute_delta_weights(delta, prev_output)
        
        if old_deltas is not None:
            # old_deltas[i] contiene il vecchio aggiornamento per questo strato
            grad = grad + (alpha_momentum * old_deltas[i])
            
        weight_gradients_list.append(grad)

    # Poiché abbiamo iterato all'indietro (Output -> Input), le liste sono invertite.
    # Dobbiamo girarle per farle combaciare con l'ordine all'esterno
    weight_gradients_list.reverse()
    
    weight_gradients_list, grad_norm = gradient_norm_with_clipping(weight_gradients_list, max_norm_gradient_for_clipping)

    return weight_gradients_list, grad_norm

def gradient_norm_with_clipping(grad_list: List[Array2D], max_norm: float) -> float:
    """ 
    Calcola la norma globale del gradiente, aggiornato con una versione
    di clipping, quindi la norma massima viene passata per parametro
    
    """
    total_sum = 0.0
    for g in grad_list:
        total_sum += np.sum(g ** 2)
    
    current_norm = math.sqrt(total_sum)
        
    if current_norm > max_norm:
        scaling_factor = max_norm / (current_norm + 1e-6)
        grad_list = [g * scaling_factor for g in grad_list]
        return grad_list, max_norm # La norma ora è limitata a max_norm
    
    return grad_list, current_norm