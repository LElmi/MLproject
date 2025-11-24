import numpy as np

# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray

####        INIZIALIZZAZIONE RETE NEURALE STATICA VALORI HARDCODED FULL CONNECT         ####
#
#.  HIDDEN LAYER = 1 
#.  UNITS = 28
#   ---->  numero totale di pesi in una full connect è: 
#               numero unità input layer (12) * numero unità hidden layer (28 FIXED) = totale 336 unità
#
####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####


def initialize_neuraln(x_i, d) -> tuple[Array1D, Array2D, Array2D, Array1D]:
    """
    Costruisce una rete neurale fissata di un hidden layer

    Args:
        x_i: l'input layer
        x_k: l'output layer
        d: targets
    
    Ritorna:
        x_i: l'input layer
        w_ji: matrice dei pesi W verso l'hidden layer
        w_kj:  ""   ""    ""   K verso l'output layer  
        x_k: l'output layer
    """

    # Numero features e dimensioni NN 
    n_inputs = x_i.shape[1] + 1     # 12 + bias
    n_hidden = 28                   # fissato
    n_outputs = d.shape[1]          # 4

    # Inizializzazione pesi
    w_ji = np.random.uniform(low = -0.7, high = 0.7, size = (n_inputs, n_hidden)) # (12 × 28)
    w_kj = np.random.uniform(low = -0.7, high = 0.7, size = (n_hidden, n_outputs)) # (28 × 4) 

    # BIAS
    # Il bias deve essere aggiunto come un valore (= 1) in più sul vettore x -> x_0 (= 1) + x_1 + .... + x_n
    # e come peso in più w_0. Questo funziona da treshold
    rows = x_i.shape[0]
    cols = x_i.shape[0]
    bias = [[1] * 1 for _ in range(rows)]
    x_ibiased = [[0] * cols for _ in range(rows)]

    x_ibiased = np.hstack((x_i,bias))

    
    return x_ibiased, w_ji, w_kj, d
            



"""
random.uniform(low=0.0, high=1.0, size=None)

Draw samples from a uniform distribution.

Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by uniform.

Note

New code should use the uniform method of a Generator instance instead; please see the Quick start.

Parameters:

    low
    float or array_like of floats, optional

        Lower boundary of the output interval. All values generated will be greater than or equal to low. The default value is 0.
    high
    float or array_like of floats

        Upper boundary of the output interval. All values generated will be less than or equal to high. The high limit may be included in the returned array of floats due to floating-point rounding in the equation low + (high-low) * random_sample(). The default value is 1.0.
    size
    int or tuple of ints, optional

        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if low and high are both scalars. Otherwise, np.broadcast(low, high).size samples are drawn.


"""