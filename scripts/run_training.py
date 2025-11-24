# MAIN PROGETTO ML

import numpy as np
from src.staticnn.model.fixednn import initialize_neuraln
from src.staticnn.training.forward.forward_pass import *
from data.utils.load_data import load_data
from src.staticnn.activationf.sigm import *
from src.staticnn.training.backward.backprop import *


# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray
Array0D = np.ndarray


def run_training() -> None:
    # X: dataset input (N, 12)
    # D: dataset target (N, 4)
    x_i, d = load_data("data/training_data/ML-CUP25-TR.csv")

    x_i = x_i.to_numpy()
    d = d.to_numpy()

    # ----------------------- INIZIALIZZA NN -------------------------
    # x: vettore input layer
    # w: matrice pesi input layer - hidden layer
    # k: matrice pesi hidden layer - output layer
    # d: vettori risultati target

    x_i, w_ji, w_kj, d = initialize_neuraln(x_i, d) # <-- Inizializza NN (static)
    # ----------------------------------------------------------------

    # -------------------- FORWARD PRIMO PATTERN ---------------------    
    x_j = forward_hidden(x_i[0], w_ji)     # <-- Calcolo valori nodi unico hidden layer
    x_k = forward_output(x_j, w_kj)        # <-- Calcolo valori output
    # ----------------------------------------------------------------
    
    eta = 1
    # per un pattern
    #w_new = w + etha * (-2 * (e[0]) * dsigmaf(x)) 
    
    print("x_k =", x_k)


if __name__ == "__main__":
    run_training()
