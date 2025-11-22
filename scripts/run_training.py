# MAIN PROGETTO ML

import numpy as np
from src.staticnn.model.fixednn import initialize_neuraln
from src.staticnn.training.forward.forward_pass import *
from data.utils.load_data import load_data


# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray


def run_training() -> None:
    # X: dataset input (N, 12)
    # D: dataset target (N, 4)
    x, d = load_data("data/training_data/ML-CUP25-TR.csv")

    x = x.to_numpy()
    d = d.to_numpy()

    # ----------------------- INIZIALIZZA NN -------------------------
    x, w, k, d = initialize_neuraln(x, d) # <-- Inizializza NN (static)
    # ----------------------------------------------------------------

    # -------------------- FORWARD PRIMO PATTERN ---------------------    
    x_1 = forward_hidden(x[0], w)      # <-- Calcolo valori nodi unico hidden layer
    out = forward_output(x_1, k)       # <-- Calcolo valori output
    # ----------------------------------------------------------------

    print("OUT =", out)


if __name__ == "__main__":
    run_training()
