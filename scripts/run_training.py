# MAIN PROGETTO ML

import numpy as np
import matplotlib.pyplot as plt
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
    x_i, d = load_data("../data/training_data/ML-CUP25-TR.csv")

    print("x_i.shape: ", x_i.shape, " d.shape: ", d.shape)
    x_i = x_i.to_numpy()
    d = d.to_numpy()

    # ----------------------- INIZIALIZZA NN -------------------------
    # x: vettore input layer
    # w: matrice pesi input layer - hidden layer
    # k: matrice pesi hidden layer - output layer
    # d: vettori risultati target

    x_i, w_ji, w_kj, d = initialize_neuraln(x_i, d) # <-- Inizializza NN (static)

    print("x_i_biased.shape: ", x_i.shape, "w_ji.shape: ", w_ji.shape, "w_kj.shape: ", w_kj.shape)
    # ----------------------------------------------------------------

    # -------------------- FORWARD PRIMO PATTERN ---------------------    
    #x_j = forward_hidden(x_i[0], w_ji)     # <-- Calcolo valori nodi unico hidden layer
    #x_k = forward_output(x_j, w_kj)        # <-- Calcolo valori output
    # ----------------------------------------------------------------
    
    #x_k, x_j = forward_all_layers(x_i[0], w_ji, w_kj)

    eta = 0.01
    # per un pattern
    #w_new = w + etha * (-2 * (e[0]) * dsigmaf(x)) 
    
    # ||||||||||||||||||||||            ONLINE          ||||||||||||||||||||||
    #"""
    #while True:
    epochs = 10
    total_error_array = np.zeros(epochs)
    patterns = x_i.shape[0]
    for epoch in range(epochs):


        total_error = np.zeros(w_kj.shape[1]) # total error si azzera ad ogni epoch
        MSE_k = np.zeros(w_kj.shape[1])
        for pattern in range (patterns):

            x_k, x_j = forward_all_layers(x_i[pattern], w_ji, w_kj)
            #print("x_k.shape: ", x_k.shape, " x_j.shape: ", x_j.shape)

            dj, dk = compute_delta_all_layers(d[pattern], x_k, w_kj, x_j, x_i[pattern], w_ji, dsigmaf)
            #print("delta j: ", dj, "\n\ndelta k: ", dk, "\n\n\n")

            for kunit in range (w_kj.shape[1]):
                for junit in range (w_kj.shape[0]):

                    w_kj[junit][kunit] +=  ( eta * dk[kunit] * x_j[junit] )

                #total_error += dk[kunit]   # <-- Calcolo errore totale come sommatoria degli errori dei pattern, da poi plottare
                MSE_k[kunit]+= (d[pattern][kunit]-x_k[kunit])**2

            for junit in range (w_ji.shape[1]):
                for iunit in range (w_ji.shape[0]):
                    w_ji[iunit][junit] += ( eta * dj[junit] * x_i[pattern][iunit] )
            print("pattern=", pattern, "\nx_k", x_k, "\ntargets=", d[pattern], "\ndelta_k=",dk)
        MSE_k =  MSE_k / patterns
        MSE_tot=0
        for i in range (w_kj.shape[1]):
            print(MSE_k[i])
            MSE_tot+= MSE_k[i]
        total_error=MSE_tot/w_kj.shape[1]
        total_error_array[epoch] = total_error
        print("!!! TOTAL ERROR: ", total_error, " !!!")




    ep = [ x for x in range(epochs) ]

    plt.plot(ep, total_error_array)
    plt.show()
        
    #    if total_error <= eps:
    #        break
        
        #print("|| Pattern number: ", pattern, " delta_k: ", dk, " delta_j: ", dj," ||", "\n w_ji: ", w_ji, "\n w_kj: ", w_kj)
        #print("|| Pattern number: ", pattern, " delta_k: ", dk, " delta_j: ", dj," ||")
    #"""

    # ||||||||||||||||||||||            BATCH          ||||||||||||||||||||||
    """
    while True: 
        
        patterns = 500

        for pattern in range(patterns):


            x_k, x_j = forward_all_layers(x_i[pattern], w_ji, w_kj)
            #print("x_k.shape: ", x_k.shape, " x_j.shape: ", x_j.shape   )
            dj, dk += compute_delta_all_layers(d[pattern], x_k, w_kj, x_j, x_i[pattern], w_ji, dsigmaf)

            dj += 


            


        if total_error >= e :


            break

    """


if __name__ == "__main__":
    run_training()
