# MAIN PROGETTO ML

import numpy as np
import time
from src.nn.nn import NN
from src.training.forward.forward_pass import *
#from src.activationf.sigmoid import *
from src.training.backward.backprop import *
from src.utils import visualization as vs
#from tqdm import tqdm # Barra di progressione


# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray


class Trainer:
    """
    La classe ha lo scopo di orchestrare le fasi di training in base ai parametri passati,
    il suo scopo è:
        1) Creare cartella relativa all'istanza di training, con gli storici e i dati interessanti
        2) Orchestrare il forward, update weights, backpropagation

    """

    def __init__ (self,
                 input_size: int,
                 n_hidden1: int = 64,
                 n_hidden2: int = 32,
                 n_outputs: int = 4,
                 learning_rate: float = 0.000025,
                 epochs: int = 3000):
        
        self.epochs = epochs
        #self.stopping_criteria = stopping_criteria

        # Inizializza la rete neurale
        self.neuraln = NN(
                        input_size, 
                        n_hidden1, 
                        n_hidden2, 
                        n_outputs, 
                        learning_rate)
        
        self.mee_error_history = [] # <- Quella per la CUP, meno sensibile agli outliers
        self.mse_error_history = []
        # self.total_error_array = []


    def train(self, input_matrix, d_matrix, dfunact: Callable):

        patterns = input_matrix.shape[0]
        # Inizializza il vettore dove ogni elemento è la sommatoria degli errori dei pattern per epoca
        self.mee_error_history = np.zeros(self.epochs)
        self.mse_error_history = np.zeros(self.epochs)
        # self.total_error_array = np.zeros(self.epochs)  ## 



        #total_error_array = np.zeros(self.epochs)

        start_time = time.perf_counter()
        print(f"Inizio training...")

        for epoch in range(self.epochs):

            shuffled_indeces = np.random.permutation(patterns)
            mee_pattern_error = 0.0
            mse_pattern_error = 0.0 


            # total_error = np.zeros(self.neuraln.w_kj2.shape[1]) 
            # MSE_k = np.zeros(self.neuraln.w_kj2.shape[1])

            for idx_pattern in shuffled_indeces:
                
                x_i = input_matrix[idx_pattern]
                d = d_matrix[idx_pattern]
                self.neuraln.forward(input_matrix[idx_pattern])

                dj1, dj2, dk = compute_delta_all_layers(
                                                d,
                                                self.neuraln.x_k,
                                                self.neuraln.w_kj2, 
                                                self.neuraln.x_j1,
                                                self.neuraln.w_j2j1,
                                                self.neuraln.x_j2,
                                                self.neuraln.w_j1i,
                                                x_i,
                                                dfunact
                                            )

                mee_pattern_error += (np.sum((d - self.neuraln.x_k) ** 2)) ** 0.5
                mse_pattern_error += (np.sum((d - self.neuraln.x_k) ** 2))
                # MSE_k += (d - self.neuraln.x_k) ** 2

                self.neuraln.update_weights(dk, dj2, dj1, x_i)

               # print("size d: ", d.shape, "self.neuraln.x_k shape", self.neuraln.x_k.shape)

            mean_epoch_mee_error = mee_pattern_error / patterns
            mean_epoch_mse_error = mse_pattern_error / patterns

            self.mee_error_history[epoch] = mean_epoch_mee_error
            self.mse_error_history[epoch] = mean_epoch_mse_error


            # dà un'idea dell'errore medio per ogni neurono di output,
            # ora  potrebbe essere utile per identificare quali output della rete hanno più difficoltà nell'apprendimento
            # accumula gli errori quadratici di ogni output
            # commentato momentaneamente per performance
            
            #MSE_k = np.sqrt(MSE_k) / patterns
            #MSE_tot = 0
            #for i in range(self.neuraln.w_kj2.shape[1]):
            #    MSE_tot += MSE_k[i]
            #total_error = MSE_tot / self.neuraln.w_kj2.shape[1]
            #self.total_error_array[epoch] = total_error
            

            print("|| epooch n° ", epoch, ", total mee error: ", mean_epoch_mee_error, " ||")


        tempo_di_training = time.perf_counter() - start_time


        print(" Total mee error: ", mean_epoch_mee_error) 
        print(" Total mse error: ", mean_epoch_mse_error) 
        print(" Tempo di training: ", tempo_di_training)



        print("\n--- Training Completato ---\n")
                
        # Genera il vettore delle epoche (X-axis)
        epoche = np.arange(self.epochs) # Usa np.arange per coerenza

        vs.plot_errors(self, epoche, tempo_di_training)
        # self.plot_errors_separate(epoche) # Se preferisci grafici separati
