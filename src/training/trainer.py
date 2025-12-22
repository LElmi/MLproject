# MAIN PROGETTO ML

import numpy as np
import time

import config
from src.nn.nn import NN
from src.training.forward.forward_pass import *
from src.training.backward.backprop import *
from src.utils import visualization as vs
from src.utils import save_model as sm
from typing import Callable, Dict, Tuple, List


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
                 n_hidden1: int,
                 n_hidden2: int,
                 n_outputs: int,
                 f_act: Callable,
                 learning_rate: float,
                 batch: bool,
                 epochs: int,
                 early_stopping: bool,
                 epsilon: float,
                 patience: int,
                 momentum: bool,
                 alpha_mom: float,
                 split: float):
        

        self.f_act = f_act
        self.batch = batch
        self.epochs = epochs
        self.epsilon = epsilon
        self.patience = patience
        self.momentum = momentum
        self.alpha_mom = alpha_mom
        self.split = split
        #self.stopping_criteria = stopping_criteria
        self.early_stopping = early_stopping
        # Inizializza la rete neurale
        self.neuraln = NN(
                        input_size, 
                        n_hidden1, 
                        n_hidden2, 
                        n_outputs, 
                        f_act,
                        learning_rate)
        
        self.mee_error_history = [] # <- Quella per la CUP, meno sensibile agli outliers
        self.mse_error_history = []
        # self.total_error_array = []

        self.old_deltas = None # <-- Necessario per il momentum

    def train_standard(self, input_matrix, d_matrix):

        n_patterns = input_matrix.shape[0]
        prev_gradient_norm_epoch = None
        epoch = 0

        start_time = time.perf_counter()
        print(f"Inizio training...")
        for i in range (self.epochs):
            epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns)

            self.mee_error_history.append(epoch_results["mee"])
            self.mse_error_history.append(epoch_results["mse"])


            print("|| epooch n° ", i, ", total mee error: ", epoch_results["mee"], " ||")

        print(" Total mee error: ", epoch_results["mee"])
        print(" Total mse error: ", epoch_results["mse"])
        print(" Tempo di training: ", time.perf_counter() - start_time)
        print("\n--- Training Completato ---\n")

        # Per PLOT
        vs.plot_errors(self, time.perf_counter() - start_time)

    def train_with_early_stopping(self, input_matrix, d_matrix):

        n_patterns = input_matrix.shape[0]
        gradient_misbehave = 0
        prev_gradient_norm_epoch = None
        epoch = 0
        
        start_time = time.perf_counter()
        print(f"Inizio training...")
            
        # Aggiunge lo STOPPING CRITERIA basato sulla norma del gradiente, che se non scende di molto
        # oltre un certo numero di epoche (= patience) allora ritorna il risultato
        while gradient_misbehave < self.patience and epoch < self.epochs:

            epoch += 1
            
            # Computa un epoca
            epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns)

            self.mee_error_history.append(epoch_results["mee"])
            self.mse_error_history.append(epoch_results["mse"])

            # dà un'idea dell'errore medio per ogni neurono di output,
            #MSE_k = np.sqrt(MSE_k) / patterns
            #MSE_tot = 0
            #for i in range(self.neuraln.w_kj2.shape[1]):
            #    MSE_tot += MSE_k[i]
            #total_error = MSE_tot / self.neuraln.w_kj2.shape[1]
            #self.total_error_array[epoch] = total_error
            
            # Necessario perché confronti i risultati prima con il nuovo

            prev_gradient_norm_epoch, gradient_misbehave = self._check_patience(gradient_misbehave,
                                                                                prev_gradient_norm_epoch, 
                                                                                epoch_results["grad_norm"],
                                                                                n_patterns)

            print("|| epooch n° ", epoch, ", total mee error: ", epoch_results["mee"], " ||")


        print(" Total mee error: ", epoch_results["mee"]) 
        print(" Total mse error: ", epoch_results["mse"]) 
        print(" Tempo di training: ", time.perf_counter() - start_time)



        print("\n--- Training Completato ---\n")
        sm.save_model(self.neuraln.w_kj2,self.neuraln.w_j2j1,self.neuraln.w_j1i)
                
        # Per PLOT
        vs.plot_errors(self, time.perf_counter() - start_time)



    def _run_epoch(self, input_matrix: np.ndarray, d_matrix: np.ndarray, n_patterns: int) -> Dict[str, float]:
        """
        Metodo che nasce con l'esigenza di portare un po' di logica fuori dal train,
        runna una epoca, restitutuendo le informazioni sull'errore
        """

        indices = np.arange(n_patterns)
        if not self.batch: np.random.shuffle(indices)

        epoch_mee, epoch_mse, epoch_grad = 0.0, 0.0, 0.0

        # Creo una lista di batch quindi batch_deltas = [ [dwk], [dwj2], [dwj1]]
        batch_deltas = [np.zeros_like(w) for w in [self.neuraln.w_kj2, self.neuraln.w_j2j1, self.neuraln.w_j1i]]

        for idx in indices:
            x, d = input_matrix[idx], d_matrix[idx]

            self.neuraln.forward(x)

            epoch_mee += (np.sum((d - self.neuraln.x_k) ** 2)) ** 0.5
            epoch_mse += (np.sum((d - self.neuraln.x_k) ** 2))

            # L'asterisco serve per raggruppare in lista tutti i risultati
            # tranne, in questo caso, l'ultimo. Essendo specificato.
            # quindi deltas = [dwk, dwj2, dwj1]

            if self.batch and self.momentum:
                *deltas, grad_norm = compute_delta_all_layers_with_momentum(
                    d, self.neuraln.x_k, self.neuraln.w_kj2, self.neuraln.x_j2,
                    self.neuraln.w_j2j1, self.neuraln.x_j1, self.neuraln.w_j1i, x, self.f_act,
                    self.old_deltas, self.alpha_mom)
                
                self.old_deltas = deltas

            else:
                *deltas, grad_norm = compute_delta_all_layers(
                    d, self.neuraln.x_k, self.neuraln.w_kj2, self.neuraln.x_j2,
                    self.neuraln.w_j2j1, self.neuraln.x_j1, self.neuraln.w_j1i, x, self.f_act
                )

            
            epoch_grad += grad_norm


            if self.batch:
                # Accumula i delta
                for i in range(len(batch_deltas)): batch_deltas[i] += deltas[i]
            else:
                self.neuraln.update_weights(*deltas)
        
        dwk_epoch, dwj2j1_epoch, dwj1i_epoch = batch_deltas
        
        if self.batch:

            # (?)
            dwk_epoch    /= n_patterns
            dwj2j1_epoch /= n_patterns
            dwj1i_epoch  /= n_patterns  
            
            self.neuraln.update_weights(dwk_epoch, dwj2j1_epoch, dwj1i_epoch)

        return {
            'mee': epoch_mee / n_patterns,
            'mse': epoch_mse / n_patterns,
            'grad_norm': epoch_grad / n_patterns
        }
    

    def _check_patience(self, gradient_misbehave: int, prev_gradient_norm_epoch: float, grad_norm: float, n_patterns: int) -> float: 
        
        current_grad_norm = grad_norm / n_patterns

        # Controlla se il gradiente sale ancora
        if prev_gradient_norm_epoch is not None:
            diff = abs(current_grad_norm - prev_gradient_norm_epoch) / (prev_gradient_norm_epoch + 1e-10)
            if diff < self.epsilon:
                gradient_misbehave += 1
            else:
                gradient_misbehave = 0
        
        return current_grad_norm, gradient_misbehave

    def grid_search_train(self, input_matrix, d_matrix, epochs):

        n_patterns = input_matrix.shape[0]
        for i in range (epochs):
            epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns)

        self.mee_error_history.append(epoch_results["mee"])
        self.mse_error_history.append(epoch_results["mse"])

        return epoch_results["mee"]


    def grid_search_train_with_early_stopping(self, input_matrix, d_matrix, epochs):
        n_patterns = input_matrix.shape[0]
        gradient_misbehave = 0
        prev_gradient_norm_epoch = None
        epoch = 0

        # Aggiunge lo STOPPING CRITERIA basato sulla norma del gradiente, che se non scende di molto
        # oltre un certo numero di epoche (= patience) allora ritorna il risultato
        while gradient_misbehave < self.patience and epoch < self.epochs:
            epoch += 1
            epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns)

            self.mee_error_history.append(epoch_results["mee"])
            self.mse_error_history.append(epoch_results["mse"])
            prev_gradient_norm_epoch, gradient_misbehave = self._check_patience(gradient_misbehave,
                                                                          prev_gradient_norm_epoch,
                                                                                epoch_results["grad_norm"],
                                                                                n_patterns)
        print("Early stopping triggerato dopo ",epoch," epochs")

        return epoch_results["mee"]

