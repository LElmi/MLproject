# MAIN PROGETTO ML

import numpy as np
import time
from typing import Callable, Dict, Tuple, List

import config.cup_config as config
from src.nn.nn import NN
from src.training.trainer.forward.forward_pass import *
from src.training.trainer.backward.backprop import compute_delta_all_layers_list
from src.utils import visualization as vs
from src.utils import save_model as sm


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
                 units_list: list[int],
                 n_outputs: int,
                 f_act: Callable,
                 learning_rate: float,
                 use_decay: bool,
                 decay_factor: float,
                 decay_step: int,
                 batch: bool,
                 epochs: int,
                 early_stopping: bool,
                 epsilon: float,
                 patience: int,
                 momentum: bool,
                 alpha_mom: float,
                 split: float,
                 verbose: bool = False):
        

        self.f_act = f_act
        self.batch = batch
        self.epochs = epochs
        self.epsilon = epsilon
        self.patience = patience
        self.momentum = momentum
        self.alpha_mom = alpha_mom
        self.split = split
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.use_decay = use_decay
        self.decay_factor = decay_factor
        self.decay_step = decay_step

        # Inizializza la rete neurale
        self.neuraln = NN(
                    n_inputs = input_size, 
                    units_list = units_list, 
                    n_outputs = n_outputs, 
                    f_act = f_act,
                )
        
        self.mee_error_history = [] # <- Quella per la CUP, meno sensibile agli outliers
        self.mse_error_history = []
        # self.total_error_array = []
        self.verbose = verbose

        self.old_deltas = None # <-- Necessario per il momentum

    def fit(self, input_matrix, d_matrix):
        """
        Metodo centrale della classe Train. Fa le seguenti cose:
            - Prende come argomenti:
                . Tutti i pattern dei valori in ingresso CUP (500, 14)
                . Tutti i corrispondenti pattern dei valori risultati CUP (500, 4)

            - Itera le epoche al cui interno si itera sui pattern, per ogni epoca:
                . Avvia il timer
                . Chiama il metodo interno run epoch che restituisce i risultati sull'errore corrente
                . Controlla se il criterio di fermata è soddisfatto, se sì ferma l'iterazione
        """

        n_patterns = input_matrix.shape[0]
        gradient_misbehave = 0
        prev_gradient_norm = None

        start_time = time.perf_counter()

        if self.verbose:
            print(f"Inizio training (Early stopping: {self.early_stopping})...")

        for epoch in range (1, self.epochs + 1):
            #print("\n\n\n\n\nsize_x: ", input_matrix.shape, "\n\n\n\n\n", "size_d: ", d_matrix.shape)
            epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns, epoch)

            self.mee_error_history.append(epoch_results["mee"])
            self.mse_error_history.append(epoch_results["mse"])

            if self.verbose and epoch % 10 == 0:
                print("|| epooch n° ", epoch, ", total mee error: ", epoch_results["mee"], " ||")

            if self.early_stopping:
                prev_gradient_norm, gradient_misbehave = self._check_patience(
                    gradient_misbehave, prev_gradient_norm, epoch_results["grad_norm"], n_patterns
                )
                if gradient_misbehave >= self.patience:
                    break

        if self.verbose: 
            print(" Total mee error: ", epoch_results["mee"])
            print(" Total mse error: ", epoch_results["mse"])
            print(" Tempo di training: ", time.perf_counter() - start_time)
            print("\n--- Training Completato ---\n")
            vs.plot_errors(self, time.perf_counter() - start_time)

        return self.mee_error_history[-1]
    

    def _run_epoch(self, input_matrix: np.ndarray, d_matrix: np.ndarray, n_patterns: int, epoch: int) -> Dict[str, float]:
        """
        Metodo che nasce con l'esigenza di portare un po' di logica fuori dal train,
        runna una epoca, restitutuendo le informazioni sull'errore.
        Questo metodo gestisce la divisione della logica in base all'esplorazione tramite batch o online,

        """

        indices = np.arange(n_patterns)
        if not self.batch: np.random.shuffle(indices)

        epoch_mee, epoch_mse, epoch_grad = 0.0, 0.0, 0.0

        # Crea una lista di batch quindi batch_deltas = [ [dwk], [dwj2], [dwj1]], 
        # in base a quanti sono le matrici dentro la lista dei pesi
        if self.batch: 
            batch_deltas = [np.zeros_like(w) for w in self.neuraln.weights_matrix_list]

        # Scorre tutti gli indici dividendo il comportamento in base a se è batch oppure online
        for idx in indices:
            #print("\n\n\n\n\nsize_d: ", d_matrix.shape)
            x, d = input_matrix[idx], d_matrix[idx]
            #print("\n\n\n\n\nsize_x: ", x.shape, "\n\n\n\n\n", "size_d: ", d.shape)

            layer_results = self.neuraln.forward_network(x)
            final_output = layer_results[-1]

            epoch_mee += (np.sum((d - final_output) ** 2)) ** 0.5
            epoch_mse += (np.sum((d - final_output) ** 2))

            # L'asterisco serve per raggruppare in lista tutti i risultati
            # tranne, in questo caso, l'ultimo. Essendo specificato.
            # quindi deltas = [dwk, dwj2, dwj1], nel caso di una rete con 2 hidden layer

            deltas, grad_norm = compute_delta_all_layers_list(
                            d = d,
                            layer_results_list = layer_results,
                            weights_matrix_list = self.neuraln.weights_matrix_list,
                            x_pattern = x,
                            df_act = self.f_act, # Nota: La funzione backprop gestirà la derivata internamente
                            old_deltas = self.old_deltas if self.momentum else None,
                            alpha_momentum = self.alpha_mom,
                            max_norm_gradient_for_clipping = 5
                        )
                        
            epoch_grad += grad_norm

            #if epoch % 50 == 0:
            #    print("VEDERE SE NECESSARIO CLIPPING: ", grad_norm, "EPOCA: ", epoch, "\n")

            # CASO BATCH
            if self.batch:
                for i in range(len(batch_deltas)):
                    batch_deltas[i] += deltas[i]
            # CASO ONLINE
            else:
                self.neuraln.update_weights(deltas)
                if self.momentum: self.old_deltas = deltas

        # Se fine epoca e se batch, aggiorna gli update weights 
        if self.batch:
            # Media dei gradienti
            avg_deltas = [d_mat / n_patterns for d_mat in batch_deltas]

            if self.use_decay and epoch > 0 and epoch % self.decay_step == 0:
                self.learning_rate *= self.decay_factor
            
            self.neuraln.update_weights(avg_deltas, eta = self.learning_rate)
            # Salva per momentum prossima epoca
            if self.momentum: self.old_deltas = avg_deltas

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

