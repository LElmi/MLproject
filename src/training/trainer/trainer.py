# MAIN PROGETTO ML

import numpy as np
import time
from typing import Callable, Dict
from src.nn.nn import NN
from src.training.trainer.forward.forward_pass import *
from src.training.trainer.backward.backprop import compute_delta_all_layers_list
from src.utils import *
from src.training.validation.validation_monk import accuracy
from src.training.trainer.stopper import Stopper

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
                 max_gradient_norm: float,
                 split: float,
                 verbose: bool = False,      # <- Importante da togliere nella grid search
                 validation: bool = False):  # <- Importante da togliere nella grid search


        self.f_act = f_act
        self.learning_rate = learning_rate
        self.batch = batch
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.patience = patience
        self.momentum = momentum
        self.alpha_mom = alpha_mom
        self.use_decay = use_decay
        self.decay_factor = decay_factor
        self.decay_step = decay_step
        self.max_gradient_norm = max_gradient_norm
        self.verbose = verbose
        self.validation = validation

        self.epoch = 0
        self.old_deltas = None

        # Inizializza la rete neurale
        self.neuraln = NN(
                    n_inputs = input_size, 
                    units_list = units_list, 
                    n_outputs = n_outputs, 
                    f_act = f_act,
                )
                
        self.tr_mee_history = [] 
        self.tr_mse_history = []

        self.vl_mee_history = []
        self.vl_mse_history = []

    def fit(self, 
            tr_x: np.ndarray, 
            tr_d: np.ndarray,
            vl_x: np.ndarray = None,
            vl_d: np.ndarray = None,
            metric_fun: Callable = None,
            metric_mode: str = 'min'):
        """
        Metodo centrale della classe Train. Fa le seguenti cose:
            - Prende come argomenti:
                . Tutti i pattern dei valori in ingresso CUP (500, 14)
                . Tutti i corrispondenti pattern dei valori risultati CUP (500, 4)
                . metric_fn: Funzione che accetta (output, target) e ritorna un float
                    tipo compute_mee per CUP, compute_accuracy per MONK.
                . metric_mode, perché? 
                    1) Nel caso ad esempio della mee abbiamo bisogno di fermarci appena l'mee inizia a salire
                    2) Nel caso ad esempio dell'accuracy o di altre statistiche ci vogliamo fermare quando inizia
                        a scendere

            - Itera le epoche al cui interno si itera sui pattern, per ogni epoca:
                . Avvia il timer
                . Chiama il metodo interno run epoch che restituisce i risultati sull'errore corrente
                . Controlla se il criterio di fermata è soddisfatto, se sì ferma l'iterazione
        """

        n_patterns = tr_x.shape[0]
        start_time = time.perf_counter()

        # Gestisce la logica dell'early stopping
        # e si assicura di salvare la matrice dei pesi
        # appena capisce che si deve fermare, l'early stopping
        # per adesso è attiva solo nel caso sia attiva la validation
        stopper = Stopper(
            patience = self.patience,
            min_delta = self.epsilon,
            mode = metric_mode
        )
        
        # ! AVVIO CICLO DI EPOCHE ! #
        for epoch in range(1, self.epochs + 1):
            
            self.epoch = epoch 
            tr_epoch_results = self._run_epoch(tr_x, tr_d, n_patterns)

            self.tr_mee_history.append(tr_epoch_results["mee_tr"])
            self.tr_mse_history.append(tr_epoch_results["mse_tr"])

            if self.validation:
                
                # Al suo interno viene presa la media del risultato
                # su tutti i pattern
                vl_epoch_results = self._run_epoch_vl(vl_x, vl_d, metric_fun)

                self.vl_mee_history.append(vl_epoch_results["mee_vl"])
                self.vl_mse_history.append(vl_epoch_results["mse_vl"])

                vl_score = vl_epoch_results["vl_score"] # <- Ciò che vede l'early stopping

                if self.early_stopping:
                    should_stop = stopper(vl_score, self.neuraln.weights_matrix_list)

                    if should_stop: 
                        break


            if self.verbose and epoch % 10 == 0:
                print("|| epooch n° ", epoch, ", total mee error: ", tr_epoch_results["mee_tr"], " ||")

        if self.verbose: 
            print(" Final mee error: ", tr_epoch_results["mee_tr"])
            print(" Final mse error: ", tr_epoch_results["mse_tr"])
            print(" Tempo di fitting: ", time.perf_counter() - start_time)
            print("\n--- Fitting Completato ---\n")

            if self.validation: 
                plot_errors_with_validation_error(self, time.perf_counter() - start_time)
            else:
                plot_errors(self, time.perf_counter() - start_time)


        return (self.tr_mee_history[-1], self.tr_mse_history[-1], 
                self.vl_mee_history[-1] if self.validation else 0.0,
                self.vl_mse_history[-1] if self.validation else 0.0) 
    

    def _run_epoch(self, 
                   input_matrix: np.ndarray, 
                   d_matrix: np.ndarray, 
                   n_patterns: int) -> Dict[str, float]:
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

            x_pattern, d_pattern = input_matrix[idx], d_matrix[idx]

            layer_results = self.neuraln.forward_network(x_pattern)
            final_output = layer_results[-1]

            epoch_mee += (np.sum((d_pattern - final_output) ** 2)) ** 0.5
            epoch_mse += (np.sum((d_pattern - final_output) ** 2))

            # L'asterisco serve per raggruppare in lista tutti i risultati
            # tranne, in questo caso, l'ultimo. Essendo specificato.
            # quindi deltas = [dwk, dwj2, dwj1], nel caso di una rete con 2 hidden layer

            deltas, grad_norm = compute_delta_all_layers_list(
                            d = d_pattern,
                            layer_results_list = layer_results,
                            weights_matrix_list = self.neuraln.weights_matrix_list,
                            x_pattern = x_pattern,
                            df_act = self.f_act, # Nota: La funzione backprop gestirà la derivata internamente
                            old_deltas = self.old_deltas if self.momentum else None,
                            alpha_momentum = self.alpha_mom,
                            max_norm_gradient_for_clipping = self.max_gradient_norm
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
                self.neuraln.update_weights(deltas, eta=self.learning_rate)
                if self.momentum: self.old_deltas = deltas

        # Se fine epoca e se batch, aggiorna gli update weights 
        if self.batch:
            # Media dei gradienti
            avg_deltas = [d_mat / n_patterns for d_mat in batch_deltas]

            if self.use_decay and self.epoch > 0 and self.epoch % self.decay_step == 0:
                self.learning_rate *= self.decay_factor
            
            self.neuraln.update_weights(avg_deltas, eta = self.learning_rate)
            # Salva per momentum prossima epoca
            if self.momentum: self.old_deltas = avg_deltas

        return {
            'mee_tr': epoch_mee / n_patterns,
            'mse_tr': epoch_mse / n_patterns,
            'grad_norm': epoch_grad / n_patterns
        }
    

    def _run_epoch_vl(self, vl_x, vl_d, metric_fn: Callable):
        """
        Metodo che nasce con l'esigenza di portare un po' di logica fuori dal train,
        runna una epoca, restitutuendo le informazioni sull'errore.
        Questo metodo gestisce la divisione della logica in base all'esplorazione tramite batch o online.
        Argomenti:
            - vl_x = lista di matrici (split, n_input)
            - vl_d = lista di matrici (split, n_output)
        """
        n_patterns = vl_x.shape[0]
        epoch_mee_vl, epoch_mse_vl  = 0.0, 0.0
        correct_predictions = 0

        # Scorre tutti gli indici dividendo il comportamento in base a se è batch oppure online
        for pattern in range(n_patterns):

            vl_layer_results = self.neuraln.forward_network(vl_x[pattern])

            vl_final_output = vl_layer_results[-1]

            if metric_fn == compute_accuracy:
                """
                Siamo nel caso del MONK, abbiamo:
                il final output: float da convertire in 0 o 1 
                e d: che ha valore 1 o 0
                """

                vl_score = compute_accuracy()
                (vl_final_output, d): correct_predictions += 1

            epoch_mee_vl += mean_euclidean_error(vl_final_output, vl_d)
            epoch_mse_vl += mean_squared_error(vl_final_output, vl_d)

            # L'asterisco serve per raggruppare in lista tutti i risultati
            # tranne, in questo caso, l'ultimo. Essendo specificato.
            # quindi deltas = [dwk, dwj2, dwj1], nel caso di una rete con 2 hidden layer                    

        return {
            'vl_score': vl_score,
            'mee_vl': epoch_mee_vl,
            'mse_vl': epoch_mse_vl,
        }
