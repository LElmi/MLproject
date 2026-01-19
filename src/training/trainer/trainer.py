# MAIN PROGETTO ML

import numpy as np
import time
from typing import Callable, Dict
from src.nn.nn import NN
from src.training.trainer.stopper import EarlyStopper
from src.training.trainer.forward.forward_pass import *
from src.training.trainer.backward.backprop import compute_delta_all_layers_list
from src.utils import *
from src.utils.compute_accuracy import compute_accuracy
# from src.training.trainer.stopper import Stopper
from src.activationf.sigmoid import sigmaf
from src.activationf.linear import linear
#from src.utils.visualization import plot_accuracy
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
                 f_act_hidden: Callable,
                 f_act_output: Callable,
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
                 validation: bool = False,
                 lambdal2: float =1e-4):  # <- Importante da togliere nella grid search


        self.f_act_hidden = f_act_hidden
        self.f_act_output = f_act_output
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
        self.lambdal2=lambdal2
        self.epoch = 0
        self.old_deltas = None

        # Inizializza la rete neurale
        self.neuraln = NN(
                    n_inputs = input_size, 
                    units_list = units_list, 
                    n_outputs = n_outputs, 
                    f_act_hidden = f_act_hidden,
                    f_act_output = f_act_output
                )
                
        self.tr_mee_history = [] 
        self.tr_mse_history = []

        self.vl_mee_history = []
        self.vl_mse_history = []

        self.accuracy_history = []

    def fit_k_fold(self,
                   input_matrix: np.ndarray,
                   d_matrix: np.ndarray,
                   fold: int,
                   vl_input: np.ndarray = None,
                   vl_targets: np.ndarray = None,
                   metric_fn: Callable = None,  # Aggiunto per gestire Accuracy
                   metric_mode: str = 'min'):  # Aggiunto per Early Stopper

        n_patterns = input_matrix.shape[0]
        start_time = time.perf_counter()

        # Inizializza lo stopper come nel metodo fit standard
        stopper = EarlyStopper(
            patience=self.patience,
            min_delta=self.epsilon,
            mode=metric_mode
        )

        final_vl_accuracy = 0.0  # Valore di default

        if self.verbose:
            print(f"--- Inizio Training Fold {fold} (Early stopping: {self.early_stopping}) ---")

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch

            # CORREZIONE: Rimosso il 4° argomento 'epoch' che causava TypeError
            epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns)

            self.tr_mee_history.append(epoch_results["mee_tr"])
            self.tr_mse_history.append(epoch_results["mse_tr"])

            # Gestione Validation
            if self.validation and vl_input is not None:
                # Passiamo metric_fn per calcolare accuracy se serve
                if metric_fn is compute_accuracy:
                    epoch_vl_results = self._run_epoch_vl(vl_input, vl_targets, metric_fn)
                else:
                    epoch_vl_results = self._run_epoch_vl_CUP(vl_input, vl_targets, metric_fn)
                self.vl_mee_history.append(epoch_vl_results["mee_vl"])
                self.vl_mse_history.append(epoch_vl_results["mse_vl"])

                # Salviamo accuracy se presente
                final_vl_accuracy = epoch_vl_results.get("accuracy_vl", 0.0)
                if final_vl_accuracy > 0:
                    self.accuracy_history.append(final_vl_accuracy)

                # Early Stopping Logic (Coerente con fit)
                vl_score = epoch_vl_results["vl_score"]

                if self.early_stopping:
                    if stopper(vl_score, self.neuraln.weights_matrix_list):
                        if self.verbose:
                            print(f"Early Stopping al fold {fold}, epoca {epoch}")
                        # Ripristina i pesi migliori
                        self.neuraln.weights_matrix_list = stopper.best_weights
                        break

            if self.verbose and epoch % 10 == 0:
                print(f"|| Fold {fold} | Epoch {epoch} | TR MEE: {epoch_results['mee_tr']:.4f} ||")
            if (epoch == self.epochs):
                print("\n\n\n\nFine Training all'epoca:", epoch)

        # Salvataggio modello a fine training (o se early stopping attivato)
        # Nota: save_model deve essere importata da src.utils
        if self.verbose: print(f"Salvataggio modello per fold {fold}...")
        save_model(self, fold)

        if self.verbose:
            print(f"\n--- Training Fold {fold} Completato in {time.perf_counter() - start_time:.2f}s ---\n")
        if self.validation and self.validation:
            plot_errors_with_validation_error(self, time.perf_counter() - start_time)

        elif self.validation and metric_fn == compute_accuracy:
            plot_accuracy(self, time.perf_counter() - start_time)
        else:
            plot_errors(self, time.perf_counter() - start_time)
        # RETURN STANDARD A 5 VALORI (Come fit)
        return (self.tr_mee_history[-1],
                self.tr_mse_history[-1],
                self.vl_mee_history[-1] if self.validation else 0.0,
                self.vl_mse_history[-1] if self.validation else 0.0,
                final_vl_accuracy)

    def fit(self, 
            tr_x: np.ndarray, 
            tr_d: np.ndarray,
            vl_x: np.ndarray = None,
            vl_d: np.ndarray = None,
            metric_fn: Callable = None,
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
        stopper = EarlyStopper(
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
                if metric_fn is compute_accuracy:
                    vl_epoch_results = self._run_epoch_vl(vl_x, vl_d, metric_fn)
                else:
                    vl_epoch_results = self._run_epoch_vl_CUP(vl_x, vl_d, metric_fn)

                self.vl_mee_history.append(vl_epoch_results["mee_vl"])
                self.vl_mse_history.append(vl_epoch_results["mse_vl"])

                final_vl_accuracy = vl_epoch_results.get("accuracy_vl", 0.0)

                vl_score = vl_epoch_results["vl_score"]
                
                if self.early_stopping:
                    should_stop = stopper(vl_score, self.neuraln.weights_matrix_list)

                    if should_stop:
                        break
                    

            if self.verbose and epoch % 10 == 0:
                print("|| epooch n° ", epoch, ", total mee error with training patterns: ", tr_epoch_results["mee_tr"], " ||")

        if self.verbose: 
            print(" Final mee error: ", tr_epoch_results["mee_tr"])
            print(" Final mse error: ", tr_epoch_results["mse_tr"])
            print(" Tempo di fitting: ", time.perf_counter() - start_time)
            print("\n--- Fitting Completato ---\n")

            if self.validation: 
                plot_errors_with_validation_error(self, time.perf_counter() - start_time)
            else:
                plot_errors(self, time.perf_counter() - start_time)

        return (self.tr_mee_history[-1], 
                        self.tr_mse_history[-1], 
                        self.vl_mee_history[-1] if self.validation else 0.0,
                        self.vl_mse_history[-1] if self.validation else 0.0,)
                        #final_vl_accuracy)

    def _run_epoch(self,
                   input_matrix: np.ndarray,
                   d_matrix: np.ndarray,
                   n_patterns: int) -> Dict[str, float]:

        indices = np.arange(n_patterns)
        if not self.batch:
            np.random.shuffle(indices)

        epoch_grad = 0.0
        final_output = []
#triggera online solo se pattern_in_batch==pattern totali
        batch_percentage = getattr(self, "batch_percentage", 0.1)
        #patterns_in_batch = max(1, int(round(n_patterns * batch_percentage)))
        patterns_in_batch=64
        # ---------------------------------batch
        if self.batch:
            batch_deltas = [np.zeros_like(w) for w in self.neuraln.weights_matrix_list]

            for idx in indices:
                x_pattern = input_matrix[idx]
                d_pattern = d_matrix[idx]

                layer_results, layer_nets = self.neuraln.forward_network(
                    x_pattern,
                    self.f_act_hidden,
                    self.f_act_output
                )
                final_output.append(layer_results[-1])

                deltas, grad_norm = compute_delta_all_layers_list(
                    d=d_pattern,
                    layer_results_list=layer_results,
                    layer_net_list=layer_nets,
                    weights_matrix_list=self.neuraln.weights_matrix_list,
                    x_pattern=x_pattern,
                    f_act_hidden=self.f_act_hidden,
                    f_act_output=self.f_act_output,
                    old_deltas=self.old_deltas if self.momentum else None,
                    alpha_momentum=self.alpha_mom,
                    max_norm_gradient_for_clipping=self.max_gradient_norm
                )

                epoch_grad += grad_norm
                for i in range(len(batch_deltas)):
                    batch_deltas[i] += deltas[i]

            avg_batch_deltas = [d_mat / n_patterns for d_mat in batch_deltas]
            self.neuraln.update_weights(
                avg_batch_deltas,
                eta=self.learning_rate,
                lambda_l2=self.lambdal2
            )

            if self.momentum:
                self.old_deltas = avg_batch_deltas

        # ----------------------minibatch
        elif patterns_in_batch < n_patterns:
            all_minibatches = [
                indices[i:i + patterns_in_batch]
                for i in range(0, n_patterns, patterns_in_batch)
            ]
            for minibatch in all_minibatches:
                minibatch_deltas = [np.zeros_like(w) for w in self.neuraln.weights_matrix_list]
                minibatch_grad = 0

                for idx in minibatch:
                    x_pattern = input_matrix[idx]
                    d_pattern = d_matrix[idx]

                    layer_results, layer_nets = self.neuraln.forward_network(
                        x_pattern,
                        self.f_act_hidden,
                        self.f_act_output
                    )
                    final_output.append(layer_results[-1])

                    deltas, grad_norm = compute_delta_all_layers_list(
                        d=d_pattern,
                        layer_results_list=layer_results,
                        layer_net_list=layer_nets,
                        weights_matrix_list=self.neuraln.weights_matrix_list,
                        x_pattern=x_pattern,
                        f_act_hidden=self.f_act_hidden,
                        f_act_output=self.f_act_output,
                        old_deltas=self.old_deltas if self.momentum else None,
                        alpha_momentum=self.alpha_mom,
                        max_norm_gradient_for_clipping=self.max_gradient_norm
                    )

                    minibatch_grad += grad_norm
                    for i in range(len(minibatch_deltas)):
                        minibatch_deltas[i] += deltas[i]

                # MEDIA
                avg_minibatch_deltas = [d_mat / len(minibatch) for d_mat in minibatch_deltas]
                minibatch_grad /= len(minibatch)

                epoch_grad += minibatch_grad

                self.neuraln.update_weights(
                    avg_minibatch_deltas,
                    eta=self.learning_rate,
                    lambda_l2=self.lambdal2
                )

                if self.momentum:
                    self.old_deltas = avg_minibatch_deltas

        #---------------------------------Online
        else:
            for idx in indices:
                x_pattern = input_matrix[idx]
                d_pattern = d_matrix[idx]

                layer_results, layer_nets = self.neuraln.forward_network(
                    x_pattern,
                    self.f_act_hidden,
                    self.f_act_output
                )
                final_output.append(layer_results[-1])

                deltas, grad_norm = compute_delta_all_layers_list(
                    d=d_pattern,
                    layer_results_list=layer_results,
                    layer_net_list=layer_nets,
                    weights_matrix_list=self.neuraln.weights_matrix_list,
                    x_pattern=x_pattern,
                    f_act_hidden=self.f_act_hidden,
                    f_act_output=self.f_act_output,
                    old_deltas=self.old_deltas if self.momentum else None,
                    alpha_momentum=self.alpha_mom,
                    max_norm_gradient_for_clipping=self.max_gradient_norm
                )

                epoch_grad += grad_norm

                self.neuraln.update_weights(
                    deltas,
                    eta=self.learning_rate,
                    lambda_l2=self.lambdal2
                )

                if self.momentum:
                    self.old_deltas = deltas

        epoch_mee = mean_euclidean_error(final_output, d_matrix)
        epoch_mse = mean_squared_error(final_output, d_matrix)

        return {
            "mee_tr": epoch_mee,
            "mse_tr": epoch_mse,
            "grad": epoch_grad
        }

        return {
            'mee_tr': epoch_mee,
            'mse_tr': epoch_mse,
            'grad_norm': epoch_grad / n_patterns
        }

    def _run_epoch_vl(self, vl_x, vl_d, metric_fn: Callable = None):
        n_patterns = vl_x.shape[0]
        #epoch_mee_vl, epoch_mse_vl = [], []
        correct_predictions = 0
        vl_final_output_array = []
        
        for pattern in range(vl_x.shape[0]):

            vl_layer_results = self.neuraln.forward_network(vl_x[pattern],self.f_act_hidden, sigmaf)

            vl_final_output = vl_layer_results[-1]

            vl_final_output_array.append(vl_final_output.tolist())

            # Calcola accuracy per monitoring
            if metric_fn == compute_accuracy:
                if compute_accuracy(vl_final_output, vl_d[pattern]):
                    correct_predictions += 1
        epoch_mee_vl = mean_euclidean_error(vl_final_output_array, vl_d)
        epoch_mse_vl = mean_squared_error(vl_final_output_array, vl_d)
        
        # IMPORTANTE: calcola le medie
        #avg_mee_vl = epoch_mee_vl / n_patterns
        #avg_mse_vl = epoch_mse_vl / n_patterns
        accuracy_vl = correct_predictions / n_patterns
        if metric_fn is not compute_accuracy:
            vl_score = epoch_mee_vl
        else:
        # Early stopping su MSE medio (non cumulativo!)
            vl_score = epoch_mse_vl
        
        if self.verbose:

            rounded_numbers = [round(num,1) for num in np.asarray(vl_final_output_array).flatten()]
            print(f"|||| CLASSIFICATION ACCURACY: {accuracy_vl:.4f} ({correct_predictions}/{n_patterns})","\noutput modello: ",np.asarray(rounded_numbers).flatten(),"\ntargets: ",np.asarray(vl_d).flatten())
        
        return {
            'vl_score': vl_score,
            'mee_vl': epoch_mee_vl,
            'mse_vl': epoch_mse_vl,
            'accuracy_vl': accuracy_vl
        }

    def _run_epoch_vl_CUP(self, vl_x, vl_d, metric_fn: Callable = None):
        n_patterns = vl_x.shape[0]
        # epoch_mee_vl, epoch_mse_vl = [], []
        vl_final_output_array = []

        for pattern in range(n_patterns):

            if metric_fn == compute_accuracy:

                vl_layer_results = self.neuraln.forward_network(vl_x[pattern], self.f_act_hidden, sigmaf)
            else:
                vl_layer_results = self.neuraln.forward_network(vl_x[pattern], self.f_act_hidden, linear)

            vl_final_output = vl_layer_results[-1]

            vl_final_output_array.append(vl_final_output.tolist())

            # Calcola accuracy per monitoring
        epoch_mee_vl = mean_euclidean_error(vl_final_output_array, vl_d)
        epoch_mse_vl = mean_squared_error(vl_final_output_array, vl_d)

        # IMPORTANTE: calcola le medie
        # avg_mee_vl = epoch_mee_vl / n_patterns
        # avg_mse_vl = epoch_mse_vl / n_patterns
        if metric_fn is not compute_accuracy:
            vl_score = epoch_mee_vl
        else:
            # Early stopping su MSE medio (non cumulativo!)
            vl_score = epoch_mse_vl


        return {
            'vl_score': vl_score,
            'mee_vl': epoch_mee_vl,
            'mse_vl': epoch_mse_vl,
            'accuracy_vl': 0.0
        }