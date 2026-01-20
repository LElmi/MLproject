from src.training.trainer.trainer import Trainer
from src.training.trainer.stopper import EarlyStopper
from typing import Callable
from src.utils.visualization import plot_errors_with_validation_error, plot_errors
from src.utils import *

import numpy as np
import time



class TrainerCup(Trainer):
    """
    Sottoclasse di Trainer.
    Mantiene traccia dei parametri necessari al:
        - Train della cup (MSE)
        - Validation (K-FOLD)
        - Early Stopping su validation error
    """

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.vl_mee_history = []
        self.vl_mse_history = []

    def fit(self, 
            tr_x: np.ndarray, 
            tr_d: np.ndarray,
            vl_x: np.ndarray = None,
            vl_d: np.ndarray = None,
            fold_id: int = None,
            metric_fn: Callable = None):

        n_patterns = tr_x.shape[0]
        start_time = time.perf_counter()

        # Gestisce la logica dell'early stopping
        # e si assicura di salvare la matrice dei pesi
        # appena capisce che si deve fermare, l'early stopping
        # per adesso è attiva solo nel caso sia attiva la validation
        should_stop = EarlyStopper(
            patience = self.patience,
            min_delta = self.epsilon,
            mode="min"
        )
        
        # ! AVVIO CICLO DI EPOCHE ! #
        for epoch in range(1, self.epochs + 1):
            
            self.epoch = epoch 
            tr_epoch_results = self._run_epoch(tr_x, tr_d, n_patterns)

            self.tr_mee_history.append(tr_epoch_results["mee_tr"])
            self.tr_mse_history.append(tr_epoch_results["mse_tr"])


            if self.validation and vl_x is not None:
                
                # Al suo interno viene presa la media del risultato
                # su tutti i pattern
                vl_epoch_results = self._run_epoch_vl(vl_x, vl_d, metric_fn)

                self.vl_mee_history.append(vl_epoch_results["mee_vl"])
                self.vl_mse_history.append(vl_epoch_results["mse_vl"])

                vl_score = vl_epoch_results["vl_score"]
                
                # Se should stop è vero allora l'Early Stopper ha superato la patience
                if should_stop(vl_score, self.neuraln.weights_matrix_list):
                    break
                    

            if self.verbose and epoch % 15 == 0:
                print("||[CUP] epoch n° ", epoch, ", total mse error with training patterns: ", tr_epoch_results["mse_tr"], " ||")

        #if fold_id is not None:
            #save_model(self, fold_id)

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
                self.vl_mse_history[-1] if self.validation else 0.0)
                #final_vl_accuracy if self.validation else 0.0)


    def _run_epoch_vl(self, vl_x, vl_d, metric_fn: Callable = None):
        n_patterns = vl_x.shape[0]
        # epoch_mee_vl, epoch_mse_vl = [], []
        vl_final_output_array = []

        for pattern in range(n_patterns):
            vl_layer_results, _ = self.neuraln.forward_network(vl_x[pattern], self.f_act_hidden, self.f_act_output)
            vl_final_output_array.append(vl_layer_results[-1].tolist())

            # Calcola accuracy per monitoring
        epoch_mee_vl = mean_euclidean_error(vl_final_output_array, vl_d)
        epoch_mse_vl = mean_squared_error(vl_final_output_array, vl_d)
        vl_score = epoch_mse_vl


        return {
            'vl_score': vl_score,
            'mee_vl': epoch_mee_vl,
            'mse_vl': epoch_mse_vl
            #'accuracy_vl': 0.0
        }
    
"""    def fit_k_fold(self,
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
                final_vl_accuracy)"""