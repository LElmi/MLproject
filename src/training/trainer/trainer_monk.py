from src.training.trainer.trainer import Trainer
import time
from src.utils.visualization import plot_monk




class TrainerMonk(Trainer):
    """
    Sottoclasse di Trainer.
    Mantiene traccia dei parametri necessari al:
        - Train del monk
        - Test set ad ogni epoca
        - Confusion Matrix
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tr_accuracy_history = []
        self.ts_accuracy_history = []


    def fit(self, tr_x, tr_d, ts_x = None, ts_d = None, metric_mode='max'):
        """
        Il fit del monk 
        """


        n_patterns = tr_x.shape[0]
        start_time = time.perf_counter()
        
        # ! AVVIO CICLO DI EPOCHE ! #
        for epoch in range(1, self.epochs + 1):
            
            self.epoch = epoch
            matched_tr = 0
            matched_ts = 0

            tr_epoch_results = self._run_epoch(tr_x, tr_d, n_patterns)
            self.tr_mse_history.append(tr_epoch_results["mse_tr"])


            # Analizza l'accuracy sul training set
            for i, tr_x_pattern in enumerate(tr_x):
                # Forward pass
                
                layer_results_tr, _ = self.neuraln.forward_network(tr_x_pattern, self.f_act_hidden, self.f_act_output)
                output = layer_results_tr[-1]  
                prediction = 1.0 if output >= 0.5 else 0.0  
                
                if prediction == tr_d[i]:
                    matched_tr += 1

            self.tr_accuracy_history.append(matched_tr / tr_x.shape[0])

            if ts_x is not None and ts_d is not None:

                for i, ts_x_pattern in enumerate(ts_x):
                    # Forward pass
                    
                    layer_results, _ = self.neuraln.forward_network(ts_x_pattern, self.f_act_hidden, self.f_act_output)
                    output = layer_results[-1]  
                    prediction = 1.0 if output >= 0.5 else 0.0  
                    
                    if prediction == ts_d[i]:
                        matched_ts += 1

            self.ts_accuracy_history.append(matched_ts / ts_x.shape[0])


            if self.verbose and epoch % 10 == 0:
                print("||[MONK] fine epoch n° ", epoch, ", mse error: ", tr_epoch_results["mse_tr"], " ||")


        if self.verbose: 
            training_time = time.perf_counter() - start_time
            print(" Final mse error: ", tr_epoch_results["mse_tr"])
            print(" Accuracy finale su training set: ", self.tr_accuracy_history[-1]*100,"%")
            print(" Accuracy finale su test set: ", self.ts_accuracy_history[-1]*100,"%")
            print(" Tempo di fitting: ", training_time)
            print("\n--- Fitting Completato ---\n")

            plot_monk(self.tr_mse_history, self.tr_accuracy_history, self.ts_accuracy_history, training_time)

        return (self.tr_mse_history[-1])
                #self.vl_mee_history[-1] if self.validation else 0.0,
                #self.vl_mse_history[-1] if self.validation else 0.0,
                #final_vl_accuracy if self.validation else 0.0)



"""Lascio qui nel caso volessimo inserire il validation nel monk


    def _run_epoch_vl(self, vl_x, vl_d, metric_fn: Callable = None):
        n_patterns = vl_x.shape[0]
        #epoch_mee_vl, epoch_mse_vl = [], []
        correct_predictions = 0
        vl_final_output_array = []
        
        for pattern in range(n_patterns):

            if metric_fn == compute_accuracy:

                vl_layer_results, _ = self.neuraln.forward_network(vl_x[pattern],self.f_act_hidden, sigmaf)
            else : 
                vl_layer_results, _ = self.neuraln.forward_network(vl_x[pattern],self.f_act_hidden,linear)

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
            print(f"|||| CLASSIFICATION ACCURACY: {accuracy_vl:.4f} ({correct_predictions}/{n_patterns})")
        
        return {
            'vl_score': vl_score,
            'mee_vl': epoch_mee_vl,
            'mse_vl': epoch_mse_vl,
            'accuracy_vl': accuracy_vl
        }
"""