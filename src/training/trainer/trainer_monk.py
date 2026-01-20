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


    def fit(self, tr_x, tr_d, ts_x = None, ts_d = None):
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