import itertools
from src.training.trainer.trainer_cup import TrainerCup
from src.training.trainer.trainer_monk import TrainerMonk
from src.utils.visualization import plot_grid_analysis
from src.training.validation.k_fold import run_k_fold_cup



class GridSearch:
    """
    Creata per gestire il caso esaustivo e caso intervalli
    """
    def __init__(self, **kwargs):
        # **kwargs contiene tutti gli argomenti passati come lista di liste

        self.params_to_explore = kwargs
        # Risultato da restituire con un getter
        self.best_config = None
        self.best_mse = float('inf')
        # self.best_accuracy = 0
        self.combinations = self._generate_combinations()

    def _generate_combinations(self):
        """
        Metodo che scompone la lista di liste in chiavi e valori e fa
        il prodotto cartesiano esaustivo
        """
        # Ad esempio keys = ['units', 'lr']
        keys = self.params_to_explore.keys()
        # Ad esempio values = [ [[32], [64]], [0.1, 0.01] ]
        values = self.params_to_explore.values()

        all_configs = []
        for combination in itertools.product(*values):
            # Crea una combinazione di valori MANTENENDO l'ordine
            # quindi zippo
            comb = dict(zip(keys, combination))
            all_configs.append(comb)
        
        return all_configs
    

    def run_for_monk(self, x_train, d_train,vl_input, vl_targets, metric_fn):
        """
        Da completare...

        Metodo che si chiama dall'esterno, questo è il cuore 
        del grid search, 
        """
        for i, config_dict in enumerate(self.combinations):
            # Qui istanziamo il Trainer usando lo spacchettamento del dizionario **
            trainer = TrainerMonk(
                input_size=x_train.shape[1],
                **config_dict 
            )
            
            # Per ogni combinazione chiama il fit
            current_mse_tr = trainer.fit(x_train, d_train, vl_input, vl_targets,metric_fn)

            print(f"Config {i+1}/{len(self.combinations)} | MSE in training: {current_mse_tr:.4f}")
            #print(f"Config {i + 1}/{len(self.combinations)} | Accuracy in Validation: {accuracy:.4f}")
            #if accuracy > self.best_accuracy:
            #    self.best_accuracy = accuracy
            #    self.best_config = config_dict
        
        return self.best_config, self.best_accuracy

    def run_for_cup(self, x_train, d_train, vl_input, vl_targets):
        """
        Metodo che si chiama dall'esterno, questo è il cuore
        del grid search
        """

        all_results = []

        for i, config_dict in enumerate(self.combinations):
            # Qui istanziamo il Trainer usando lo spacchettamento del dizionario **
            trainer = TrainerCup(
                input_size=x_train.shape[1],
                **config_dict
            )

            #  Per ogni combinazione chiama il fit
            current_mee_tr, current_mse_tr, current_mee_vl, current_mse_vl = trainer.fit(x_train, d_train,
                                                                                        vl_input, vl_targets)

            print(f"Config {i + 1}/{len(self.combinations)} | MSE in training: {current_mse_tr:.4f}")
            print(f"Config {i + 1}/{len(self.combinations)} | MSE in Validation: {current_mse_vl:.4f}")

            if current_mse_vl < self.best_mse:
                self.best_mse = current_mse_vl
                self.best_config = config_dict

            result_entry = {
                'params': config_dict,
                'tr_mse': trainer.tr_mse_history, # Lista MSE train per ogni epoca
                'vl_mse': trainer.vl_mse_history, # Lista MSE val per ogni epoca
                # Se volessi plottare anche il MEE in futuro, puoi salvarlo qui:
                # 'tr_mee': trainer.tr_mee_history,
                # 'vl_mee': trainer.vl_mee_history
            }
            all_results.append(result_entry)

        plot_grid_analysis(
                    all_results, 
                    top_k_individual=5,              # Salva i dettagli delle migliori 5 configurazioni
                    relative_path="results/cup/grid_search"
                )

        return self.best_config, self.best_mse
    

    
    def run_for_cup_with_kfold(self, x_full, d_full, k_folds=4):
            """
            Model Selection robusta: K-Fold su ogni configurazione.
            """
            print(f"🚀 Inizio Grid Search con K-Fold (K={k_folds})...")

            all_results = [] # Lista che passeremo al plotter

            for i, config_dict in enumerate(self.combinations):
                
                # Chiamiamo la funzione aggiornata sopra
                stats = run_k_fold_cup(
                    x_full=x_full,
                    d_full=d_full,
                    k_folds=k_folds,
                    model_config=config_dict,
                    verbose=False 
                )
                
                mean_mse_val = stats['mean_mse'] 
                
                print(f"Config {i+1}/{len(self.combinations)} | Mean MSE: {mean_mse_val:.5f}")

                # Aggiorna il best model
                if mean_mse_val < self.best_mse:
                    self.best_mse = mean_mse_val
                    self.best_config = config_dict
                    print(f"   ⭐️ New Best Found!")
                
                # --- COSTRUZIONE DATI PER IL PLOT ---
                result_entry = {
                    'params': config_dict,
                    'tr_mse': stats['mean_tr_history'], # Usiamo la curva media calcolata
                    'vl_mse': stats['mean_vl_history']  # Usiamo la curva media calcolata
                }
                all_results.append(result_entry)

            # Chiamata al plotter FUORI dal ciclo for
            print("Generazione grafici...")
            plot_grid_analysis(all_results, top_k_individual=5, relative_path="results/cup/grid_kfold")

            return self.best_config, self.best_mse