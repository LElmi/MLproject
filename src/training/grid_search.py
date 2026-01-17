import itertools
from src.training.trainer.trainer import Trainer


class GridSearch:
    """
    Creata per gestire il caso esaustivo e caso intervalli
    """
    def __init__(self, **kwargs):
        # **kwargs contiene tutti gli argomenti passati come lista di liste

        self.params_to_explore = kwargs
        # Risultato da restituire con un getter
        self.best_config = None
        self.best_mee = float('inf')
        self.best_accuracy = 0
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
        Metodo che si chiama dall'esterno, questo è il cuore 
        del grid search, 
        """
        for i, config_dict in enumerate(self.combinations):
            # Qui istanziamo il Trainer usando lo spacchettamento del dizionario **
            trainer = Trainer(
                input_size=x_train.shape[1],
                **config_dict 
            )
            
            # Per ogni combinazione chiama il fit
            current_mee_tr, current_mse_tr, current_mee_vl, current_mse_vl, accuracy = trainer.fit(x_train, d_train, vl_input, vl_targets,metric_fn)

            print(f"Config {i+1}/{len(self.combinations)} | MEE in training: {current_mee_tr:.4f}")
            print(f"Config {i + 1}/{len(self.combinations)} | MEE in Validation: {current_mee_vl:.4f}")
            print(f"Config {i + 1}/{len(self.combinations)} | Accuracy in Validation: {accuracy:.4f}")
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_config = config_dict
        
        return self.best_config, self.best_accuracy

    def run_for_cup(self, x_train, d_train, vl_input, vl_targets, metric_fn):
        """
        Metodo che si chiama dall'esterno, questo è il cuore
        del grid search,
        """
        for i, config_dict in enumerate(self.combinations):
            # Qui istanziamo il Trainer usando lo spacchettamento del dizionario **
            trainer = Trainer(
                input_size=x_train.shape[1],
                **config_dict
            )

            #  Per ogni combinazione chiama il fit
            current_mee_tr, current_mse_tr, current_mee_vl, current_mse_vl, accuracy = trainer.fit(x_train, d_train,
                                                                                                   vl_input, vl_targets,
                                                                                                   metric_fn)

            print(f"Config {i + 1}/{len(self.combinations)} | MEE in training: {current_mee_tr:.4f}")
            print(f"Config {i + 1}/{len(self.combinations)} | MEE in Validation: {current_mee_vl:.4f}")
            if current_mee_vl < self.best_mee:
                self.best_mee = current_mee_vl
                self.best_config = config_dict

        return self.best_config, self.best_mee