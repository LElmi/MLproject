import numpy as np
from src.training.trainer.trainer_cup import TrainerCup

def k_fold_split(x_i, d, folds, validation_fold):

    # Divide l'array in "folds" parti uguali
    x_i_splitted = np.array_split(x_i, folds)
    d_splitted   = np.array_split(d, folds)

    vl_input   = x_i_splitted[validation_fold]
    vl_targets = d_splitted[validation_fold]

    # Il training set diventa la concatenazione di tutte le parti
    tr_input = np.concatenate(
        [sub for i, sub in enumerate(x_i_splitted) if i != validation_fold]
    )

    tr_targets = np.concatenate(
        [sub for i, sub in enumerate(d_splitted) if i != validation_fold]
    )

    return tr_input, tr_targets, vl_input, vl_targets


def run_k_fold_cup(x_full, d_full, k_folds, model_config, verbose=True):
    """
    Esegue K-Fold e restituisce:
    - Statistiche Scalari: Mean MSE, Mean MEE (con std dev)
    - Curve Medie: Per i grafici (solo MSE, o anche MEE se vuoi)
    """
    
    mse_vals = []
    mee_vals = [] 
    
    all_tr_mse_histories = [] 
    all_vl_mse_histories = []

    for i in range(k_folds):
        if verbose: print(f"--- Fold {i+1}/{k_folds} ---")
            
        tr_input, tr_target, vl_input, vl_target = k_fold_split(x_full, d_full, k_folds, i)
        
        trainer = TrainerCup(input_size=tr_input.shape[1], **model_config)
        
        # Fit restituisce i valori finali, ma noi accediamo alle liste interne per precisione
        trainer.fit(
            tr_x=tr_input, tr_d=tr_target, 
            vl_x=vl_input, vl_d=vl_target, 
            fold_id=i if verbose else None
        )
        
        mse_vals.append(min(trainer.vl_mse_history))
        mee_vals.append(min(trainer.vl_mee_history)) 
        
        all_tr_mse_histories.append(trainer.tr_mse_history)
        all_vl_mse_histories.append(trainer.vl_mse_history)

    min_len = min(len(h) for h in all_tr_mse_histories)
    
    tr_cut = [h[:min_len] for h in all_tr_mse_histories]
    vl_cut = [h[:min_len] for h in all_vl_mse_histories]
    
    mean_tr_curve = np.mean(tr_cut, axis=0).tolist()
    mean_vl_curve = np.mean(vl_cut, axis=0).tolist()

    return {
        "mean_mse": np.mean(mse_vals),
        "std_mse": np.std(mse_vals),
        
        "mean_mee": np.mean(mee_vals), 
        "std_mee": np.std(mee_vals),
        
        "mean_tr_history": mean_tr_curve,
        "mean_vl_history": mean_vl_curve
    }