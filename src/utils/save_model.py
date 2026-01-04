import json
import os
from datetime import datetime

def save_model(weights_list, architecture_info, folder="../results/models"):
    """
    Salva i pesi e l'architettura.
    weights_list: lista di matrici (np.ndarray)
    architecture_info: dizionario con i dettagli della rete
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    serializable_weights = [w.tolist() for w in weights_list]
    
    model_data = {
        'architecture': architecture_info,
        'weights': serializable_weights
    }

    filename = f"{folder}/model_{time_str}.json"
    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=4)
    
    print(f"Modello salvato correttamente in: {filename}")
    return filename