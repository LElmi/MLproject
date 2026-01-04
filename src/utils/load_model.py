import json
import numpy as np

def load_model(filepath):
    """
    Carica il modello da un unico file json
    Ritorna: weights_list (lista di matrici), architecture (dict)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    architecture = data['architecture']
    
    weights_list = [np.array(w) for w in data['weights']]
    
    print(f"Modello caricato: {len(weights_list)} layer di pesi trovati.")
    return weights_list, architecture