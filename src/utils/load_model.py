import json
import numpy as np

def load_model(weights_filename, architecture_filename):
    # Carica i pesi dal file di testo
    loaded_weights = {}
    with open(weights_filename, 'r') as f:
        lines = f.readlines()
        current_layer = None
        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                # Ogni nuova sezione di pesi inizia con 'nome_layer:'
                current_layer = line[:-1]
                loaded_weights[current_layer] = []
            elif line:
                # Converti la riga di testo in numeri float
                loaded_weights[current_layer].append(list(map(float, line.split(','))))

    # Converti le liste in array NumPy
    for layer in loaded_weights:
        loaded_weights[layer] = np.array(loaded_weights[layer])

    # Carica l'architettura dal file JSON
    with open(architecture_filename, 'r') as f:
        loaded_architecture = json.load(f)

    return loaded_weights, loaded_architecture
