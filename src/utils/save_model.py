import json
from datetime import datetime
import numpy as np

def save_model(weights_list, layer_units, activations):
    """
    Salva i pesi e l'architettura di una rete neurale generica.

    weights_list: lista di matrici NumPy (ogni matrice è un set di pesi tra due layer)
    layer_units: lista di interi con il numero di unità per ogni layer
    activations: lista di stringhe con l'attivazione per ogni layer tranne l'input

    Esempio:
    layer_units = [3, 5, 4, 2]
    activations = ["relu", "relu", "linear"]
    weights_list = [W0, W1, W2]
    """

    # Creazione dizionario dei pesi
    weights = {}
    for i, w in enumerate(weights_list):
        # convertiamo ciascuna matrice NumPy in lista Python con tolist()
        weights[f"layer_{i}_weights"] = w.tolist()

    # Costruzione dell'architettura come lista di layer
    architecture = {"layers": []}

    # Layer di input
    architecture["layers"].append({
        "type": "input",
        "units": layer_units[0]
    })

    # Layer densi (hidden + output)
    for idx in range(1, len(layer_units)):
        architecture["layers"].append({
            "type": "dense",
            "units": layer_units[idx],
            "activation": activations[idx - 1]
        })

    # Salvataggio dei pesi su file di testo
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    weights_filename = f'weights_{current_time}.txt'
    with open(weights_filename, 'w') as f:
        for name, matrix in weights.items():
            f.write(f"{name}:\n")
            for row in matrix:
                f.write(','.join(map(str, row)) + "\n")
            f.write("\n")

    # Salvataggio dell'architettura in JSON
    architecture_filename = f'architecture_{current_time}.json'
    with open(architecture_filename, 'w') as f:
        json.dump(architecture, f, indent=2)

    return weights_filename, architecture_filename

