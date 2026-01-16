import json
import os
from datetime import datetime
import numpy as np

def make_json_serializable(obj):
    """
    Converte ricorsivamente un oggetto in una versione serializzabile da JSON.

    - ndarray -> list
    - numpy scalar -> Python scalar
    - list -> ricorsione
    - dict -> ricorsione
    - callable -> nome della funzione (stringa)
    - altri oggetti -> stringa descrittiva
    """
    # NumPy array -> lista Python
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # NumPy scalari -> Python scalari
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    # Dict -> ricorsione chiave -> valore
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    # Lista -> ricorsione su ogni elemento
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    # Funzioni -> salva il nome
    if callable(obj):
        return obj.__name__
    # Tipi base Python sono già OK
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # Oggetti sconosciuti -> rappresentazione stringa
    return str(obj)


def save_model(trainer, weights_list, layer_units, activations, numero=1):
    """
    Salva:
     • i pesi (weights_list)
     • l'architettura della rete (layer_units + activations)
     • tutti i parametri dell'oggetto trainer in forma JSON-serializzabile

    trainer: istanza del tuo trainer
    weights_list: lista di matrici NumPy (pesi)
    layer_units: lista di interi (numero di unità per layer)
    activations: lista di funzioni di attivazione, una per layer (tranne input)
    numero: indice o ID della fold (o qualsiasi altro identificatore)
    """

    weights = {}
    for i, w in enumerate(weights_list):
        weights[f"layer_{i}_weights"] = w.tolist()

    architecture = {"layers": []}


    architecture["layers"].append({
        "type": "input",
        "units": layer_units[0]
    })

    for idx in range(1, len(layer_units)):
        architecture["layers"].append({
            "type": "dense",
            "units": layer_units[idx],
            "activation": activations[idx - 1].__name__
        })

    current_time = datetime.now().strftime("%Y-%m-%d_%H")
    folder_path = os.path.join("../results/models", current_time)
    os.makedirs(folder_path, exist_ok=True)

    weights_filename = f"weights_{numero}.txt"
    weights_path = os.path.join(folder_path, weights_filename)
    with open(weights_path, "w") as f:
        for name, matrix in weights.items():
            f.write(f"{name}:\n")
            for row in matrix:
                f.write(",".join(map(str, row)) + "\n")
            f.write("\n")

    trainer_params = {}
    for key, val in vars(trainer).items():
        if key == "neuraln":
            # non serializziamo l’oggetto rete
            continue
        trainer_params[key] = make_json_serializable(val)

    model_data = {
        "architecture": architecture,
        "trainer_params": trainer_params
    }

    architecture_filename = f"architecture_{numero}.json"
    arch_path = os.path.join(folder_path, architecture_filename)
    with open(arch_path, "w") as f:
        json.dump(model_data, f, indent=2)

    return weights_filename, architecture_filename
