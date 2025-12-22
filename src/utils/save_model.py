import json
from datetime import datetime

def save_model(output_HL2, HL2_HL1, HL1_input ):
    weights = {
        'input_to_hidden1': HL1_input,
        'hidden1_to_hidden2': HL2_HL1,
        'hidden2_to_output': output_HL2
    }

    architecture = {
        'layers': [
            {'type': 'input', 'units': 3},
            {'type': 'dense', 'units': 4, 'activation': 'relu'},
            {'type': 'dense', 'units': 2, 'activation': 'relu'},
            {'type': 'dense', 'units': 2, 'activation': 'linear'}
        ]
    }

    # Get current date and time
    current_time = datetime.now()
    # Format the date and time for the filename (e.g., 2025-10-10_14-30)
    time_str = current_time.strftime("%Y-%m-%d_%H-%M")

    # Save the weights with the current date and hour in the filename
    weights_filename = f'weights_{time_str}.txt'
    with open(weights_filename, 'w') as f:
        for layer, weight_matrix in weights.items():
            f.write(f"{layer}:\n")
            for row in weight_matrix:
                f.write(','.join(map(str, row)) + '\n')
            f.write('\n')

    with open('architecture.json', 'w') as f:
        json.dump(architecture, f)


