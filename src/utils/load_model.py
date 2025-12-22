import json

def load_model(weights_filename, architecture_filename):
    loaded_weights = {}
    with open(weights_filename, 'r') as f:
        lines = f.readlines()
        current_layer = None
        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                current_layer = line[:-1]
                loaded_weights[current_layer] = []
            elif line:
                loaded_weights[current_layer].append(list(map(float, line.split(','))))

    # Load the architecture
    with open(architecture_filename, 'r') as f:
        loaded_architecture = json.load(f)
    HL1_input = loaded_weights['input_to_hidden1']
    HL2_HL1 = loaded_weights['hidden1_to_hidden2']
    output_HL2 = loaded_weights['hidden2_to_output']
    return HL1_input,HL2_HL1,output_HL2, loaded_architecture