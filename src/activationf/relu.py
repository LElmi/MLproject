import numpy as np

def relu(z, derivata: bool = False):

    if derivata:    
        return (z > 0).astype(float)

    else:
        return np.maximum(0, z)