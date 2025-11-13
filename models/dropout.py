import numpy as np

def apply_dropout(a, dropout_rate, training):
    if not training or dropout_rate == 0:
        return a, 1
    mask = (np.random.rand(*a.shape) > dropout_rate).astype(float)
    a *= mask / (1 - dropout_rate)
    return a, mask
