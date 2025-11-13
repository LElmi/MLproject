import numpy as np
from models.relu import relu
from models.dropout import apply_dropout

def forward(X, W1, b1, W2, b2, W3, b3, W4, b4, dropout_rate, training=True):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    a1, mask1 = apply_dropout(a1, dropout_rate, training)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    a2, mask2 = apply_dropout(a2, dropout_rate, training)

    z3 = a2 @ W3 + b3
    a3 = relu(z3)
    a3, mask3 = apply_dropout(a3, dropout_rate, training)

    z4 = a3 @ W4 + b4
    cache = (X, z1, a1, mask1, z2, a2, mask2, z3, a3, mask3)
    return z4, cache
