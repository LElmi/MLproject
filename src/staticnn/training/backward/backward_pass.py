import numpy as np
from src.staticnn.activationf.sigm import sigmaf
#from src.staticnn.training.forward.forward_pass import forward_hidden, forward_output
def backprop_output(O_u, delta_t):
    DelW=O_u * delta_t

    return DelW