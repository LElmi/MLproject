import math

### ACTIVATION FUNCTION SIGMA
#.  arg -> alpha is for the slop

def sigmaf(x, alpha=1, derivata: bool = False):

    if derivata: 
        s = sigmaf(x, alpha)
        return s * (1 - s)
    else:
        return 1 / (1 + math.e ** (-alpha * x))