import math

### ACTIVATION FUNCTION SIGMA
#.  arg -> alpha is for the slop

def sigmaf(x, alpha=1):
    return 1 / (1 + math.e ** (-alpha * x))

def dsigmaf(x, alpha=1):
    s = sigmaf(x, alpha)
    return s * (1 - s)