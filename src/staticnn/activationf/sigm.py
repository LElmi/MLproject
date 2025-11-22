import math

### ACTIVATION FUNCTION SIGMA
#.  arg -> alpha is for the slop

def sigmaf(x, alpha = 1): 

    return 1 / (math.e ** (alpha * x))