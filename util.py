import numpy as np

def is_binary(x):
    x = np.array(x)
    return ((x == 0) | (x == 1)).all()
