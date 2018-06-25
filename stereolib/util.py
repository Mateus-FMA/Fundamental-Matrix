import numpy as np

def get_cross_matrix(v):
    return np.matrix([[0, -v[2,0], v[1,0]], \
                      [v[2,0], 0, -v[0,0]], \
                      [-v[1,0], v[0,0], 0]])