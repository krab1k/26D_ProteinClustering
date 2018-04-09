import numpy as np
from numba import jit


@jit(nopython=True)
def validate(labels, distances):
    # exclude diagonal and make symmetric
    d = distances - np.eye(len(distances))
    d = (d + d.T) / 2
    hits = 0
    n = len(d)
    indices = [(i, j) for i in range(n) for j in range(n) if d[i, j] > 0.5]
    for i, j in indices:
        if labels[i] == labels[j]:
            hits += 1

    return hits / len(indices)
