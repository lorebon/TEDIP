"""Distance functions for equal-length subsequences; Euclidean is the default."""

import numpy as np


def euclidean(x, y):
    return float(np.linalg.norm(np.asarray(x) - np.asarray(y)))


def scaled_euclidean(x, y):
    """Root mean squared distance (not z-normalized Euclidean distance)."""
    return euclidean(x, y) / np.sqrt(len(x))


def minkowski(x, y):
    """Manhattan distance (Minkowski p=1)."""
    return float(np.linalg.norm(np.asarray(x) - np.asarray(y), ord=1))


def scaled_minkowski(x, y):
    return minkowski(x, y) / len(x)


def DTW(x, y):
    """Optional helper; requires tslearn, which the default experiments do not use."""
    from tslearn.metrics import dtw

    return dtw(x, y)


def scaled_DTW(x, y):
    return DTW(x, y) / len(x)
