import numpy as np
from tslearn.metrics import dtw


def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))


def scaled_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2)/len(x))


def minkowski(x, y):
    return np.linalg.norm(x - y, ord=1)


def scaled_minkowski(x, y):
    return np.linalg.norm(x - y, ord=1)/len(x)


def DTW(x, y):
    return dtw(x, y)


def scaled_DTW(x, y):
    return dtw(x, y)/len(x)