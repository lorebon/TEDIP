import numpy as np
import sktime.datatypes._panel._convert as conv
from sklearn.model_selection import train_test_split


def preprocessTrain(X, y):
    y = np.asarray(list(map(int, y)))
    y_set = set(y)

    embedding = np.arange(0, len(y_set))
    for idx, label in enumerate(y_set):
        for i, data in enumerate(y):
            if data == label:
                y[i] = embedding[idx]

    X = conv.from_nested_to_2d_array(X)
    X = X.to_numpy()
    y_emb = set(y)

    return X, y, y_set, y_emb, None


def preprocessTest(X, y, y_set, scaler):
    y = np.asarray(list(map(int, y)))

    embedding = np.arange(0, len(y_set))
    for idx, label in enumerate(y_set):
        for i, data in enumerate(y):
            if data == label:
                y[i] = embedding[idx]

    X = conv.from_nested_to_2d_array(X)
    X = X.to_numpy()

    return X, y








