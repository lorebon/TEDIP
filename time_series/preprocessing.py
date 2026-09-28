"""Convert univariate equal-length series and encode labels without mutation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def series_array(X):
    """Accept nested sktime frames, (n, time), or (n, 1, time) arrays."""
    if isinstance(X, pd.DataFrame) and X.shape[1] == 1 and len(X) and not np.isscalar(X.iloc[0, 0]):
        rows = [np.asarray(value, dtype=float) for value in X.iloc[:, 0]]
        if len({len(row) for row in rows}) != 1:
            raise ValueError("Variable-length series are not supported.")
        X = np.stack(rows)
    elif (
        isinstance(X, pd.DataFrame)
        and len(X)
        and any(not np.isscalar(value) for value in X.iloc[0])
    ):
        raise ValueError("Only univariate time series are supported.")
    X = np.asarray(X, dtype=float)
    if X.ndim == 3 and X.shape[1] == 1:
        X = X[:, 0, :]
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("Expected finite, nonempty, equal-length univariate series.")
    return X


def preprocessTrain(X, y):
    encoder = LabelEncoder().fit(np.asarray(y).reshape(-1))
    encoded = encoder.transform(np.asarray(y).reshape(-1))
    classes = tuple(encoder.classes_.tolist())
    X = series_array(X)
    if len(X) != len(encoded):
        raise ValueError("Series and labels have different row counts.")
    return X, encoded, classes, set(encoded.tolist()), None


def preprocessTest(X, y, y_set, scaler=None):
    """Reuse training class order; fail clearly on unseen labels."""
    classes = sorted(y_set) if isinstance(y_set, set) else list(y_set)
    original = np.asarray(y).reshape(-1)
    mapping = {label: index for index, label in enumerate(classes)}
    try:
        encoded = np.array([mapping[label] for label in original])
    except KeyError as error:
        raise ValueError(f"Test class {error.args[0]!r} did not occur in training.") from error
    X = series_array(X)
    if len(X) != len(encoded):
        raise ValueError("Series and labels have different row counts.")
    return X, encoded
