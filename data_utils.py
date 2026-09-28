"""Dataset loading and categorical preprocessing for standalone experiments."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_diabetes, load_iris
from sklearn.preprocessing import OneHotEncoder


def clean_tabular(X, y, task):
    """Drop incomplete rows by position, keeping features and targets aligned."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(np.asarray(y).reshape(-1))
    if len(X) != len(y):
        raise ValueError("Features and targets must have the same row count.")
    if task == "regression":
        y = pd.to_numeric(y, errors="raise")
    valid = X.notna().all(axis=1) & y.notna()
    dropped = int((~valid).sum())
    if dropped:
        warnings.warn(
            f"Dropped {dropped} rows with missing features or targets.", UserWarning, stacklevel=2
        )
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].to_numpy()
    if len(y) < 2 or X.shape[1] == 0:
        raise ValueError("At least two complete rows and one feature are required.")
    numeric = X.select_dtypes(include="number").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (task == "regression" and not np.isfinite(y).all()):
        raise ValueError("Input data contain infinite values.")
    return X, y


def load_tabular(dataset, task, target=None):
    """Load builtin:iris/diabetes, CSV, uci:NAME, openml:ID, or an OpenML name.

    OpenML data IDs are preferable to names for reproducible experiments.
    Sources are explicit: failed OpenML requests do not switch to UCI.
    """
    if dataset.startswith("builtin:"):
        name = dataset.split(":", 1)[1]
        expected = "iris" if task == "classification" else "diabetes"
        if name != expected:
            raise ValueError(f"Use builtin:{expected} for {task}.")
        data = (load_iris if task == "classification" else load_diabetes)(as_frame=True)
        X, y = data.data, data.target
    elif Path(dataset).is_file():
        if target is None:
            raise ValueError("A local CSV requires --target COLUMN.")
        frame = pd.read_csv(dataset)
        if target not in frame:
            raise ValueError(f"Target column {target!r} is absent from the CSV.")
        X, y = frame.drop(columns=target), frame[target]
    elif dataset.startswith("uci:"):
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(name=dataset.split(":", 1)[1])
        X, y = data.data.features, data.data.targets
        if y is None or y.shape[1] != 1:
            raise ValueError("Only a single target column is supported.")
    else:
        if dataset.lower().endswith(".csv"):
            raise FileNotFoundError(f"CSV file does not exist: {dataset}")
        options = {"as_frame": True, "return_X_y": True}
        if dataset.startswith("openml:"):
            options["data_id"] = int(dataset.split(":", 1)[1])
        else:
            options["name"] = dataset
        X, y = fetch_openml(**options)
        if np.asarray(y).ndim == 2 and np.asarray(y).shape[1] != 1:
            raise ValueError("Only a single target column is supported.")
    return clean_tabular(X, y, task)


def encode_features(X_train, X_test):
    """Fit one-hot categories on training rows; ignore unseen test categories."""
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]
    encoder = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    train = encoder.fit_transform(X_train).astype(float)
    test = encoder.transform(X_test).astype(float)
    return train, test, encoder.get_feature_names_out().tolist(), encoder
