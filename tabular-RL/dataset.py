import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler, LabelEncoder
from scipy.io import arff
from ucimlrepo import fetch_ucirepo


def load_openml(dataset):
    try:
        X, y = fetch_openml(dataset, return_X_y=True, cache=False)
    except Exception:
        df = fetch_ucirepo(name=dataset)
        X = df.data.features
        y = df.data.targets.to_numpy()[:, 0]
    try:
        if dataset == 'MAGIC-Gamma-Telescope-Dataset':
            y = X["class"]
            X = X.drop(columns=["class", "Unnamed:_0"])
        y = np.asarray(y).astype(int)
    except ValueError:
        labelenc = LabelEncoder()
        labelenc.fit(y)
        y = labelenc.transform(y)
        y = np.asarray(y).astype(int)
    X = X.dropna()
    y = y[list(X.index.values)]
    encoder = OneHotEncoder(sparse_output=False)
    X_categ = X.select_dtypes(include=[object, 'string', 'category'])
    XX = encoder.fit_transform(X_categ)
    X = X.drop(columns=X_categ.columns).to_numpy()
    X = np.concatenate([X, XX], axis=1)

    return X, y
