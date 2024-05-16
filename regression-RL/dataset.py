import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from scipy.io import arff
from ucimlrepo import fetch_ucirepo
    

def load_openml(dataset):
    X, y = fetch_openml(dataset, return_X_y=True)
    y = np.asarray(y)
    X = X.dropna()
    y = y[list(X.index.values)]
    encoder = OneHotEncoder(sparse_output=False)
    X_categ = X.select_dtypes(include=[object, 'string', 'category'])
    XX = encoder.fit_transform(X_categ)
    X = X.drop(columns=X_categ.columns).to_numpy()
    X = np.concatenate([X, XX], axis=1)

    return X, y
