import copy
import time
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statistics import fmean
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import Parallel, delayed
from GenerateProblem import computeAll
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
from dataset import load_openml
from sklearn.datasets import fetch_openml


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)

    ### Load data
    #dataset = fetch_ucirepo(id=9)
    X, y = load_openml("us_crime")

    train_ratio = 0.75
    test_ratio = 0.25

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=seed)

    # std scaler
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))
    p = X_train.shape[1]
    n = X_train.shape[0]

    depth = 3
    leaf_nodes = 15
    eps = 1

    start = time.time()
    acc, leaves, og_score, runtime = computeAll(X_train, y_train, X_test, y_test, depth, seed=seed, eps=eps, leaf_nodes=leaf_nodes)
    print("test MSE with {} leaves:".format(len(leaves)), round(acc, 2))
    #print("trust: {}%".format(round(acc*100/og_score, 2)))
    print("total elapsed time:", round(time.time() - start, 2))
    print("gurobi runtime:", round(runtime, 2))

    #print("extracted rules:")
    #for l in leaves:
    #    print(l)


