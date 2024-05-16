import copy
import time
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statistics import fmean
import numpy as np
from joblib import Parallel, delayed
from GenerateProblem import computeAll
from sklearn.model_selection import train_test_split
import py_uci
from ucimlrepo import fetch_ucirepo
from dataset import load_openml


if __name__ == '__main__':
    seed = 2
    np.random.seed(seed)

    ### Load data
    dataset = "wdbc"
    X, y = load_openml(dataset)

    train_ratio = 0.75
    test_ratio = 0.25

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=seed)

    p = X_train.shape[1]
    n = X_train.shape[0]
    K = len(set(y_train))

    depth = 2
    leaf_nodes = 4
    eps = 1

    start = time.time()
    acc, leaves, og_score, runtime = computeAll(X_train, y_train, X_test, y_test, depth, seed=seed, eps=eps, leaf_nodes=leaf_nodes)
    print("test acc with {} leaves:".format(len(leaves)), round(acc, 2))
    print("trust: {}%".format(round(acc*100/og_score, 2)))
    print("total elapsed time:", round(time.time() - start, 2))
    print("gurobi runtime:", round(runtime, 2))

    print("extracted rules:")
    for l in leaves:
        print(l)


