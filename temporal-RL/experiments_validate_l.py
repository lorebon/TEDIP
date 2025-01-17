# Lorenzo Bonasera 2024

import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from ColumnGeneration_tree import computeValidation, computeTest
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from metrics import euclidean
from preprocessing import preprocessTrain, preprocessTest
import multiprocessing as mp
import argparse
import os
from sktime.datasets import load_UCR_UEA_dataset


def computeResults(dataset, minshap, maxshap, n_shap, seed):
    np.random.seed(seed)

    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)
    X_train, y_train, y_set, y_emb, scaler = preprocessTrain(X_train, y_train)
    J = X_train.shape[1]
    K = len(y_set)
    if K > 8:
        return None

    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler)

    columns = ['Seed', 'Represented trees', 'Represented paths', 'Leaves', 'Accuracy', 'Full Model']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    #depth = max(int(np.ceil(np.log2(K))), 3)
    depth = 3
    if n_shap == 'def.':
        n_shap = int(np.sqrt((J*J)/2))
    else:
        n_shap = int(n_shap)

    # start = time.time()
    val_scores = []
    val_params = []

    # validation
    for train_index, val_index in skf.split(np.zeros(X_train.shape[0]), y_train):
        best_leaves, best_score = computeValidation(X_train[train_index], y_train[train_index],
                                                        X_train[val_index], y_train[val_index],
                                    depth, seed, minshap, maxshap, n_shap, Nmin=None, leaf_nodes=None)
        val_params.append(best_leaves)
        val_scores.append(best_score)

    # check infeasibility
    print(val_scores)
    best_val_leaves = val_params[np.argmax(val_scores)]
    if best_val_leaves == 0:
        return None

    reptrees, reprpaths, paths, labels, clf, acc, og_score = computeTest(X_train, y_train, X_test, y_test, depth, seed, minshap, maxshap, n_shap, lambd=1, leaf_nodes=best_val_leaves, Nmin=None)
    if reptrees is None:
        return None

    data = [seed, reptrees, reprpaths, len(paths), acc, og_score]
    df = pd.DataFrame([data], columns=columns)
    return df


if __name__ == '__main__':
    print("Command-line arguments:", sys.argv)
    # parse dataset name
    parser = argparse.ArgumentParser(description="Process dataset.")
    parser.add_argument("dataset", help="Path to the dataset file.", type=str)
    parser.add_argument("l", help="minshap.", type=float, default=0.0)
    parser.add_argument("u", help="maxshap.", type=float, default=1.0)
    parser.add_argument("r", help="n_shap.", type=str)
    args = parser.parse_args()
    dataset = args.dataset
    minshap = args.l
    maxshap = args.u
    n_shap = args.r

    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)
    X_train, y_train, y_set, y_emb, scaler = preprocessTrain(X_train, y_train)
    K = len(y_set)
    if K > 8:
        exit(2)

    # parallelize over different seeds
    seeds = list(range(10))
    pool = mp.Pool(len(seeds))
    list_of_df = pool.starmap(computeResults, [(dataset, minshap, maxshap, n_shap, seed) for seed in seeds])
    summary_df = pd.concat(list_of_df)
    summary_df = summary_df.round(3)

    if not os.path.exists('TEST'):
        os.makedirs('TEST')
    summary_df.to_csv('TEST/{}.csv'.format(dataset), sep=';', index=False)





