# Lorenzo Bonasera 2024

import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from ColumnGeneration_tree import *
from fidelityMeasure import regressionDisagreement, F1ComplCorr
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from dataset import load_openml
import pandas as pd
import multiprocessing as mp
import argparse
import os
from sktime.datasets import load_UCR_UEA_dataset
from preprocessing import preprocessTrain, preprocessTest


def computeResults(dataset, minshap, maxshap, n_shap, seed):
    np.random.seed(seed)

    columns = ['Seed', 'Represented trees', 'Represented paths', 'Disagreement', 'Leaves', 'Accuracy', 'Full Model', 'Best tree disag.', 'Best tree Accuracy']

    ### Load data
    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)
    X_train, y_train, y_set, y_emb, scaler = preprocessTrain(X_train, y_train)
    J = X_train.shape[1]
    K = len(y_set)
    if K > 8:
        return None

    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler)

    depth = 3
    if n_shap == 'def.':
        n_shap = int(np.sqrt((J * J) / 2))
    else:
        n_shap = int(n_shap)

    if n_shap < 100:
        n_shap = 100

    leaf_nodes = 6

    reptrees, reprpaths, paths, labels, clf, acc, og_score, best_tree_disag, best_tree_acc = computeAll(X_train, y_train, X_test, y_test, depth, seed, minshap, maxshap, n_shap, Nmin=None, leaf_nodes=leaf_nodes)
    disag = regressionDisagreement(X_test, y_test, clf, paths, labels)
    #f1_score = F1ComplCorr(clf, paths)
    print("accuracy:", acc)
    data = [seed, reptrees, reprpaths, disag, len(paths), acc, og_score, best_tree_disag, best_tree_acc]
    df = pd.DataFrame([data], columns=columns)

    return df

    timeseries = X_train[3, :]
    shapelet = paths[0][0][0]
    shapelet_len = len(shapelet)
    min_distance = 100


    for offset in range(len(timeseries) - shapelet_len + 1):
        subsequence = timeseries[offset:offset + shapelet_len]
        # Compute Euclidean distance between subsequence and shapelet
        distance = np.sqrt(np.sum((subsequence - shapelet) ** 2))

        if distance < min_distance:
            min_distance = distance
            best_offset = offset

    # Output the result
    print(f"Minimum Euclidean distance: {min_distance}")
    print(f"Best offset (index of best match): {best_offset}")

    exit(2)

    # Save timeseries data to 'timeseries.txt'
    with open('timeseries_2.txt', 'w') as f:
        for idx, value in enumerate(timeseries):
            f.write(f"{idx} {value}\n")

    # Save shapelet data to 'shapelet.txt'
    with open('shapelet_2.txt', 'w') as f:
        for idx, value in enumerate(shapelet):
            f.write(f"{idx+best_offset} {value}\n")


if __name__ == '__main__':
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

    test = computeResults(dataset, minshap, maxshap, n_shap, 6)
    #print(test)
    exit(2)

    # parallelize over different seeds
    seeds = list(range(10))
    pool = mp.Pool(10)
    list_of_df = pool.starmap(computeResults, [(dataset, minshap, maxshap, n_shap, seed) for seed in seeds])
    summary_df = pd.concat(list_of_df)
    summary_df = summary_df.round(3)

    if not os.path.exists('test_noval'):
        os.makedirs('test_noval')
    summary_df.to_csv('test_noval/{}.csv'.format(dataset), sep=';', index=False)





