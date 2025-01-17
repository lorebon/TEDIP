import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from GenerateProblem import computeAll, computeDifferentLambda
from fidelityMeasure import regressionDisagreement, F1ComplCorr
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from dataset import load_openml
import pandas as pd
import multiprocessing as mp
import argparse
import os


def computeResults(dataset, seed):
    np.random.seed(seed)

    ### Load data
    X, y = load_openml(dataset)

    columns = ['Seed', 'Represented trees', 'Represented paths', 'Disagreement', 'F1', 'Leaves', 'MSE', 'Full Model']

    train_ratio = 0.75

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=seed)

    # std scaler
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))

    leaf_nodes = 15
    depth = 3

    reptrees, reprpaths, paths, labels, clf, acc, og_score = computeAll(X_train, y_train, X_test, y_test, depth, seed=seed, eps=None, leaf_nodes=leaf_nodes)
    disag = regressionDisagreement(X_test, y_test, clf, paths, labels)
    f1_score = F1ComplCorr(clf, paths)
    data = [seed, reptrees, reprpaths, disag, f1_score, len(paths), acc, og_score]
    df = pd.DataFrame([data], columns=columns)
    return df


if __name__ == '__main__':
    # parse dataset name
    parser = argparse.ArgumentParser(description="Process dataset.")
    parser.add_argument("dataset", help="Path to the dataset file.")
    args = parser.parse_args()
    dataset = args.dataset

    #computeResults(dataset, 1)

    # parallelize over different seeds
    seeds = list(range(30))
    pool = mp.Pool(10)
    list_of_df = pool.starmap(computeResults, [(dataset, seed) for seed in seeds])
    summary_df = pd.concat(list_of_df)
    summary_df = summary_df.round(3)
    summary_df.to_csv('test.csv'.format(dataset), sep=';', index=False)
