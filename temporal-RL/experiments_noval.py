# Lorenzo Bonasera 2024

import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from GenerateProblem import *
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

    columns = ['Seed', 'Represented trees', 'Represented paths', 'Leaves', 'Accuracy', 'Full Model']

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

    leaf_nodes = 6

    reptrees, reprpaths, paths, labels, clf, acc, og_score = computeAll(X_train, y_train, X_test, y_test, depth, seed, minshap, maxshap, n_shap, Nmin=None, leaf_nodes=leaf_nodes)
    print("accuracy:", acc)
    data = [seed, reptrees, reprpaths, disag, len(paths), acc, og_score]
    df = pd.DataFrame([data], columns=columns)

    return df

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
    
    # parallelize over different seeds
    seeds = list(range(10))
    pool = mp.Pool(10)
    list_of_df = pool.starmap(computeResults, [(dataset, minshap, maxshap, n_shap, seed) for seed in seeds])
    summary_df = pd.concat(list_of_df)
    summary_df = summary_df.round(3)

    if not os.path.exists('test_noval'):
        os.makedirs('test_noval')
    summary_df.to_csv('test_noval/{}.csv'.format(dataset), sep=';', index=False)





