import copy
import time
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statistics import fmean
from sklearn.model_selection import StratifiedKFold
from sklearn_extra.cluster import KMedoids as KMED
from metrics import euclidean, scaled_euclidean, DTW, scaled_DTW, minkowski, scaled_minkowski
import numpy as np
from joblib import Parallel, delayed
from GenerateProblem import computeAll, parallelized_distance_calculation
from sktime.datasets import load_UCR_UEA_dataset
from preprocessing import preprocessTrain, preprocessTest
from tslearn.barycenters import euclidean_barycenter
from warm_start import computePaths, computeLoss, computeScore
from mip import generateProblem
from wildboar.ensemble import ShapeletForestClassifier


def precompute(X_train, y_train, exemplars, H):
    K = len(set(y_train))
    J = X_train.shape[1]
    num_shapelet = J - H + 1

    ### Compute shapelets distances
    shapelets = {}
    for shap in range(num_shapelet):
        for k in range(K):
            shapelets[shap, k] = exemplars[k, shap:shap+H]

    distances_train = parallelized_distance_calculation(num_shapelet, K, shapelets, X_train, metric, -1)
    return distances_train, shapelets


def cross_validate(X_train_cv, y_train_cv, X_test_cv, y_test_cv, metric, leaf_nodes, seed):
    acc, _ = computeAll(X_train_cv, y_train_cv, X_test_cv, y_test_cv, metric, leaf_nodes, seed)
    return acc


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)

    ### Load data
    dataset = 'Lightning2'
    metric = euclidean

    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)
    X_train, y_train, y_set, y_emb, scaler = preprocessTrain(X_train, y_train)
    J = X_train.shape[1]
    n = X_train.shape[0]
    K = len(y_set)

    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler)

    leaf_nodes = K

    # compute upper bound on l
    max_leaf_nodes = 10
    _, ub = computeAll(X_train, y_train, X_test, y_test, metric, max_leaf_nodes, seed=seed, alpha=100000)
    print("ub for l:", ub)

    # lower bound coincides with the number of classes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    leaf_values = np.arange(K, ub+1)
    leaf_values = np.arange(ub, ub+1)
    scores = np.zeros(len(leaf_values))

    start = time.time()
    for i, leaf_nodes in enumerate(leaf_values):

        scores_cv = Parallel(n_jobs=skf.n_splits, backend="multiprocessing")(
            delayed(cross_validate)(X_train[train_index], y_train[train_index], X_train[test_index],
                                    y_train[test_index], metric, leaf_nodes.item(), seed)
            for train_index, test_index in skf.split(np.zeros(X_train.shape[0]), y_train))
        scores[i] = fmean(scores_cv)

    print(scores)
    best_l = leaf_values[np.argmax(scores)]

    # test accuracy
    test_acc, _ = computeAll(X_train, y_train, X_test, y_test, metric, best_l.item(), seed)
    end = time.time()
    print("test accuracy {}%".format(round(test_acc, 2)))
    print("total elapsed time:", round(end-start, 2))


