import copy
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import StratifiedKFold
from metrics import euclidean, scaled_euclidean, DTW, scaled_DTW, minkowski, scaled_minkowski
import numpy as np
from joblib import Parallel, delayed
from GenerateProblem import computeAll, parallelized_distance_calculation
from sktime.datasets import load_UCR_UEA_dataset
from preprocessing import preprocessTrain, preprocessTest
from warm_start import computePaths, computeLoss, computeScore
from mip import generateProblem
from wildboar.ensemble import ShapeletForestClassifier


if __name__ == '__main__':
    seed = 5
    np.random.seed(seed)

    ### Load data
    dataset = 'FaceAll'
    metric = euclidean

    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)
    X_train, y_train, y_set, y_emb, scaler = preprocessTrain(X_train, y_train)
    J = X_train.shape[1]
    n = X_train.shape[0]
    K = len(y_set)

    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler)

    depth = int(np.ceil(np.log2(K)))
    print("depth:", depth)
    minshap = 0.2
    maxshap = 0.5
    n_shap = 100

    start = time.time()
    acc, leaves, og_score, runtime = computeAll(X_train, y_train, X_test, y_test, metric, depth, seed,
                                                minshap, maxshap, n_shap)
    print("test acc with {} leaves:".format(len(leaves)), round(acc, 2))
    print("trust: {}%".format(round(acc*100/og_score, 2)))
    print("total elapsed time:", round(time.time() - start, 2))
    print("gurobi time:", round(runtime, 2))
    for l in leaves:
        print(l)



