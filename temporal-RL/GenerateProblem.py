import copy
import time
from metrics import euclidean, scaled_euclidean, DTW, scaled_DTW, minkowski, scaled_minkowski
from mip import generateCSP, generateProblemSoft
import numpy as np
from warm_start import computePaths, computeLoss, computeScore
from joblib import Parallel, delayed
import gurobipy as gp


def minDist(shapelet, data, metric):
    min_distance = float('inf')

    for i in range(len(data) - len(shapelet) + 1):
        distance = metric(shapelet, data[i:i + len(shapelet)])
        min_distance = min(min_distance, distance)

    return min_distance


def calculate_min_distance(args):
    i, num_shapelets, shapelets, X_train, metric, K = args
    return [np.argmin([minDist(shapelets[shap, k], X_train[i, :], metric) for k in range(K)]) for shap
            in range(num_shapelets)]


def parallelized_distance_calculation(num_shapelets, K, shapelets, X_train, metric, num_processes=None):
    n = X_train.shape[0]

    distances_train = np.array(
        Parallel(n_jobs=num_processes, backend="multiprocessing")(
            delayed(calculate_min_distance)((i, num_shapelets, shapelets, X_train, metric, K)) for i in range(n))
    )

    return distances_train



def computeValidation(X_train, y_train, X_test, y_test, metric, depth, seed, minshap, maxshap, n_shap):
    n = X_train.shape[0]

    # start random forest
    paths, clf = computePaths(X_train, y_train, 500, minshap, maxshap,  n_shap, depth, seed)
    og_score = clf.score(X_test, y_test)*100
    print("test score RF:", round(og_score, 2))
    loss, samples, labels, paths, freq = computeLoss(X_train, y_train, paths, metric, None)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # retrieve solutions
    accs = []
    leaves_val = []

    # check upper bound on l
    model, z = generateCSP(L, n, A)
    model.setObjective(z.sum(), sense=gp.GRB.MAXIMIZE)
    model.setParam('OutputFlag', 0)
    model.update()
    model.optimize()
    upper_l = int(model.getObjective().getValue())
    print("upper bound for leaves:", upper_l)

    # check lower bound on l
    model.setObjective(z.sum(), sense=gp.GRB.MINIMIZE)
    model.update()
    model.optimize()
    lower_l = int(model.getObjective().getValue())
    if lower_l < 2:
        lower_l = 2
    print("lower bound for leaves:", lower_l)

    model.addConstr(sum(z.select('*')) <= lower_l, name='card')
    for l_test in np.arange(lower_l, upper_l+1):
        if l_test > lower_l:
            model.getConstrByName("card").setAttr('rhs', l_test)
        model.setObjective(gp.quicksum(freq[j] * z[j] for j in range(L)) - gp.quicksum(loss[j] * z[j] for j in range(L)),
                               sense=gp.GRB.MAXIMIZE)
        model.update()
        model.optimize()

        if model.Status == 2:
            # retrieve solution
            leaves = []
            for i in range(L):
                var = model.getVarByName("z[{}]".format(i))
                if int(var.x) == 1:
                    leaves.append(i)

            print("leaves:", leaves)
            acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric)
            accs.append(acc)
            leaves_val.append(leaves)
    print(accs)
    accs = np.asarray(accs)
    best_score = np.max(accs)
    best_leaves = leaves_val[np.argmax(accs)]
    return len(best_leaves), best_score



def computeTest(X_train, y_train, X_test, y_test, metric, depth, seed, minshap, maxshap, n_shap, leaf_nodes):
    n = X_train.shape[0]

    # start random forest
    paths, clf = computePaths(X_train, y_train, 500, minshap, maxshap,  n_shap, depth, seed)
    og_score = clf.score(X_test, y_test)*100
    print("test score RF:", round(og_score, 2))
    loss, samples, labels, paths, freq = computeLoss(X_train, y_train, paths, metric, None)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # check upper bound on l
    model, z = generateCSP(L, n, A)
    model.setParam('OutputFlag', 0)
    model.addConstr(z.sum() <= leaf_nodes, name='card')
    model.setObjective(gp.quicksum(freq[j] * z[j] for j in range(L)) - gp.quicksum(loss[j] * z[j] for j in range(L)),
                       sense=gp.GRB.MAXIMIZE)

    model.update()
    model.optimize()
    if model.Status == 2:
        # retrieve solution
        leaves = []
        for i in range(L):
            var = model.getVarByName("z[{}]".format(i))
            if int(var.x) == 1:
                leaves.append(i)

        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric)
    else:
        acc = 0
    return acc, og_score, model.Runtime


def computeAll(X_train, y_train, X_test, y_test, metric, depth, seed, minshap, maxshap, n_shap, leaf_nodes=None):
    n = X_train.shape[0]

    # start random forest
    paths, clf = computePaths(X_train, y_train, 500, minshap, maxshap, n_shap, depth, seed)
    og_score = clf.score(X_test, y_test)*100
    print("test score RF:", round(og_score, 2))
    loss, samples, labels, paths, freq = computeLoss(X_train, y_train, paths, metric, 0.95)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # check upper bound on l
    if leaf_nodes is not None:
        model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, 1)
        model.optimize()
        leaves = []
        for i in range(L):
            var = model.getVarByName("z[{}]".format(i))
            if int(var.x) == 1:
                leaves.append(i)

        print("leaves:", leaves)
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric)
        return acc, [paths[i] for i in leaves], og_score, model.Runtime

    # retrieve solutions
    accs = []
    leaves_val = []

    # check upper bound on l
    model, z = generateCSP(L, n, A)
    model.setObjective(z.sum(), sense=gp.GRB.MAXIMIZE)
    model.update()
    model.optimize()
    upper_l = int(model.getObjective().getValue())
    print("upper bound for leaves:", upper_l)

    # check lower bound on l
    model.setObjective(z.sum(), sense=gp.GRB.MINIMIZE)
    model.update()
    model.optimize()
    lower_l = int(model.getObjective().getValue())
    print("lower bound for leaves:", lower_l)

    model.addConstr(sum(z.select('*')) <= lower_l, name='card')
    for l_test in np.arange(lower_l, upper_l+1):
        if l_test > lower_l:
            model.getConstrByName("card").setAttr('rhs', l_test)
        model.setObjective(gp.quicksum(freq[j] * z[j] for j in range(L)) - gp.quicksum(loss[j] * z[j] for j in range(L)),
                               sense=gp.GRB.MAXIMIZE)
        model.update()
        model.optimize()

        if model.Status == 2:
            # retrieve solution
            leaves = []
            for i in range(L):
                var = model.getVarByName("z[{}]".format(i))
                if int(var.x) == 1:
                    leaves.append(i)

            print("leaves:", leaves)
            acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric)
            accs.append(acc)
            leaves_val.append(leaves)
    print(accs)
    accs = np.asarray(accs)
    best_score = np.max(accs)
    best_leaves = leaves_val[np.argmax(accs)]
    return best_score, [paths[i] for i in best_leaves], og_score, model.Runtime






