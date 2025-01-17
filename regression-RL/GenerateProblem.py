from mip import generateProblem, generateProblemSoft, generateCSP
import numpy as np
from warm_start import *
import gurobipy as gp
import matplotlib.pyplot as plt
from fidelityMeasure import regressionDisagreement
from sklearn.metrics import mean_squared_error
from gurobipy import GRB, quicksum


def computeValidation(X_train, y_train, X_test, y_test, depth, seed, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = mean_squared_error(y_pred, y_test)
    print("mean squared error RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, Nmin)
    if loss is None:
        return 100

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
    print("lower bound for leaves:", lower_l)

    model.addConstr(z.sum() <= lower_l, name='card')
    for l_test in np.arange(lower_l, upper_l+1):
        if l_test > lower_l:
            model.getConstrByName("card").setAttr('rhs', l_test)
        model.setObjective(gp.quicksum((lambd) * freq[j] * z[j] for j in range(L)) - (1-lambd) * gp.quicksum(loss[j] * z[j] for j in range(L)),
                           sense=gp.GRB.MAXIMIZE)
        model.update()
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            return 100

        if model.Status == 2:
            leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0.5]
            acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], [weights[i] for i in leaves])
            accs.append(acc)
            leaves_val.append(leaves)

    best_score = np.min(accs)
    best_leaves = leaves_val[np.argmin(accs)]
    return len(best_leaves), best_score



def computeTest(X_train, y_train, X_test, y_test, depth, leaf_nodes, seed, lambd=1, Nmin=None):
    n = X_train.shape[0]

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = mean_squared_error(y_pred, y_test)
    print("mean squared error RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, Nmin)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # generate problem
    model, z = generateCSP(L, n, A)
    model.setParam('OutputFlag', 0)
    model.addConstr(z.sum() <= leaf_nodes, name='card')
    model.setObjective(gp.quicksum((lambd) * freq[j] * z[j] for j in range(L)) - (1 - lambd) * gp.quicksum(loss[j] * z[j] for j in range(L)),
                       sense=gp.GRB.MAXIMIZE)
    model.update()
    model.optimize()

    if model.Status == 2:
        # retrieve solution
        leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0.5]
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], [weights[i] for i in leaves])
    else:
        acc = 0
    return acc, og_score, model.Runtime



def computeAll(X_train, y_train, X_test, y_test, depth, seed, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]
    #depth = np.ceil(np.log2(leaf_nodes)).astype(int)

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = mean_squared_error(y_pred, y_test)
    print("mean squared error RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, Nmin)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # retrieve solutions
    accs = []
    leaves_val = []

    model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, lambd)
    model.optimize()

    leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0.5]

    # print("leaves:", leaves)
    reprtrees = checkTrees([paths[i] for i in leaves], trees_noded)
    reprpaths = checkTreePaths([paths[i] for i in leaves], trees_pathed)
    # plotPaths(loss, freq, labels, leaves)
    acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], [weights[i] for i in leaves])
    return reprtrees, reprpaths, [paths[i] for i in leaves], [labels[i] for i in leaves], clf, acc, og_score
        



