from scipy import stats
from mip import generateProblemSoft, generateCSP
import numpy as np
from warm_start import *
import gurobipy as gp
import matplotlib.pyplot as plt
from fidelityMeasure import regressionDisagreement
from sklearn.metrics import accuracy_score
from gurobipy import GRB, quicksum


def computeValidation(X_train, y_train, X_test, y_test, depth, seed, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = accuracy_score(y_pred, y_test)
    print("accuracy RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, Nmin)
    if loss is None:
        return 100

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    if leaf_nodes is not None:
        model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, lambd)
        #model.setObjective(quicksum((lambd)*freq[j] * z[j] for j in range(L)) - (1-lambd) * quicksum(loss[j] * z[j] for j in range(L)),
        #               sense=GRB.MAXIMIZE)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            return 100

        leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0.5]
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], [weights[i] for i in leaves], stats.mode(y_train)[0][0])
        return acc


def computeAll(X_train, y_train, X_test, y_test, depth, seed, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]
    #depth = np.ceil(np.log2(leaf_nodes)).astype(int)

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = accuracy_score(y_pred, y_test)
    print("Accuracy RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, Nmin)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # check upper bound on l
    if leaf_nodes is not None:
        model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, lambd)
        #model.setObjective(quicksum((lambd)*freq[j] * z[j] for j in range(L)) - (1-lambd) * quicksum(loss[j] * z[j] for j in range(L)),
        #               sense=GRB.MAXIMIZE)
        model.optimize()

        leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0.5]

        # print("leaves:", leaves)
        reprtrees = checkTrees([paths[i] for i in leaves], trees_noded)
        reprpaths = checkTreePaths([paths[i] for i in leaves], trees_pathed)
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], [weights[i] for i in leaves], stats.mode(y_train)[0][0])
        return reprtrees, reprpaths, [paths[i] for i in leaves], [labels[i] for i in leaves], clf, acc, og_score
