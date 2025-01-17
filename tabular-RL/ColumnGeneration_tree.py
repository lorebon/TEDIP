# Lorenzo Bonasera 2024
from holoviews.plotting.bokeh.styles import marker
from scipy import stats
#from sympy.stats.sampling.sample_scipy import scipy
from mip import generateProblem, generateProblemSoft, generateCSP
import numpy as np
from warm_start_tree import *
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

    return 1
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
        # plotPaths(loss, freq, labels, leaves)
        best_tree, best_labels = findBestTree(X_train, y_train, trees_pathed, lambd)
        best_tree_acc = computeScoreTree(X_test, y_test, best_tree, best_labels)
        best_tree_disag = regressionDisagreement(X_test, y_test, clf, best_tree, best_labels)
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], [weights[i] for i in leaves], stats.mode(y_train)[0][0])
        return reprtrees, reprpaths, [paths[i] for i in leaves], [labels[i] for i in leaves], clf, acc, og_score, best_tree_disag, best_tree_acc


def computeMultipleSolutions(X_train, y_train, X_test, y_test, depth, seed, eps=None, leaf_nodes=None):
    n = X_train.shape[0]
    # depth = np.ceil(np.log2(leaf_nodes)).astype(int)

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    # og_score = 1 - clf.score(X_test, y_test)
    # print("fraction of unexplained variance RF:", round(og_score, 2))
    og_score = accuracy_score(y_pred, y_test)
    print("mean squared error RF:", round(og_score, 2))
    loss, samples, labels, paths, freq = computeLoss(X_train, y_train, paths, eps)

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
    if leaf_nodes is not None:
        model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, 1)

        # let's try different solutions and see what happens
        #model.setParam("PoolSearchMode", 2)
        #model.setParam("PoolSolutions", 10)
        model.optimize()

        gradient_of_obj = []
        gradient_of_acc = []

        for i in range(model.SolCount):
            model.setParam("SolutionNumber", i)
            sol = model.getAttr("Xn", model.getVars())
            objval = model.getAttr("PoolObjVal")
            leaves = [idx for idx, x in enumerate(sol) if x > 0]
            gradient_of_obj.append(objval)

            #print("leaves:", leaves)
            reprtrees = checkTrees([paths[i] for i in leaves], trees_noded)
            reprpaths = checkTreePaths([paths[i] for i in leaves], trees_pathed)
            #plotPaths(loss, freq, labels, leaves)
            #best_tree, best_labels = findBestTree(X_train, y_train, trees_pathed)
            #best_tree_acc = computeScore(X_test, y_test, best_tree, best_labels)
            #best_tree_disag = regressionDisagreement(X_test, y_test, clf, best_tree, best_labels)
            acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves])
            gradient_of_acc.append(acc)

        print(gradient_of_obj)
        print(gradient_of_acc)
        print("max obj:", max(gradient_of_obj))
        print("min obj:", min(gradient_of_obj))
        print("best mse;", min(gradient_of_acc))
        print("worst mse;", max(gradient_of_acc))


def computeDifferentLambda(X_train, y_train, X_test, y_test, depth, seed, eps=None, leaf_nodes=None):
    n = X_train.shape[0]
    # depth = np.ceil(np.log2(leaf_nodes)).astype(int)

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    # og_score = 1 - clf.score(X_test, y_test)
    # print("fraction of unexplained variance RF:", round(og_score, 2))
    og_score = accuracy_score(y_pred, y_test)
    print("mean squared error RF:", round(og_score, 2))
    loss, samples, labels, paths, freq = computeLoss(X_train, y_train, paths, eps)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # retrieve solutions
    acc_values = []
    leaves_values = []
    model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, 1)

    # check upper bound on l
    for lambd in np.arange(0.1, 1.0, 0.01):
        model.setObjective((lambd)*quicksum(freq[j] * z[j] for j in range(L)) - (1-lambd) * quicksum(loss[j] * z[j] for j in range(L)),
                       sense=GRB.MAXIMIZE)
        model.optimize()
        leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0]
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves])
        acc_values.append(acc)
        leaves_values.append(leaves)

    #plt.plot(np.arange(0.1, 1.0, 0.01), acc_values)
    #plt.xlabel('Lambda')
    plt.boxplot(acc_values)
    plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.ylabel('MSE')
    plt.title('Bi-objective variations')
    plt.tight_layout()

    # Show the plot
    plt.show()
    #print("STD on MSE values:", np.std(acc_values))


def computeClustering(X_train, y_train, depth, seed, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]
    #depth = np.ceil(np.log2(leaf_nodes)).astype(int)

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, Nmin)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    if leaf_nodes is not None:
        model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, lambd)
        model.optimize()

        leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0.5]
        return A[:, leaves], None, [paths[i] for i in leaves], [labels[i] for i in leaves]


def computeBounds(X_train, y_train, depth, seed, Nmin=None):
    n = X_train.shape[0]

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, depth, seed)
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, Nmin)

    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    # check upper bound on l
    model, z = generateCSP(L, n, A)
    model.setObjective(z.sum(), sense=gp.GRB.MAXIMIZE)
    model.setParam('OutputFlag', 0)
    model.update()
    model.optimize()
    upper_l = int(model.getObjective().getValue())
    test_uppel_l = max(len(tree) for tree in trees_pathed)

    model.setObjective(z.sum(), sense=gp.GRB.MINIMIZE)
    model.update()
    model.optimize()
    lower_l = int(model.getObjective().getValue())
    test_lower_l = min(len(tree) for tree in trees_pathed)

    return upper_l, test_uppel_l, lower_l, test_lower_l