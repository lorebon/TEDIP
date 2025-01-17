# Lorenzo Bonasera 2024
from scipy import stats
from mip import generateProblem, generateProblemSoft, generateCSP
import numpy as np
from metrics import euclidean, scaled_euclidean, minkowski, scaled_minkowski
from ShapeletNode import minDist
from warm_start_tree import *
import gurobipy as gp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from gurobipy import GRB, quicksum


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


def computeValidation(X_train, y_train, X_test, y_test, depth, seed, minshap, maxshap, n_shap, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]
    #depth = np.ceil(np.log2(leaf_nodes)).astype(int)
    metric = euclidean

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, minshap, maxshap, n_shap, depth, seed)
    #y_pred = clf.predict(X_test)
    og_score = clf.score(X_test, y_test)*100
    print("accuracy RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, metric, Nmin)
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
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric, [weights[i] for i in leaves], stats.mode(y_train)[0][0])
        return acc

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
    #lower_l = min(len(tree) for tree in trees_pathed)
    #test_uppel_l = max(len(tree) for tree in trees_pathed)
    #if test_uppel_l != upper_l:
        #print("Upper bound: {}, heur bound: {}".format(upper_l, test_uppel_l))
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

            #print("leaves:", leaves)
            acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric,
                               [weights[i] for i in leaves], stats.mode(y_train)[0][0])
            accs.append(acc)
            leaves_val.append(leaves)

    #print(accs)
    accs = np.asarray(accs)
    best_score = np.max(accs)
    best_leaves = leaves_val[np.argmax(accs)]
    return len(best_leaves), best_score


def computeAll(X_train, y_train, X_test, y_test, depth, seed, minshap, maxshap, n_shap, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]
    metric = euclidean

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, minshap, maxshap, n_shap, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = accuracy_score(y_pred, y_test)
    print("Accuracy RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, metric, Nmin)

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
        #best_tree, best_labels = findBestTree(X_train, y_train, trees_pathed, metric, lambd)
        #best_tree_acc = computeScoreTree(X_test, y_test, best_tree, best_labels, metric)
        #best_tree_disag = regressionDisagreement(X_test, y_test, clf, best_tree, best_labels)
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric, [weights[i] for i in leaves], stats.mode(y_train)[0][0])
        return reprtrees, reprpaths, [paths[i] for i in leaves], [labels[i] for i in leaves], clf, acc, og_score, None, None


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


def computeTest(X_train, y_train, X_test, y_test, depth, seed, minshap, maxshap, n_shap, lambd=1, leaf_nodes=None, Nmin=None):
    n = X_train.shape[0]
    metric = euclidean

    # start random forest
    paths, clf, trees_pathed, trees_noded = computePaths(X_train, y_train, 500, minshap, maxshap, n_shap, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = accuracy_score(y_pred, y_test)
    print("Accuracy RF:", round(og_score, 2))
    loss, samples, labels, paths, freq, weights = computeLoss(X_train, y_train, paths, metric, Nmin)

    # prepare mip
    print("solving MIP...")
    L = len(paths)
    A = np.zeros((n, L))
    for j in range(L):
        A[samples[j], j] = 1

    model, z = generateProblemSoft(L, n, A, leaf_nodes, loss, freq, lambd)
    # model.setObjective(quicksum((lambd)*freq[j] * z[j] for j in range(L)) - (1-lambd) * quicksum(loss[j] * z[j] for j in range(L)),
    #               sense=GRB.MAXIMIZE)
    model.optimize()
    if model.Status == 2:
        leaves = [idx for idx, x in enumerate(model.getAttr("x", model.getVars())) if x > 0.5]

        # print("leaves:", leaves)
        reprtrees = checkTrees([paths[i] for i in leaves], trees_noded)
        reprpaths = checkTreePaths([paths[i] for i in leaves], trees_pathed)
        # plotPaths(loss, freq, labels, leaves)
        #best_tree, best_labels = findBestTree(X_train, y_train, trees_pathed, metric, lambd)
        #best_tree_acc = computeScoreTree(X_test, y_test, best_tree, best_labels, metric)
        #best_tree_disag = regressionDisagreement(X_test, y_test, clf, best_tree, best_labels)
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves], metric,
                           [weights[i] for i in leaves], stats.mode(y_train)[0][0])
        return reprtrees, reprpaths, [paths[i] for i in leaves], [labels[i] for i in leaves], clf, acc, og_score

    return None, None, None, None, None, None, None, None, None


