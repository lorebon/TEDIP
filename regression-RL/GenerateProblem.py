from mip import generateProblem, generateProblemSoft, generateCSP
import numpy as np
from warm_start import computePaths, computeLoss, computeScore
import gurobipy as gp
from sklearn.metrics import mean_squared_error


def computeValidation(X_train, y_train, X_test, y_test, depth, seed, eps=None):
    n = X_train.shape[0]
    #depth = np.ceil(np.log2(leaf_nodes)).astype(int)

    # start random forest
    paths, clf = computePaths(X_train, y_train, 500, depth, seed)
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
        model.setObjective(gp.quicksum(freq[j] * z[j] for j in range(L)) - gp.quicksum(loss[j] * z[j] for j in range(L)),
                           sense=gp.GRB.MAXIMIZE)
        model.update()
        model.optimize()

        if model.Status == 2:
            leaves = []
            # retrieve solution
            for i in range(L):
                var = model.getVarByName("z[{}]".format(i))
                if int(var.x) == 1:
                    leaves.append(i)

            acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves])
            accs.append(acc)
            leaves_val.append(leaves)
    best_score = np.min(accs)
    best_leaves = leaves_val[np.argmin(accs)]
    return len(best_leaves), best_score



def computeTest(X_train, y_train, X_test, y_test, leaf_nodes, seed, eps=None):
    n = X_train.shape[0]
    #depth = np.ceil(np.log2(leaf_nodes)).astype(int)
    depth = 3

    # start random forest
    paths, clf = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    og_score = mean_squared_error(y_pred, y_test)
    loss, samples, labels, paths, freq = computeLoss(X_train, y_train, paths, eps)

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

        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves])
    else:
        acc = 0
    return acc, og_score, model.Runtime



def computeAll(X_train, y_train, X_test, y_test, depth, seed, eps=None, leaf_nodes=None):
    n = X_train.shape[0]
    #depth = np.ceil(np.log2(leaf_nodes)).astype(int)

    # start random forest
    paths, clf = computePaths(X_train, y_train, 500, depth, seed)
    y_pred = clf.predict(X_test)
    #og_score = 1 - clf.score(X_test, y_test)
    #print("fraction of unexplained variance RF:", round(og_score, 2))
    og_score = mean_squared_error(y_pred, y_test)
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
        model.optimize()
        leaves = []
        for i in range(L):
            var = model.getVarByName("z[{}]".format(i))
            if int(var.x) == 1:
                leaves.append(i)

        print("leaves:", leaves)
        acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves])
        return acc, [paths[i] for i in leaves], og_score, model.Runtime

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
            acc = computeScore(X_test, y_test, [paths[i] for i in leaves], [labels[i] for i in leaves])
            accs.append(acc)
            leaves_val.append(leaves)
    print(accs)
    accs = np.asarray(accs)

    ### CARE: R^2 MAX, 1 - R^2 or MSE MIN
    best_score = np.min(accs)
    best_leaves = leaves_val[np.argmin(accs)]
    return best_score, [paths[i] for i in best_leaves], og_score, model.Runtime







