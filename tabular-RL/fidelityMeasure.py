# Lorenzo Bonasera 2024

from mip import generateProblem, generateProblemSoft, generateCSP
import numpy as np
from warm_start_tree import computePaths, computeLoss, computeScore, computeSample
import gurobipy as gp
from sklearn.metrics import accuracy_score
from itertools import chain


def regressionDisagreement(X, y, clf, paths, labels):
    n = X.shape[0]

    # compute explainations
    y_pred = np.zeros(len(y))
    y_opaque = clf.predict(X)
    for p, path in enumerate(paths):
        for i in range(n):
            if computeSample(X[i, :], path):
                y_pred[i] = labels[p]

    return 1 - accuracy_score(y_opaque, y_pred)


def F1ComplCorr(clf, paths):
    opaque_importance = clf.feature_importances_
    percentile = np.percentile(opaque_importance, 95)
    x_t = set([idx for idx, x in enumerate(opaque_importance) if x >= percentile])

    # compute extracted features
    x_e = set([x[0] for x in list(chain.from_iterable(paths))])

    # compute precision and recall
    if len(x_e) == 0:
        return 0

    recall = len(x_t & x_e) / len(x_t)
    prec = len(x_t & x_e) / len(x_e)

    if prec + recall == 0:
        return 0

    return 2 * (prec * recall) / (prec + recall)