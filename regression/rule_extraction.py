"""Tabular regression: forest paths, rule statistics, and predictions."""

from pathlib import Path
import sys

# Keep these files runnable directly without installing this repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rule_utils import (
    compute_rule_statistics,
    tabular_forest_paths,
    predict_rules,
    represented_nodes,
    represented_paths,
    rule_mask,
    score_rules,
)


def computePaths(X, y, n_estimators, max_depth, seed):
    """Train a forest and retain each tree's complete root-to-leaf paths."""
    clf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=1
    )
    return tabular_forest_paths(clf, X, y)


def computeSample(data, path, idx=0):
    """Whether one row satisfies all conditions from idx onward."""
    return bool(rule_mask(np.asarray(data).reshape(1, -1), path[idx:])[0])


def computeLoss(X, y, paths, Nmin=None):
    """Return loss, covered rows, predictions, paths, stability, and supports."""
    return compute_rule_statistics(X, y, paths, "regression", Nmin)


def predictRules(X, paths, labels, weights, default):
    """Use the training fallback for uncovered rows."""
    return predict_rules(X, paths, labels, weights, default)


def computeScore(X, y, paths, labels, weights, default=None):
    """Return MSE with training-mean fallback for uncovered rows.

    Supports weight a partition of training rows, so weighting the rule means
    recovers the training response mean. Otherwise supply that mean explicitly.
    """
    if default is None:
        default = float(np.average(labels, weights=weights))
    return score_rules(X, y, paths, labels, weights, default, "regression")


checkTrees = represented_nodes
checkTreePaths = represented_paths
