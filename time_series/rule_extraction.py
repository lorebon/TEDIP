"""Shapelet-forest path extraction and rule evaluation."""

from pathlib import Path
import sys

# Resolve shared helpers relative to this file, independent of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from wildboar.ensemble import ShapeletForestClassifier
from rule_utils import (
    compute_rule_statistics,
    extract_paths,
    min_distance,
    predict_rules,
    represented_nodes,
    represented_paths,
    rule_mask,
    score_rules,
)


def computePaths(X, y, n_estimators, min_size, max_size, n_shap, max_depth, seed):
    """Train an Euclidean shapelet forest and retain metadata for every tree.

    This adapter reads Wildboar's tree representation; supported/tested with
    Wildboar 1.2.1. Only univariate, equal-length series are supported.
    """
    clf = ShapeletForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        min_shapelet_size=min_size,
        max_shapelet_size=max_size,
        n_shapelets=n_shap,
        max_depth=max_depth,
        metric="euclidean",
        n_jobs=1,
    )
    clf.fit(X, np.asarray(y).reshape(-1))
    trees_pathed, trees_noded = [], []
    for estimator in clf.estimators_:
        tree = estimator.tree_
        attributes = tree.attribute

        def shapelet_at(node):
            try:
                dimension, (_, values) = attributes[node]
                if dimension != 0:
                    raise ValueError("Only univariate shapelet trees are supported.")
                return tuple(np.asarray(values, dtype=float).tolist())
            except (TypeError, IndexError) as error:
                raise RuntimeError(
                    "Unsupported Wildboar tree layout; use wildboar==1.2.1."
                ) from error

        paths = extract_paths(tree.left, tree.right, tree.threshold, shapelet_at)
        trees_pathed.append(paths)
        trees_noded.append({condition for path in paths for condition in path})
    return [path for tree in trees_pathed for path in tree], clf, trees_pathed, trees_noded


def minDist(shapelet, data, metric):
    return min_distance(shapelet, data, metric)


def computeSample(data, path, metric, idx=0):
    return bool(rule_mask(np.asarray(data).reshape(1, -1), path[idx:], metric)[0])


def computeLoss(X, y, paths, metric, Nmin=None):
    return compute_rule_statistics(X, y, paths, "classification", Nmin, metric)


def predictRules(X, paths, labels, metric, weights, mode):
    return predict_rules(X, paths, labels, weights, mode, metric)


def computeScore(X, y, paths, labels, metric, weights, mode):
    return score_rules(X, y, paths, labels, weights, mode, "classification", metric)


checkTrees = represented_nodes
checkTreePaths = represented_paths
