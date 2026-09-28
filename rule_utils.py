"""Shared rule operations for the three experiment folders (no installation needed).

A rule is a root-to-leaf list of (feature, threshold, direction) conditions.
Features are column indices for tabular data and tuples of values for shapelets.
Directions are 'L' (<=) or 'R' (>). An empty rule matches every sample.
"""

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def majority_label(y):
    """Return the most frequent label; break ties by first occurrence."""
    return Counter(np.asarray(y).reshape(-1).tolist()).most_common(1)[0][0]


def extract_paths(left, right, thresholds, get_feature):
    """Traverse a binary tree once, including a root that is itself a leaf."""
    paths = []
    stack = [(0, [])]
    while stack:
        node, path = stack.pop()
        if left[node] == -1:
            paths.append(path)
            continue
        feature = get_feature(node)
        threshold = float(thresholds[node])
        stack.append((int(right[node]), path + [(feature, threshold, "R")]))
        stack.append((int(left[node]), path + [(feature, threshold, "L")]))
    return paths


def tabular_forest_paths(clf, X, y):
    """Fit a sklearn forest and extract path/condition metadata for each tree."""
    clf.fit(X, np.asarray(y).reshape(-1))
    trees_pathed, trees_noded = [], []
    for estimator in clf.estimators_:
        tree = estimator.tree_
        paths = extract_paths(
            tree.children_left,
            tree.children_right,
            tree.threshold,
            lambda node: int(tree.feature[node]),
        )
        trees_pathed.append(paths)
        trees_noded.append({condition for path in paths for condition in path})
    return [path for tree in trees_pathed for path in tree], clf, trees_pathed, trees_noded


def min_distance(shapelet, data, metric):
    """Minimum distance to a contiguous, same-length subsequence."""
    shapelet, data = np.asarray(shapelet), np.asarray(data)
    if shapelet.ndim != 1 or data.ndim != 1 or not 0 < len(shapelet) <= len(data):
        raise ValueError("A shapelet must be nonempty and no longer than the series.")
    return min(
        metric(shapelet, data[i : i + len(shapelet)]) for i in range(len(data) - len(shapelet) + 1)
    )


def rule_mask(X, path, metric=None):
    """Evaluate a rule over rows; shapelet distances use the supplied metric."""
    X = np.asarray(X)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, two-dimensional numeric array.")
    # Match sklearn's input precision when evaluating extracted tabular paths.
    if metric is None:
        X = X.astype(np.float32, copy=False)
    mask = np.ones(len(X), dtype=bool)
    for feature, threshold, direction in path:
        if direction not in {"L", "R"}:
            raise ValueError("Rule direction must be 'L' or 'R'.")
        if metric is None:
            values = X[:, feature]
        else:
            values = np.array([min_distance(feature, row, metric) for row in X])
        # Keep the tree threshold at double precision after input conversion.
        values = values.astype(float, copy=False)
        mask &= values <= threshold if direction == "L" else values > threshold
    return mask


def condition_key(condition, include_direction=True):
    """Make tabular and array-valued shapelet conditions hashable."""
    feature, threshold, direction = condition
    if not np.isscalar(feature):
        feature = tuple(np.asarray(feature).tolist())
    key = (feature, float(threshold))
    return key + (direction,) if include_direction else key


def stability_scores(paths, ensemble_weights=None):
    """Sum Dice similarities of splitting sets, ignoring order and direction.

    Indexed counts avoid an explicit quadratic comparison of every pair.
    Duplicate candidate paths remain distinct. Empty rules have zero stability.
    """
    splits = [{condition_key(node, False) for node in path} for path in paths]
    weights = (
        np.ones(len(paths))
        if ensemble_weights is None
        else np.asarray(ensemble_weights, dtype=float)
    )
    if weights.shape != (len(paths),) or not np.isfinite(weights).all():
        raise ValueError("Ensemble weights must be a finite vector with one value per path.")
    weights = np.abs(weights)
    counts = defaultdict(Counter)
    for splitting_set, weight in zip(splits, weights):
        for splitting in splitting_set:
            counts[splitting][len(splitting_set)] += weight
    scores = []
    for splitting_set, weight in zip(splits, weights):
        score = sum(
            2 * count / (len(splitting_set) + length)
            for splitting in splitting_set
            for length, count in counts[splitting].items()
        )
        scores.append(max(0.0, score - weight * bool(splitting_set)))
    return np.asarray(scores)


def normalize(values):
    """Scale to [0, 1]; constant vectors map to zero (no relative preference)."""
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def compute_rule_statistics(X, y, paths, task, min_support=None, metric=None, *, scale=True):
    """Compute loss, covered rows, labels, filtered paths, stability, support.

    Raw losses follow Equation (7): misclassification count or within-rule MSE.
    Stability compares against the entire ensemble, before support filtering.
    Set scale=False to return raw coefficients for subsequent support selection.
    """
    X, y = np.asarray(X), np.asarray(y).reshape(-1)
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("X and y must contain the same nonzero number of rows.")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")
    if min_support is not None and not 0 <= min_support <= 1:
        raise ValueError("Nmin must be a fraction between 0 and 1.")
    minimum = max(1, int(np.ceil((min_support or 0) * len(X))))
    samples, kept_paths, labels, losses, weights = [], [], [], [], []
    all_stability = stability_scores(paths)
    kept_indices = []
    for index, path in enumerate(paths):
        sample = np.flatnonzero(rule_mask(X, path, metric))
        if len(sample) < minimum:
            continue
        targets = y[sample]
        if task == "classification":
            label = majority_label(targets)
            loss = float(np.count_nonzero(targets != label))
        else:
            label = float(np.mean(targets))
            loss = float(np.mean((targets - label) ** 2))
        samples.append(sample)
        kept_paths.append(path)
        labels.append(label)
        losses.append(loss)
        weights.append(len(sample))
        kept_indices.append(index)
    if not kept_paths:
        raise ValueError("No candidate rules remain. Reduce Nmin or check the data.")
    losses = np.asarray(losses)
    stability = all_stability[kept_indices]
    return (
        normalize(losses) if scale else losses,
        samples,
        labels,
        kept_paths,
        normalize(stability) if scale else stability,
        weights,
    )


def predict_rules(X, paths, labels, weights, default, metric=None):
    """Use the matching rule with largest training support; ties use first rule.

    Exact coverage is imposed only on training rows. Uncovered test rows use
    the supplied training majority class or training response mean.
    """
    if not paths or len(paths) != len(labels) or len(paths) != len(weights):
        raise ValueError("paths, labels and weights must have equal, nonzero length.")
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Rule supports must be finite and positive.")
    predictions = np.full(len(X), default, dtype=np.asarray(list(labels) + [default]).dtype)
    best_support = np.full(len(X), -np.inf)
    for path, label, weight in zip(paths, labels, weights):
        chosen = rule_mask(X, path, metric) & (weight > best_support)
        predictions[chosen] = label
        best_support[chosen] = weight
    return predictions


def score_rules(X, y, paths, labels, weights, default, task, metric=None):
    predictions = predict_rules(X, paths, labels, weights, default, metric)
    score = accuracy_score if task == "classification" else mean_squared_error
    return float(score(y, predictions))


def represented_paths(paths, trees):
    """Fraction of trees with at least one selected complete path."""
    selected = {tuple(condition_key(node) for node in path) for path in paths}
    return (
        sum(
            any(tuple(condition_key(node) for node in path) in selected for path in tree)
            for tree in trees
        )
        / len(trees)
        if trees
        else 0.0
    )


def represented_nodes(paths, trees):
    """Fraction of trees sharing a complete condition, including its sign."""
    selected = {condition_key(node) for path in paths for node in path}
    return (
        sum(bool(selected & {condition_key(node) for node in tree}) for tree in trees) / len(trees)
        if trees
        else 0.0
    )


def feature_f1(clf, paths):
    """Feature-set F1 against the top ceil(5% * p) forest features by importance."""
    importance = clf.feature_importances_
    count = max(1, int(np.ceil(0.05 * len(importance))))
    relevant = set(np.argsort(-np.asarray(importance), kind="stable")[:count])
    extracted = {node[0] for path in paths for node in path}
    return 2 * len(relevant & extracted) / (len(relevant) + len(extracted))
