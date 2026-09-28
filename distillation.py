"""Train/extract, solve, and evaluate without mixing test data into fitting."""

from dataclasses import dataclass, replace

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from optimization import coverage_matrix, generateCSP, generateProblemSoft, selected_rules
from rule_utils import (
    compute_rule_statistics,
    majority_label,
    normalize,
    predict_rules,
    represented_nodes,
    represented_paths,
)


@dataclass
class PreparedRules:
    """Training-only quantities reused when evaluating several rule budgets."""

    forest: object
    paths: list
    labels: list
    weights: list
    loss: np.ndarray
    stability: np.ndarray
    coverage: object
    trees_pathed: list
    trees_noded: list
    default: object
    task: str
    metric: object = None
    raw_loss: object = None
    raw_stability: object = None


@dataclass
class DistillationResult:
    """Selected rules, prediction policy, and solver diagnostics."""

    paths: list
    labels: list
    weights: list
    forest: object
    default: object
    task: str
    represented_trees: float
    represented_paths: float
    solver_status: int
    solver_gap: float
    solver_runtime: float
    metric: object = None

    def predict(self, X):
        return predict_rules(X, self.paths, self.labels, self.weights, self.default, self.metric)

    def score(self, X, y):
        """Accuracy for classification; MSE (lower is better) for regression."""
        metric = accuracy_score if self.task == "classification" else mean_squared_error
        return float(metric(y, self.predict(X)))

    def forest_score(self, X, y):
        metric = accuracy_score if self.task == "classification" else mean_squared_error
        return float(metric(y, self.forest.predict(X)))

    def disagreement(self, X):
        """Compare forest predictions with the same predictions used for scoring."""
        predictions = self.predict(X)
        forest_predictions = self.forest.predict(X)
        if self.task == "classification":
            return float(np.mean(predictions != forest_predictions))
        return float(mean_squared_error(forest_predictions, predictions))

    def as_tuple(self, X, y):
        """Seven-value result returned by computeAll."""
        return (
            self.represented_trees,
            self.represented_paths,
            self.paths,
            self.labels,
            self.forest,
            self.score(X, y),
            self.forest_score(X, y),
        )


def prepare_rules(X, y, extracted, task, Nmin=None, metric=None):
    """Prepare a forest/path tuple returned by a folder's computePaths."""
    paths, forest, trees_pathed, trees_noded = extracted
    loss, samples, labels, paths, stability, weights = compute_rule_statistics(
        X, y, paths, task, Nmin, metric, scale=False
    )
    default = majority_label(y) if task == "classification" else float(np.mean(y))
    return PreparedRules(
        forest,
        paths,
        labels,
        weights,
        normalize(loss),
        normalize(stability),
        coverage_matrix(samples, len(X)),
        trees_pathed,
        trees_noded,
        default,
        task,
        metric,
        loss,
        stability,
    )


def filter_support(prepared, min_support):
    """Filter cached training statistics and rescale the eligible coefficients."""
    if not 0 <= min_support <= 1:
        raise ValueError("Minimum support must lie in [0, 1].")
    if prepared.raw_loss is None or prepared.raw_stability is None:
        raise ValueError("Support filtering requires raw training statistics.")
    minimum = max(1, int(np.ceil(min_support * prepared.coverage.shape[0])))
    indices = np.flatnonzero(np.asarray(prepared.weights) >= minimum)
    if not len(indices):
        raise ValueError("No candidate rules remain. Reduce minimum support.")
    loss, stability = prepared.raw_loss[indices], prepared.raw_stability[indices]
    return replace(
        prepared,
        paths=[prepared.paths[j] for j in indices],
        labels=[prepared.labels[j] for j in indices],
        weights=[prepared.weights[j] for j in indices],
        coverage=prepared.coverage[:, indices],
        loss=normalize(loss),
        stability=normalize(stability),
        raw_loss=loss,
        raw_stability=stability,
    )


def _result(prepared, model, z):
    indices = selected_rules(model, z)
    paths = [prepared.paths[j] for j in indices]
    labels = [prepared.labels[j] for j in indices]
    return DistillationResult(
        paths,
        labels,
        [prepared.weights[j] for j in indices],
        prepared.forest,
        prepared.default,
        prepared.task,
        represented_nodes(paths, prepared.trees_noded),
        represented_paths(paths, prepared.trees_pathed),
        int(model.Status),
        float(model.MIPGap),
        float(model.Runtime),
        prepared.metric,
    )


def solve_rules(prepared, leaf_nodes=15, lambd=0.5, **solver_options):
    """Solve the set partition and release Gurobi resources even on failure."""
    n, L = prepared.coverage.shape
    model, z = generateProblemSoft(
        L,
        n,
        prepared.coverage,
        leaf_nodes,
        prepared.loss,
        prepared.stability,
        lambd,
        **solver_options,
    )
    try:
        model.optimize()
        return _result(prepared, model, z)
    finally:
        model.dispose()


def rule_count_bounds(prepared, **solver_options):
    """Solve Equation (10) in both directions to obtain exact cardinalities."""
    import gurobipy as gp

    n, L = prepared.coverage.shape
    model, z = generateCSP(L, n, prepared.coverage, **solver_options)
    bounds = []
    try:
        for sense in (gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE):
            model.setObjective(z.sum(), sense)
            model.optimize()
            selected_rules(model, z, require_optimal=True)
            bounds.append(int(round(model.ObjVal)))
    finally:
        model.dispose()
    return tuple(bounds)


def budget_scores(prepared, X_val, y_val, budgets, lambd=0.5, **solver_options):
    """Evaluate budgets by reoptimizing one MIP with a changing cardinality RHS."""
    import gurobipy as gp

    budgets = sorted(set(budgets))
    if not budgets or any(not isinstance(b, (int, np.integer)) or b < 1 for b in budgets):
        raise ValueError("Validation budgets must be positive integers and nonempty.")
    n, L = prepared.coverage.shape
    model, z = generateProblemSoft(
        L,
        n,
        prepared.coverage,
        budgets[0],
        prepared.loss,
        prepared.stability,
        lambd,
        **solver_options,
    )
    scores = {}
    try:
        for budget in budgets:
            model.getConstrByName("card").RHS = budget
            model.optimize()
            if model.Status == gp.GRB.INFEASIBLE:
                continue
            result = _result(prepared, model, z)
            scores[budget] = result.score(X_val, y_val)
    finally:
        model.dispose()
    return scores


def validate_rules(
    prepared, X_val, y_val, leaf_nodes=None, lambd=0.5, budgets=None, **solver_options
):
    """Section 4.3 search; return (allowed budget, score), smaller budget on ties."""
    from optimization import NoSolutionError

    if leaf_nodes is not None:
        result = solve_rules(prepared, leaf_nodes, lambd, **solver_options)
        return leaf_nodes, result.score(X_val, y_val)
    if budgets is None:
        lower, upper = rule_count_bounds(prepared, **solver_options)
        budgets = range(lower, upper + 1)
    scores = budget_scores(prepared, X_val, y_val, budgets, lambd, **solver_options)
    if not scores:
        raise NoSolutionError("No validation budget gave a feasible partition.")
    sign = 1 if prepared.task == "classification" else -1
    budget = max(scores, key=lambda b: (sign * scores[b], -b))
    return budget, scores[budget]


def cross_validate_budgets(folds, full_prepared, budgets=None, lambd=0.5, **solver_options):
    """Compare one exhaustive budget range across folds, then permit full refit.

    Each fold is (PreparedRules, X_validation, y_validation). Automatic search
    uses exact bounds from every training fold and the full training set. The
    shared range starts at the largest lower bound and ends at the largest upper
    bound; larger budgets remain valid upper bounds in folds with fewer leaves.
    Mean fold scores and smaller-budget tie breaking define the CV convention.
    """
    from optimization import NoSolutionError

    if not folds:
        raise ValueError("At least one validation fold is required.")
    bounds = None
    if budgets is None:
        bounds = [rule_count_bounds(p, **solver_options) for p, _, _ in folds]
        bounds.append(rule_count_bounds(full_prepared, **solver_options))
        lower = max(pair[0] for pair in bounds)
        upper = max(pair[1] for pair in bounds)
        budgets = range(lower, upper + 1)
    budgets = sorted(set(budgets))
    fold_scores = [budget_scores(p, X, y, budgets, lambd, **solver_options) for p, X, y in folds]
    scores = {b: [scores[b] for scores in fold_scores if b in scores] for b in budgets}
    means = {b: float(np.mean(values)) for b, values in scores.items() if len(values) == len(folds)}
    if not means:
        raise NoSolutionError(
            "No budget was feasible in every fold. Increase the budgets or reduce minimum support."
        )
    sign = 1 if full_prepared.task == "classification" else -1
    budget = max(means, key=lambda b: (sign * means[b], -b))
    details = {str(b): dict(fold_scores=scores[b], mean=means.get(b)) for b in budgets}
    return budget, details, bounds
