"""Tabular regression experiment entry points."""

from rule_extraction import computePaths
from distillation import prepare_rules, solve_rules, validate_rules


def prepare(X, y, depth, seed, Nmin=None, n_estimators=500):
    """Fit and preprocess once; no validation or test targets enter fitting."""
    extracted = computePaths(X, y, n_estimators, depth, seed)
    return prepare_rules(X, y, extracted, "regression", Nmin)


def fit(
    X, y, depth=3, seed=0, lambd=0.5, leaf_nodes=15, Nmin=None, n_estimators=500, **solver_options
):
    """Return predictions, scores, selected rules, and solver diagnostics."""
    prepared = prepare(X, y, depth, seed, Nmin, n_estimators)
    return solve_rules(prepared, leaf_nodes, lambd, seed=seed, **solver_options)


def computeAll(
    X_train,
    y_train,
    X_test,
    y_test,
    depth,
    seed,
    lambd=0.5,
    leaf_nodes=15,
    Nmin=None,
    n_estimators=500,
    **solver_options,
):
    """Return the seven-value experiment tuple."""
    result = fit(
        X_train, y_train, depth, seed, lambd, leaf_nodes, Nmin, n_estimators, **solver_options
    )
    return result.as_tuple(X_test, y_test)


def computeValidation(
    X_train,
    y_train,
    X_test,
    y_test,
    depth,
    seed,
    lambd=0.5,
    leaf_nodes=None,
    Nmin=None,
    n_estimators=500,
    **solver_options,
):
    """Return (selected budget, score); X_test/y_test here are validation data."""
    prepared = prepare(X_train, y_train, depth, seed, Nmin, n_estimators)
    return validate_rules(prepared, X_test, y_test, leaf_nodes, lambd, seed=seed, **solver_options)


def computeTest(
    X_train,
    y_train,
    X_test,
    y_test,
    depth,
    leaf_nodes,
    seed,
    lambd=0.5,
    Nmin=None,
    n_estimators=500,
    **solver_options,
):
    """Return rule MSE, forest MSE, and optimization runtime."""
    result = fit(
        X_train, y_train, depth, seed, lambd, leaf_nodes, Nmin, n_estimators, **solver_options
    )
    return result.score(X_test, y_test), result.forest_score(X_test, y_test), result.solver_runtime
