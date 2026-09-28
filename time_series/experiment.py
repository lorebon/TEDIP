"""Prepare, solve, and validate shapelet rule extraction."""

from rule_extraction import computePaths
from metrics import euclidean
from distillation import prepare_rules, solve_rules, validate_rules


def prepare(X, y, depth, seed, minshap, maxshap, n_shap, Nmin=None, n_estimators=500):
    extracted = computePaths(X, y, n_estimators, minshap, maxshap, n_shap, depth, seed)
    return prepare_rules(X, y, extracted, "classification", Nmin, euclidean)


def fit(
    X,
    y,
    depth=3,
    seed=0,
    minshap=0.0,
    maxshap=1.0,
    n_shap=10,
    lambd=0.5,
    leaf_nodes=6,
    Nmin=None,
    n_estimators=500,
    **solver_options,
):
    prepared = prepare(X, y, depth, seed, minshap, maxshap, n_shap, Nmin, n_estimators)
    return solve_rules(prepared, leaf_nodes, lambd, seed=seed, **solver_options)


def computeAll(
    X_train,
    y_train,
    X_test,
    y_test,
    depth,
    seed,
    minshap,
    maxshap,
    n_shap,
    lambd=0.5,
    leaf_nodes=6,
    Nmin=None,
    n_estimators=500,
    **solver_options,
):
    result = fit(
        X_train,
        y_train,
        depth,
        seed,
        minshap,
        maxshap,
        n_shap,
        lambd,
        leaf_nodes,
        Nmin,
        n_estimators,
        **solver_options,
    )
    return result.as_tuple(X_test, y_test)


def computeValidation(
    X_train,
    y_train,
    X_test,
    y_test,
    depth,
    seed,
    minshap,
    maxshap,
    n_shap,
    lambd=0.5,
    leaf_nodes=None,
    Nmin=None,
    n_estimators=500,
    **solver_options,
):
    """Return (budget, accuracy) using validation data only."""
    prepared = prepare(X_train, y_train, depth, seed, minshap, maxshap, n_shap, Nmin, n_estimators)
    return validate_rules(prepared, X_test, y_test, leaf_nodes, lambd, seed=seed, **solver_options)


computeTest = computeAll
