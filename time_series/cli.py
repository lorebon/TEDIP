"""Command-line runner for fixed-budget and cross-validated temporal experiments."""

import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiment import prepare
from preprocessing import preprocessTest, preprocessTrain
from distillation import solve_rules, cross_validate_budgets
from experiment_utils import (
    add_common_arguments,
    positive_int,
    rule_artifact,
    run_seeds,
    save_results,
    solver_options,
    validate_arguments,
)


def load_data(dataset):
    """Use official UCR splits, a local NPZ split, or a small offline demo."""
    if dataset == "builtin:synthetic":
        rng = np.random.default_rng(42)
        X = rng.normal(scale=0.25, size=(60, 24))
        y = np.arange(60) % 2
        X[y == 1, 8:14] += 2
        return X[:40], y[:40], X[40:], y[40:]
    if dataset.lower().endswith(".npz"):
        with np.load(dataset, allow_pickle=False) as data:
            required = ("X_train", "y_train", "X_test", "y_test")
            if not all(key in data for key in required):
                raise ValueError("NPZ requires X_train, y_train, X_test, y_test arrays.")
            return tuple(data[key] for key in required)
    from sktime.datasets import load_UCR_UEA_dataset

    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split="train", return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split="test", return_X_y=True)
    return X_train, y_train, X_test, y_test


def compute_run(request):
    X_train, y_train, X_test, y_test, classes, seed, args, validate = request
    n_shapelets = (
        max(1, int(np.sqrt(X_train.shape[1] ** 2 / 2))) if args.r == "def." else int(args.r)
    )
    options = solver_options(args, seed)
    budget = args.leaves
    validation = None
    validation_bounds = None
    prepared = prepare(
        X_train,
        y_train,
        args.depth,
        seed,
        args.l,
        args.u,
        n_shapelets,
        args.min_support,
        args.trees,
    )
    if validate:
        if min(np.bincount(y_train)) < args.folds:
            raise ValueError("Each training class needs at least --folds samples.")
        folds = []
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        for train, val in splitter.split(X_train, y_train):
            fold = prepare(
                X_train[train],
                y_train[train],
                args.depth,
                seed,
                args.l,
                args.u,
                n_shapelets,
                args.min_support,
                args.trees,
            )
            folds.append((fold, X_train[val], y_train[val]))
        budget, validation, validation_bounds = cross_validate_budgets(
            folds, prepared, args.budgets, args.lambd, **options
        )
    result = solve_rules(prepared, budget, args.lambd, **options)
    row = {
        "Seed": seed,
        "Represented trees": result.represented_trees,
        "Represented paths": result.represented_paths,
        "Leaves": len(result.paths),
        "Rule budget": budget,
        "Accuracy": result.score(X_test, y_test),
        "Full Model": result.forest_score(X_test, y_test),
        "Disagreement": result.disagreement(X_test),
        "Solver status": result.solver_status,
        "MIP gap": result.solver_gap,
        "Solver seconds": result.solver_runtime,
    }
    artifact = dict(
        seed=seed,
        classes=classes,
        validation=validation,
        validation_bounds=validation_bounds,
        n_shapelets=n_shapelets,
        **rule_artifact(result),
    )
    return row, artifact


def main(validate=False, argv=None):
    parser = argparse.ArgumentParser(
        description="TEDIP time-series classification with Euclidean shapelet rules.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("dataset", help="UCR dataset name, split .npz file, or builtin:synthetic.")
    parser.add_argument(
        "l", type=float, nargs="?", default=0.0, help="Minimum shapelet-length fraction."
    )
    parser.add_argument(
        "u", type=float, nargs="?", default=1.0, help="Maximum shapelet-length fraction."
    )
    parser.add_argument(
        "r", nargs="?", default="def.", help="Candidate shapelets per node, or def."
    )
    add_common_arguments(parser, default_leaves=6)
    if validate:
        parser.add_argument("--folds", type=positive_int, default=5)
        parser.add_argument(
            "--budgets",
            type=positive_int,
            nargs="+",
            default=None,
            help="Optional restricted budget grid; default is exhaustive search using exact bounds.",
        )
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if not 0 <= args.l < args.u <= 1:
        parser.error("shapelet fractions must satisfy 0 <= l < u <= 1")
    if args.r != "def." and (not args.r.isdigit() or int(args.r) < 1):
        parser.error("r must be a positive integer or def.")
    if validate and args.folds < 2:
        parser.error("--folds must be at least 2")
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    X_train, y_train, classes, _, _ = preprocessTrain(X_train, y_train)
    X_test, y_test = preprocessTest(X_test, y_test, classes)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test series must have the same length.")
    requests = [
        (X_train, y_train, X_test, y_test, classes, seed, args, validate) for seed in args.seeds
    ]
    results = run_seeds(compute_run, args.jobs, requests)
    return save_results(results, args, "temporal-validated" if validate else "temporal")
