"""Shared runner for the tabular classification and regression scripts."""

import argparse

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_utils import encode_features, load_tabular
from distillation import filter_support, prepare_rules, solve_rules
from optimization import NoSolutionError
from experiment_utils import (
    add_common_arguments,
    positive_int,
    rule_artifact,
    run_seeds,
    save_results,
    solver_options,
    validate_arguments,
)
from rule_utils import feature_f1, tabular_forest_paths


def prepare_split(task, X_train, y_train, X_test, y_test, seed, args):
    """Fit every preprocessing step and the source forest on training rows only."""
    X_train, X_test, names, _ = encode_features(X_train, X_test)
    target_metadata = {}
    if task == "regression":
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()
        target_metadata = dict(target_mean=scaler.mean_[0], target_scale=scaler.scale_[0])
        forest_type = RandomForestRegressor
    else:
        encoder = LabelEncoder().fit(y_train)
        y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
        target_metadata = dict(classes=encoder.classes_.tolist())
        forest_type = RandomForestClassifier
    forest = forest_type(n_estimators=args.trees, max_depth=args.depth, random_state=seed, n_jobs=1)
    prepared = prepare_rules(X_train, y_train, tabular_forest_paths(forest, X_train, y_train), task)
    return prepared, X_test, y_test, names, target_metadata


def select_min_support(X, y, seed, args):
    """Tune regression support on training folds as specified in Section 5.1."""
    candidates = sorted(set(args.support_grid))
    scores = {value: [] for value in candidates}
    for train, val in KFold(args.folds, shuffle=True, random_state=seed).split(X):
        prepared, X_val, y_val, _, _ = prepare_split(
            "regression", X.iloc[train], y[train], X.iloc[val], y[val], seed, args
        )
        for value in candidates:
            if max(prepared.weights) < max(1, int(np.ceil(value * len(train)))):
                continue
            filtered = filter_support(prepared, value)
            try:
                result = solve_rules(
                    filtered, args.leaves, args.lambd, **solver_options(args, seed)
                )
            except NoSolutionError as error:
                if error.status != 3:
                    raise
                continue
            scores[value].append(result.score(X_val, y_val))
    means = {
        value: float(np.mean(values))
        for value, values in scores.items()
        if len(values) == args.folds
    }
    if not means:
        raise NoSolutionError(
            "No support threshold was feasible in every fold. Reduce --support-grid or increase --leaves."
        )
    selected = min(means, key=lambda value: (means[value], value))
    details = {
        str(value): dict(fold_scores=values, mean=means.get(value))
        for value, values in scores.items()
    }
    return selected, details


def compute_run(request):
    task, X, y, seed, args = request
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )
    min_support, support_validation = args.min_support, None
    if task == "regression" and min_support is None:
        min_support, support_validation = select_min_support(X_train, y_train, seed, args)
    prepared, X_test, y_test, names, target_metadata = prepare_split(
        task, X_train, y_train, X_test, y_test, seed, args
    )
    if min_support is not None:
        prepared = filter_support(prepared, min_support)
    result = solve_rules(prepared, args.leaves, args.lambd, **solver_options(args, seed))
    row = {
        "Seed": seed,
        "Minimum support": min_support if min_support is not None else 0.0,
        "Represented trees": result.represented_trees,
        "Represented paths": result.represented_paths,
        "Disagreement": result.disagreement(X_test),
        "F1": feature_f1(result.forest, result.paths),
        "Leaves": len(result.paths),
        "MSE" if task == "regression" else "Accuracy": result.score(X_test, y_test),
        "Full Model": result.forest_score(X_test, y_test),
        "Solver status": result.solver_status,
        "MIP gap": result.solver_gap,
        "Solver seconds": result.solver_runtime,
    }
    artifact = dict(
        seed=seed,
        feature_names=names,
        min_support=min_support,
        support_validation=support_validation,
        **target_metadata,
        **rule_artifact(result, names),
    )
    return row, artifact


def main(task, argv=None):
    parser = argparse.ArgumentParser(
        description=f"TEDIP tabular {task} experiment.", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "dataset", help="builtin:iris/diabetes, CSV path, openml:ID, OpenML name, or uci:NAME."
    )
    parser.add_argument("--target", help="Target column for a local CSV.")
    parser.add_argument(
        "--test-size", type=float, default=0.25, help="Held-out fraction (default: 0.25)."
    )
    add_common_arguments(
        parser,
        default_leaves=4 if task == "classification" else 15,
        default_depth=2 if task == "classification" else 3,
    )
    if task == "regression":
        parser.add_argument("--folds", type=positive_int, default=5)
        parser.add_argument(
            "--support-grid",
            type=float,
            nargs="+",
            default=[i / 1000 for i in range(1, 11)],
            help="Training-CV support thresholds; --min-support selects a fixed value instead.",
        )
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if task == "regression":
        if args.folds < 2:
            parser.error("--folds must be at least 2")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in args.support_grid):
            parser.error("--support-grid values must lie in [0, 1]")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must lie strictly between 0 and 1")
    X, y = load_tabular(args.dataset, task, args.target)
    results = run_seeds(compute_run, args.jobs, [(task, X, y, seed, args) for seed in args.seeds])
    return save_results(results, args, task)
