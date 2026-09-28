"""Small command-line and output helpers shared by the standalone scripts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_common_arguments(parser, default_leaves, default_depth=3):
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds (default: 0).",
    )
    parser.add_argument("--jobs", type=positive_int, default=1, help="Concurrent seeds.")
    parser.add_argument("--trees", type=positive_int, default=500, help="Number of forest trees.")
    parser.add_argument(
        "--depth", type=positive_int, default=default_depth, help="Maximum tree depth."
    )
    parser.add_argument(
        "--leaves", type=positive_int, default=default_leaves, help="Maximum selected rules."
    )
    parser.add_argument(
        "--lambda", dest="lambd", type=float, default=0.5, help="Stability weight in [0, 1]."
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=None,
        help="Minimum fraction of training rows per rule.",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Seconds per MIP solve.")
    parser.add_argument("--threads", type=positive_int, default=1, help="Gurobi threads per seed.")
    parser.add_argument(
        "--solver-log", action="store_true", help="Show the Gurobi optimization log."
    )
    parser.add_argument(
        "--output", type=Path, help="Output CSV (JSON metadata and rules saved alongside)."
    )


def validate_arguments(parser, args):
    if args.output is not None and args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")
    if not 0 <= args.lambd <= 1:
        parser.error("--lambda must be between 0 and 1")
    if args.min_support is not None and not 0 <= args.min_support <= 1:
        parser.error("--min-support must be between 0 and 1")
    if args.time_limit is not None and (not np.isfinite(args.time_limit) or args.time_limit <= 0):
        parser.error("--time-limit must be positive and finite")
    if any(seed < 0 or seed > 2**31 - 1 for seed in args.seeds):
        parser.error("seeds must be in [0, 2147483647]")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")


def solver_options(args, seed):
    return dict(
        time_limit=args.time_limit, threads=args.threads, seed=seed, verbose=args.solver_log
    )


def run_seeds(worker, jobs, requests):
    """Propagate failures; never turn infeasible runs into successful scores."""
    if jobs == 1:
        return [worker(request) for request in requests]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(worker, requests))


def rule_artifact(result, feature_names=None):
    rules = []
    for path, label, support in zip(result.paths, result.labels, result.weights):
        conditions, text = [], []
        for feature, threshold, direction in path:
            operator = "<=" if direction == "L" else ">"
            if result.metric is None:
                name = feature_names[feature] if feature_names else f"x[{feature}]"
                conditions.append(
                    dict(feature=int(feature), name=name, threshold=threshold, operator=operator)
                )
                text.append(f"{name} {operator} {threshold:.8g}")
            else:
                conditions.append(
                    dict(shapelet=list(feature), threshold=threshold, operator=operator)
                )
                text.append(
                    f"min_euclidean_distance(shapelet[{len(feature)}], series) {operator} {threshold:.8g}"
                )
        rules.append(
            dict(
                conditions=conditions,
                prediction=label,
                training_support=support,
                text=" AND ".join(text) if text else "TRUE",
            )
        )
    return dict(
        default_prediction=result.default,
        overlap_policy="largest training support; first on ties",
        fidelity_policy="same predictions as accuracy/MSE",
        rules=rules,
    )


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_results(results, args, task):
    """Write full precision CSV, run parameters/versions, and inspectable rules."""
    dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(args.dataset).name)
    output = args.output or Path("results") / task / f"{dataset_slug}.csv"
    if output.suffix.lower() != ".csv":
        raise ValueError("--output must end in .csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for row, artifact in results])
    frame.to_csv(output, sep=";", index=False)
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "gurobipy", "wildboar", "sktime"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    metadata = dict(
        implementation="TEDIP: Tree Ensembles Distillation through Integer Programming",
        task=task,
        created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies=versions,
        parameters=vars(args),
        score="MSE in standardized target units" if task == "regression" else "accuracy in [0, 1]",
    )
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".rules.json").write_text(
        json.dumps(
            [artifact for row, artifact in results],
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(f"Saved results to {output.resolve()}")
    return frame
