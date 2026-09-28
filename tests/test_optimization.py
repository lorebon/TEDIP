import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from optimization import NoSolutionError, generateProblemSoft, selected_rules


@pytest.mark.solver
def test_partition_objective_matches_exhaustive_search(gurobi):
    A = np.array([[1, 0, 1], [0, 1, 1]])
    loss, stability = [0.1, 0.3, 0.9], [0.8, 0.6, 0.2]
    coefficients = 0.5 * (np.array(stability) - loss)
    feasible = [
        np.array(z)
        for z in itertools.product([0, 1], repeat=3)
        if np.all(A @ z == 1) and sum(z) <= 2
    ]
    model, z = generateProblemSoft(3, 2, A, 2, loss, stability)
    try:
        model.optimize()
        selected = selected_rules(model, z)
        np.testing.assert_array_equal(A[:, selected].sum(axis=1), 1)
        assert model.ObjVal == pytest.approx(max(coefficients @ z for z in feasible))
    finally:
        model.dispose()


@pytest.mark.solver
def test_infeasible_partition_raises_instead_of_returning_a_score(gurobi):
    model, z = generateProblemSoft(2, 2, np.eye(2), 1, [0, 0], [1, 1])
    try:
        model.optimize()
        with pytest.raises(NoSolutionError, match="no feasible solution"):
            selected_rules(model, z)
    finally:
        model.dispose()


def test_time_limit_without_incumbent_never_reads_solution():
    pytest.importorskip("gurobipy")
    model = SimpleNamespace(SolCount=0, Status=9)
    with pytest.raises(NoSolutionError):
        selected_rules(model, {})


def test_time_limit_with_incumbent_is_reported():
    pytest.importorskip("gurobipy")
    model = SimpleNamespace(SolCount=1, Status=9, MIPGap=0.25)
    with pytest.warns(RuntimeWarning, match="nonoptimal"):
        assert selected_rules(model, {0: SimpleNamespace(X=1), 1: SimpleNamespace(X=0)}) == [0]
    with pytest.raises(NoSolutionError, match="optimal solve"):
        selected_rules(model, {}, require_optimal=True)


@pytest.mark.solver
@pytest.mark.parametrize(
    "folder,task", [("classification", "classification"), ("regression", "regression")]
)
def test_fit_score_fidelity_and_training_partition(gurobi, folder_modules, folder, task):
    generate = folder_modules(folder, "experiment")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int) if task == "classification" else X[:, 0] ** 2
    result = generate.fit(X, y, n_estimators=3, depth=2, leaf_nodes=4)
    from rule_utils import rule_mask

    coverage = np.array([rule_mask(X, path) for path in result.paths]).sum(axis=0)
    np.testing.assert_array_equal(coverage, 1)
    assert sum(result.weights) == len(X)
    expected = (
        np.mean(result.predict(X) != result.forest.predict(X))
        if task == "classification"
        else np.mean((result.predict(X) - result.forest.predict(X)) ** 2)
    )
    assert result.disagreement(X) == pytest.approx(expected)
    if task == "regression":
        assert result.default == pytest.approx(np.mean(y))
    budget, score = generate.computeValidation(X, y, X, y, 2, 0, n_estimators=3)
    assert budget >= 1 and np.isfinite(score)


@pytest.mark.solver
def test_temporal_optimization_honors_lambda_and_allows_one_rule(gurobi):
    from scipy import sparse
    from distillation import PreparedRules, validate_rules

    def distance(x, y):
        return float(np.linalg.norm(x - y))

    paths = [[((0.0,), 0.5, "L")], [((0.0,), 0.5, "R")], []]
    prepared = PreparedRules(
        forest=SimpleNamespace(predict=lambda X: np.zeros(len(X))),
        paths=paths,
        labels=[0, 1, 0],
        weights=[1, 1, 2],
        loss=np.array([0.0, 0.0, 1.0]),
        stability=np.array([0.2, 0.2, 0.8]),
        coverage=sparse.csr_matrix([[1, 0, 1], [0, 1, 1]]),
        trees_pathed=[paths],
        trees_noded=[paths[0] + paths[1]],
        default=0,
        task="classification",
        metric=distance,
    )
    X, y = np.array([[0.0], [1.0]]), np.array([0, 1])
    assert validate_rules(prepared, X, y, lambd=1) == (1, 0.5)
    assert validate_rules(prepared, X, y, lambd=0) == (2, 1.0)


def test_cv_exact_range_includes_odd_budgets_and_full_training_feasibility(monkeypatch):
    import distillation

    prepared = [SimpleNamespace(bounds=b, task="classification") for b in [(1, 2), (2, 4), (3, 5)]]
    seen = []
    monkeypatch.setattr(distillation, "rule_count_bounds", lambda p, **kw: p.bounds)

    def scores(p, X, y, budgets, *args, **kwargs):
        seen.append(list(budgets))
        return {3: 0.9, 4: 0.8, 5: 0.8}

    monkeypatch.setattr(distillation, "budget_scores", scores)
    budget, details, bounds = distillation.cross_validate_budgets(
        [(p, None, None) for p in prepared[:2]], prepared[2]
    )
    assert budget == 3
    assert seen == [[3, 4, 5], [3, 4, 5]]
    assert bounds == [(1, 2), (2, 4), (3, 5)]
    assert details["3"]["mean"] == 0.9


@pytest.mark.solver
def test_exact_bounds_match_exhaustive_partitions(gurobi):
    from scipy import sparse
    from distillation import rule_count_bounds

    A = np.column_stack([np.eye(3), np.ones(3)])
    prepared = SimpleNamespace(coverage=sparse.csr_matrix(A))
    sizes = [sum(z) for z in itertools.product([0, 1], repeat=4) if np.all(A @ z == 1)]
    assert rule_count_bounds(prepared) == (min(sizes), max(sizes)) == (1, 3)


@pytest.mark.solver
def test_regression_support_cv_reuses_training_forests(gurobi, monkeypatch):
    import pandas as pd
    import tabular_experiment as experiment

    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(24, 2)))
    y = X[0].to_numpy() ** 2
    args = SimpleNamespace(
        support_grid=[0.001, 0.01],
        folds=2,
        trees=2,
        depth=2,
        leaves=4,
        lambd=0.5,
        time_limit=None,
        threads=1,
        solver_log=False,
    )
    prepare = experiment.prepare_split
    calls = []

    def checked(task, train, y_train, val, y_val, seed, args):
        assert set(train.index).isdisjoint(val.index)
        calls.append(len(train))
        return prepare(task, train, y_train, val, y_val, seed, args)

    monkeypatch.setattr(experiment, "prepare_split", checked)
    selected, details = experiment.select_min_support(X, y, 0, args)
    assert calls == [12, 12]
    assert selected == 0.001
    assert all(len(value["fold_scores"]) == 2 for value in details.values())
