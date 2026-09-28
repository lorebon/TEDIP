import numpy as np
import pytest
from types import SimpleNamespace


@pytest.mark.temporal
def test_shapelet_paths_match_wildboar_tree_partition(folder_modules):
    pytest.importorskip("wildboar")
    warm = folder_modules("time_series", "rule_extraction")
    from metrics import euclidean

    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 16))
    y = np.arange(25) % 2
    paths, forest, trees, nodes = warm.computePaths(X, y, 3, 0.1, 0.8, 5, 2, 0)
    assert len(trees) == len(nodes) == 3
    assert max(map(len, paths)) <= 2
    for estimator, tree_paths in zip(forest.estimators_, trees):
        masks = np.array(
            [[warm.computeSample(row, path, euclidean) for row in X] for path in tree_paths]
        )
        np.testing.assert_array_equal(masks.sum(axis=0), 1)
        leaves = estimator.apply(X)
        assert all(len(set(leaves[mask])) == 1 for mask in masks)


@pytest.mark.temporal
@pytest.mark.solver
def test_temporal_fit_and_validation(gurobi, folder_modules):
    pytest.importorskip("wildboar")
    generate = folder_modules("time_series", "experiment")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 12))
    y = np.arange(20) % 2
    result = generate.fit(X, y, depth=1, n_estimators=2, n_shap=3, leaf_nodes=2)
    assert len(result.paths) <= 2
    assert 0 <= result.score(X, y) <= 1
    budget, score = generate.computeValidation(X, y, X, y, 1, 0, 0, 1, 3, n_estimators=2)
    assert budget >= 1 and 0 <= score <= 1


@pytest.mark.temporal
def test_cross_validation_selects_budget_by_mean_accuracy(folder_modules, monkeypatch):
    pytest.importorskip("wildboar")
    experiment = folder_modules("time_series", "cli")
    import distillation

    prepared_sizes, final_budgets = [], []

    def prepare(X, *args):
        prepared_sizes.append(len(X))
        return SimpleNamespace(fold=len(prepared_sizes) - 1, task="classification")

    def scores(prepared, X, y, budgets, *args, **kwargs):
        assert list(budgets) == [2, 4]
        return {2: [1.0, 0.0][prepared.fold - 1], 4: 0.75}

    def solve(prepared, budget, *args, **kwargs):
        assert prepared.fold == 0
        final_budgets.append(budget)
        return SimpleNamespace(
            represented_trees=0.5,
            represented_paths=0.5,
            paths=[[]],
            score=lambda X, y: 0.8,
            forest_score=lambda X, y: 0.8,
            disagreement=lambda X: 0.0,
            solver_status=2,
            solver_gap=0.0,
            solver_runtime=0.0,
        )

    monkeypatch.setattr(experiment, "prepare", prepare)
    monkeypatch.setattr(distillation, "budget_scores", scores)
    monkeypatch.setattr(experiment, "solve_rules", solve)
    monkeypatch.setattr(experiment, "rule_artifact", lambda result: {})
    args = SimpleNamespace(
        r="3",
        leaves=6,
        folds=2,
        budgets=[2, 4],
        depth=2,
        l=0.1,
        u=0.5,
        min_support=None,
        trees=2,
        lambd=0.5,
        time_limit=None,
        threads=1,
        solver_log=False,
    )
    X, y = np.ones((8, 6)), np.arange(8) % 2
    row, artifact = experiment.compute_run((X, y, X, y, [0, 1], 0, args, True))
    assert prepared_sizes == [8, 4, 4]
    assert final_budgets == [4]
    assert row["Rule budget"] == 4
    assert artifact["validation"]["2"]["mean"] == 0.5
    assert artifact["validation"]["4"]["mean"] == 0.75
