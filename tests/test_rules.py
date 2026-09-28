import numpy as np
import pytest
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from rule_utils import (
    compute_rule_statistics,
    predict_rules,
    represented_nodes,
    represented_paths,
    rule_mask,
    stability_scores,
    tabular_forest_paths,
    majority_label,
    normalize,
)


def test_stability_ignores_position_and_direction():
    a, b = (0, 0.7, "L"), (1, 12.2, "L")
    rules = [[a, b], [(1, 12.2, "R"), (0, 0.7, "R")], [(0, 0.7, "R")]]
    np.testing.assert_allclose(stability_scores(rules), [5 / 3, 5 / 3, 4 / 3])


def test_indexed_stability_matches_pairwise_set_reference():
    rng = np.random.default_rng(42)
    rules = [
        [(int(rng.integers(4)), 0.5, "L") for _ in range(int(rng.integers(5)))] for _ in range(50)
    ]
    sets = [{node[:2] for node in rule} for rule in rules]
    expected = [
        sum(
            2 * len(a & b) / (len(a) + len(b))
            for j, b in enumerate(sets)
            if i != j and len(a) + len(b)
        )
        for i, a in enumerate(sets)
    ]
    np.testing.assert_allclose(stability_scores(rules), expected, atol=1e-12)


@pytest.mark.parametrize("forest_type", [RandomForestClassifier, RandomForestRegressor])
def test_paths_partition_training_rows_and_match_forest_leaves(forest_type):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(50, 3))
    y = np.arange(50) % 3
    _, forest, trees, _ = tabular_forest_paths(
        forest_type(n_estimators=4, max_depth=3, random_state=0), X, y
    )
    for estimator, paths in zip(forest.estimators_, trees):
        masks = np.array([rule_mask(X, path) for path in paths])
        np.testing.assert_array_equal(masks.sum(axis=0), 1)
        actual_leaves = estimator.apply(X)
        recovered = [set(actual_leaves[mask]) for mask in masks]
        assert all(len(leaves) == 1 for leaves in recovered)
        assert len(set.union(*recovered)) == len(paths)


def test_tabular_root_only_paths_are_valid_rules():
    X, y = np.ones((8, 2)), np.zeros(8, dtype=int)
    paths, _, trees, nodes = tabular_forest_paths(
        RandomForestClassifier(n_estimators=2, random_state=0), X, y
    )
    assert paths == [[], []]
    statistics = compute_rule_statistics(X, y, paths, "classification")
    np.testing.assert_array_equal(statistics[4], [0, 0])
    assert represented_paths([[]], trees) == 1
    assert represented_nodes([[]], nodes) == 0


def test_score_overlap_tie_and_uncovered_rows():
    paths = [[(0, 2.0, "L")], [(0, 0.0, "R")]]
    X = np.array([[-1.0], [1.0], [3.0]])
    np.testing.assert_array_equal(predict_rules(X, paths, [10, 20], [2, 4], 99), [10, 20, 20])
    np.testing.assert_array_equal(predict_rules(X, paths, [10, 20], [4, 4], 99), [10, 10, 20])
    np.testing.assert_array_equal(predict_rules(X, [paths[0]], [10], [2], 99), [10, 10, 99])
    np.testing.assert_array_equal(
        predict_rules(X, [paths[0]], ["cat"], [2], "fallback"), ["cat", "cat", "fallback"]
    )


def test_empty_support_is_removed_and_all_filtered_rules_fail():
    X, y = np.array([[0.0], [1.0]]), np.array([1, 2])
    paths = [[(0, -1.0, "L")], [(0, 0.5, "L")], [(0, 0.5, "R")]]
    stats = compute_rule_statistics(X, y, paths, "classification")
    assert len(stats[3]) == 2
    with pytest.raises(ValueError, match="No candidate"):
        compute_rule_statistics(X, y, paths, "classification", min_support=1)


def test_rule_thresholds_match_sklearn_input_precision():
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier().fit(np.array([[0.1], [0.2]]), [0, 1])
    threshold = tree.tree_.threshold[0]
    points = np.array([[threshold], [np.nextafter(threshold, np.inf)], [0.2]])
    np.testing.assert_array_equal(
        rule_mask(points, [(0, threshold, "L")]), tree.predict(points) == 0
    )


def test_training_majority_ties_use_first_occurrence():
    assert majority_label([2, 1]) == 2


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_rule_statistics_loss_and_normalization(task):
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = np.array([1, 2, 2, 3, 3])
    paths = [[(0, 0.0, "L")], [(0, 0.0, "R")], [(0, 1.0, "L")]]
    loss, samples, labels, _, freq, weights = compute_rule_statistics(X, y, paths, task)
    reference_loss, reference_labels = [], []
    for sample in samples:
        targets = y[sample]
        if task == "classification":
            label, count = Counter(targets).most_common(1)[0]
            reference_loss.append(len(targets) - count)
        else:
            label = np.mean(targets)
            reference_loss.append(np.mean((targets - label) ** 2))
        reference_labels.append(label)
    expected = np.asarray(reference_loss)
    if max(expected) != min(expected):
        expected = (expected - min(expected)) / (max(expected) - min(expected))
    np.testing.assert_allclose(loss, expected)
    np.testing.assert_allclose(labels, reference_labels)
    assert weights == [3, 2, 4]
    np.testing.assert_allclose(freq, [1, 1, 0])


def test_regression_fallback_recovers_training_mean(folder_modules):
    warm = folder_modules("regression", "rule_extraction")
    paths = [[(0, -1.0, "L")], [(0, 1.0, "R")]]
    assert warm.computeScore(np.array([[0.0]]), [11], paths, [10, 20], [9, 1]) == 0
    assert warm.computeScore(np.array([[0.0]]), [15], paths, [10, 20], [9, 1], default=15) == 0


def test_normalization_handles_constant_and_nonconstant_values():
    np.testing.assert_allclose(normalize([2, 4, 6]), [0, 0.5, 1])
    np.testing.assert_allclose(normalize([2, 2]), [0, 0])


def test_representation_supports_shapelet_arrays_and_empty_forests():
    node = (np.array([1.0, 2.0]), 0.5, "L")
    copy = (np.array([1.0, 2.0]), 0.5, "L")
    assert represented_paths([[node]], [[[copy]]]) == 1
    assert represented_nodes([[node]], [[copy]]) == 1
    assert represented_paths([[node]], []) == 0
    assert represented_nodes([[node]], []) == 0


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fidelity_matches_scoring_with_overlap_and_uncovered_rows(task):
    from types import SimpleNamespace
    from distillation import DistillationResult

    X = np.array([[-2.0], [0.0], [2.0]])
    paths = [[(0, 1.0, "L")], [(0, -1.0, "R"), (0, 1.0, "L")]]
    forest = SimpleNamespace(predict=lambda X: np.array([10, 10, 15]))
    result = DistillationResult(
        paths=paths,
        labels=[10, 20],
        weights=[10, 5],
        forest=forest,
        default=15,
        task=task,
        represented_trees=0,
        represented_paths=0,
        solver_status=2,
        solver_gap=0,
        solver_runtime=0,
    )
    assert result.disagreement(X) == 0
    assert result.score(X, forest.predict(X)) == (1 if task == "classification" else 0)


def test_paper_equation_7_uses_raw_misclassification_count():
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])
    raw = compute_rule_statistics(X, y, [[], []], "classification", scale=False)
    np.testing.assert_array_equal(raw[0], [2, 2])


def test_stability_references_all_ensemble_rules_before_support_filtering():
    X = np.arange(4.0).reshape(-1, 1)
    paths = [[(0, 2.0, "L")], [(0, 2.0, "R")], [(0, 1.0, "L")]]
    stats = compute_rule_statistics(X, [0, 0, 1, 1], paths, "classification", min_support=0.5)
    assert stats[3] == [paths[0], paths[2]]
    np.testing.assert_array_equal(stats[4], [1, 0])


def test_equation_6_uses_absolute_ensemble_weights():
    a, b = (0, 0.7, "L"), (1, 12.2, "L")
    paths = [[a, b], [a, b], [a]]
    np.testing.assert_allclose(stability_scores(paths, [1, -2, 3]), [4, 3, 2])


def test_cached_support_filter_matches_direct_preparation():
    from distillation import filter_support, prepare_rules

    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0.0, 0.0, 1.0, 2.0])
    paths = [[(0, 2.0, "L")], [(0, 2.0, "R")], [(0, 1.0, "L")]]
    extracted = (paths, None, [], [])
    cached = filter_support(prepare_rules(X, y, extracted, "regression"), 0.5)
    direct = prepare_rules(X, y, extracted, "regression", Nmin=0.5)
    assert cached.paths == direct.paths
    assert cached.labels == direct.labels
    assert cached.weights == direct.weights
    for name in ("loss", "stability", "raw_loss", "raw_stability"):
        np.testing.assert_allclose(getattr(cached, name), getattr(direct, name))
    np.testing.assert_array_equal(cached.coverage.toarray(), direct.coverage.toarray())


def test_feature_f1_selects_top_five_percent_with_deterministic_ties():
    from types import SimpleNamespace
    from rule_utils import feature_f1

    forest = SimpleNamespace(feature_importances_=np.zeros(40))
    assert feature_f1(forest, [[(0, 0.5, "L"), (1, 0.5, "L")]]) == 1.0
    assert feature_f1(forest, [[(2, 0.5, "L")]]) == 0.0
