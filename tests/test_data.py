import numpy as np
import pandas as pd
import pytest

from data_utils import clean_tabular, encode_features, load_tabular


def test_missing_rows_align_positionally_with_nondefault_index():
    X = pd.DataFrame({"a": [1, np.nan, 3, 4]}, index=[50, 10, 90, 70])
    with pytest.warns(UserWarning, match="2 rows"):
        clean_X, y = clean_tabular(X, [5, 6, np.nan, 8], "regression")
    assert clean_X["a"].tolist() == [1, 4]
    assert y.tolist() == [5, 8]


def test_encoding_is_fitted_only_on_training_categories():
    train = pd.DataFrame({"number": [1, 2], "category": ["a", "b"]})
    test = pd.DataFrame({"number": [3], "category": ["unseen"]})
    X_train, X_test, names, encoder = encode_features(train, test)
    assert names == ["number", "category_a", "category_b"]
    np.testing.assert_array_equal(X_test, [[3, 0, 0]])
    assert X_train.shape == (2, 3)


@pytest.mark.parametrize("categorical", [False, True])
def test_all_numeric_or_all_categorical_inputs(categorical):
    data = pd.DataFrame({"x": ["a", "b"] if categorical else [1, 2]})
    train, test, names, _ = encode_features(data, data)
    assert train.shape == test.shape
    assert len(names) == train.shape[1]


def test_csv_target_and_missing_file(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2], "target": ["a", "b"]}).to_csv(path, index=False)
    X, y = load_tabular(str(path), "classification", "target")
    assert X.columns.tolist() == ["x"]
    assert y.tolist() == ["a", "b"]
    with pytest.raises(FileNotFoundError):
        load_tabular(str(tmp_path / "absent.csv"), "regression")


def test_temporal_encoding_does_not_overwrite_existing_labels(folder_modules):
    preprocessing = folder_modules("time_series", "preprocessing")
    original = np.array([1, 0, 1, 0])
    _, y, classes, _, _ = preprocessing.preprocessTrain(np.ones((4, 5)), original)
    np.testing.assert_array_equal(original, [1, 0, 1, 0])
    np.testing.assert_array_equal(y, [1, 0, 1, 0])
    _, test = preprocessing.preprocessTest(np.ones((2, 5)), [0, 1], classes)
    np.testing.assert_array_equal(test, [0, 1])
    with pytest.raises(ValueError, match="did not occur"):
        preprocessing.preprocessTest(np.ones((1, 5)), [7], classes)


def test_temporal_string_labels_and_nested_series(folder_modules):
    preprocessing = folder_modules("time_series", "preprocessing")
    X = pd.DataFrame({"series": [pd.Series([1, 2]), pd.Series([3, 4])]})
    X, y, classes, _, _ = preprocessing.preprocessTrain(X, ["red", "blue"])
    assert X.shape == (2, 2)
    np.testing.assert_array_equal(y, [1, 0])
    assert classes == ("blue", "red")
    with pytest.raises(ValueError, match="univariate"):
        preprocessing.series_array(np.ones((2, 2, 3)))
