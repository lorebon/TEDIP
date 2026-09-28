"""Check research presets without downloading datasets or running large forests."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tabular_presets_select_paper_settings(task, monkeypatch):
    import tabular_experiment as runner

    monkeypatch.setattr(runner, "run_seeds", lambda worker, jobs, requests: [])
    monkeypatch.setattr(runner, "save_results", lambda results, args, task: args)
    dataset = "builtin:iris" if task == "classification" else "builtin:diabetes"
    args = runner.main(task, [dataset, "@" + str(ROOT / "settings" / (task + ".args"))])
    assert args.seeds == list(range(30))
    assert args.trees == 500 and args.lambd == 0.5
    assert args.test_size == 0.25
    assert args.depth == (2 if task == "classification" else 3)
    assert args.leaves == (4 if task == "classification" else 15)
    if task == "regression":
        assert args.min_support is None
        assert args.folds == 5
        assert args.support_grid == pytest.approx([i / 1000 for i in range(1, 11)])


@pytest.mark.temporal
def test_time_series_preset_keeps_dataset_shapelet_settings_explicit(folder_modules, monkeypatch):
    pytest.importorskip("wildboar")
    runner = folder_modules("time_series", "cli")
    monkeypatch.setattr(runner, "run_seeds", lambda worker, jobs, requests: [])
    monkeypatch.setattr(runner, "save_results", lambda results, args, task: args)
    args = runner.main(
        validate=True,
        argv=[
            "builtin:synthetic",
            "0.2",
            "0.7",
            "13",
            "@" + str(ROOT / "settings/time_series.args"),
        ],
    )
    assert args.seeds == list(range(10))
    assert args.trees == 500 and args.depth == 3 and args.lambd == 0.5
    assert args.folds == 5 and args.budgets is None
    assert (args.l, args.u, args.r) == (0.2, 0.7, "13")
