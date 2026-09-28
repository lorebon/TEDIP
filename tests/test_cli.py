"""Exercise the documented entry points from outside the repository directory."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.solver
@pytest.mark.parametrize(
    "script,dataset,extra",
    [
        ("classification/run.py", "builtin:iris", ["--seeds", "0", "1", "--jobs", "2"]),
        (
            "regression/run.py",
            "builtin:diabetes",
            ["--seeds", "0", "--folds", "2", "--support-grid", "0.001", "0.01"],
        ),
        (
            "time_series/run.py",
            "builtin:synthetic",
            ["0.1", "0.5", "3", "--seeds", "0"],
        ),
        (
            "time_series/validate.py",
            "builtin:synthetic",
            ["0.1", "0.5", "3", "--folds", "2", "--seeds", "0"],
        ),
    ],
)
def test_standalone_scripts_export_results(gurobi, tmp_path, script, dataset, extra):
    if script.startswith("time_series"):
        pytest.importorskip("wildboar")
    output = tmp_path / "result.csv"
    settings = []
    if script != "time_series/run.py":
        settings = ["@" + str(ROOT / "settings" / (Path(script).parent.name + ".args"))]
    if script.startswith("time_series"):
        arguments = [dataset, *extra[:3], *settings, *extra[3:]]
    else:
        arguments = [dataset, *settings, *extra]
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / script),
            *arguments,
            "--trees",
            "3",
            "--depth",
            "2",
            "--leaves",
            "4",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    frame = pd.read_csv(output, sep=";")
    artifacts = json.loads(output.with_suffix(".rules.json").read_text())
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert len(frame) == len(artifacts) == len(metadata["parameters"]["seeds"])
    assert (frame["Solver status"] == 2).all()
    assert (frame["MIP gap"] == 0).all()
    assert all(
        len(artifact["rules"]) == count for artifact, count in zip(artifacts, frame["Leaves"])
    )
    assert all(rule["training_support"] > 0 for artifact in artifacts for rule in artifact["rules"])


def test_invalid_parameters_fail_before_loading_data(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "classification/run.py"),
            "nonexistent-network-dataset",
            "--lambda",
            "2",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert run.returncode == 2
    assert "--lambda must be between 0 and 1" in run.stderr
