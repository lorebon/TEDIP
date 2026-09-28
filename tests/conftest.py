"""Isolate same-named modules while testing the three script folders."""

import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def folder_modules(monkeypatch):
    def load(folder, module):
        for name in (
            "rule_extraction",
            "experiment",
            "metrics",
            "preprocessing",
            "cli",
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.syspath_prepend(str(ROOT / folder))
        return importlib.import_module(module)

    return load


@pytest.fixture
def gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", 0)
            environment.start()
            with gp.Model(env=environment):
                pass
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")
    return gp
