"""Run the regression experiment; see --help and the repository README."""

from pathlib import Path
import sys

# Resolve shared helpers relative to this file, independent of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabular_experiment import main


if __name__ == "__main__":
    main("regression")
