# TEDIP

Research code for **A unified approach to extract interpretable rules from tree
ensembles via Integer Programming**, by Lorenzo Bonasera and Emilio Carrizosa,
*Computers & Operations Research* 185 (2026), 107283.
[Read the paper](https://doi.org/10.1016/j.cor.2025.107283).

TEDIP extracts a compact set of interpretable rules from random forests using
integer programming. It supports tabular classification, regression, and
time-series classification through directly runnable Python scripts.

## Setup

Use Python 3.10–3.12. From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

For time-series experiments, also install:

```bash
python -m pip install -r requirements-temporal.txt
```

Rule selection requires a working **Gurobi license** appropriate for the model
size. See the [Gurobi setup guide](https://docs.gurobi.com/projects/optimizer/en/current/).
The tested environment is recorded in `requirements-tested.txt`.

## Quick start

Run these small examples from the repository root. They use bundled data and do
not require dataset downloads.

```bash
# Classification
python classification/run.py builtin:iris --trees 5 --depth 2 --leaves 4

# Regression
python regression/run.py builtin:diabetes --trees 5 --depth 2 --leaves 4 --min-support 0

# Time-series classification
python time_series/run.py builtin:synthetic 0.1 0.5 5 --trees 3 --depth 2 --leaves 4
```

For time-series classification with rule-budget validation, use
`time_series/validate.py`. Every entry point supports `--help` for options and
data formats. Tabular scripts accept local CSV files with `--target COLUMN`,
and time-series scripts accept UCR/UEA dataset names or local `.npz` splits.

Results are saved under `results/`: a CSV contains scores, `.metadata.json`
records settings, and `.rules.json` contains the selected rules. Use
`--output results/my-run.csv` to choose the output filename.
These small runs check the workflow; they do not reproduce the paper's results.

## Repository guide

| Path | Contents |
| --- | --- |
| [classification/](classification/) | Tabular classification scripts |
| [regression/](regression/) | Tabular regression scripts |
| [time_series/](time_series/) | Shapelet classification and rule-budget validation |
| [distillation.py](distillation.py), [optimization.py](optimization.py), [rule_utils.py](rule_utils.py) | Shared rule extraction, selection, and prediction |
| [settings/](settings/) | Reusable arguments, loaded with `@settings/FILE.args` |

## Citation and license

```bibtex
@article{bonasera2026unified,
  title   = {A unified approach to extract interpretable rules from tree ensembles via Integer Programming},
  author  = {Bonasera, Lorenzo and Carrizosa, Emilio},
  journal = {Computers & Operations Research},
  volume  = {185},
  pages   = {107283},
  year    = {2026},
  doi     = {10.1016/j.cor.2025.107283}
}
```

Code is distributed under the [MIT license](LICENSE). Dataset terms are set by
their respective providers.
