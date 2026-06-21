# Differentially Private Stochastic Gradient Descent

Source code for training and evaluating DP-SGD models on the UCI Adult/Census
Income dataset using PyTorch and Opacus.

## Run on Windows

Double-click `run_project.bat`. On its first run it creates `.venv`, installs
the packages in `requirements.txt`, processes the data, and runs the core
experiment pipeline.

Or run manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_final_experiments.py
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/` | UCI Adult source data and processed arrays |
| `src/` | Preprocessing, model, training, evaluation, plotting, and privacy-attack modules |
| `scripts/` | Core and advanced experiment runners plus utility scripts |
| `notebooks/` | Clean exploratory/demo notebooks without cell outputs |
| `run_project.bat` | One-click core experiment runner |

The repository intentionally excludes generated results, figures, reports,
slides, and project-internal Markdown documentation.
