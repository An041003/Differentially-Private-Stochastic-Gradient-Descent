# DP-SGD: Privacy-Preserving Machine Learning on UCI Adult

This repository implements and evaluates Differentially Private SGD (DP-SGD)
with PyTorch and Opacus on the UCI Adult/Census Income dataset. It compares
non-private baselines with DP-SGD, measures privacy--utility trade-offs, and
runs a confidence-based membership-inference check.

## Quick start on Windows

1. Extract `DPSGD_project_package_20260621.zip` to a writable folder.
2. Install Python 3.12 and make sure the `py` launcher is available.
3. Double-click `run_project.bat`.

The first run creates `.venv`, installs dependencies from
`requirements.txt`, then regenerates the reproducible core results. Internet
access is needed only for the initial package installation.

The script writes or refreshes:

- `data/processed/`: processed Adult arrays and metadata;
- `results/`: baseline, noise, clipping, batch-size, multi-seed, MIA, and
  final-summary CSV files;
- `figures/`: plots generated from the new CSV files;
- `report/report.md` and `report/report.docx`;
- `slides/dp_sgd_final_presentation.pptx`.

It uses CUDA automatically when the installed PyTorch build can access a
compatible GPU; otherwise it runs on CPU. The core experiment includes several
DP-SGD sweeps, so CPU execution can take considerably longer than GPU.

## Manual execution

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_final_experiments.py
.\.venv\Scripts\python.exe scripts\generate_final_deliverables.py
```

Optional advanced experiments are available in `scripts/`, including stronger
non-DP baselines, the noise--clipping stress test, model--clip--noise grid,
and GPU tuning. Their saved results and figures are already included for
inspection; they are not rerun by `run_project.bat` because they are much more
expensive.

## Package contents

| Path | Purpose |
| --- | --- |
| `data/` | Raw UCI Adult data and reproducibly generated processed arrays |
| `src/` | Reusable preprocessing, models, training, evaluation, plotting, and MIA code |
| `scripts/` | Core runner, advanced experiment runners, deliverable generators, and release builder |
| `notebooks/` | Exploration/demo notebooks; outputs are removed in the release ZIP |
| `results/`, `figures/` | Saved numerical evidence and publication-ready charts |
| `docs/`, `*.md` | Experiment notes, project guide, architecture, and documentation |
| `report/`, `slides/` | Final report artifacts and presentation |
| `run_project.bat` | One-click reproducible core pipeline |

`adult/` is an older duplicate dataset folder and is deliberately excluded
from the package. The active inputs are under `data/raw/`.

## Build the clean release ZIP again

The packaging utility copies only the required project materials and removes
notebook cell outputs in its temporary staging area; it never changes the
working notebooks.

```powershell
.\.venv\Scripts\python.exe scripts\build_release_package.py
```

The output is `dist/DPSGD_project_package_20260621.zip`. A
`PACKAGE_MANIFEST.json` inside the archive records exactly what was included.

## Main results

- Strong clean reference: engineered-feature HistGradientBoosting reached
  threshold-tuned F1 `0.7273` on the original Adult test split.
- Best DP grid result: F1 `0.6760` at noise `0.5`, clipping norm `1.0`.
- Balanced DP-SGD setting: noise `1.5`, epsilon `1.2143`, accuracy `0.8497`,
  F1 `0.6571` in the standard sweep.
- The confidence-based membership-inference AUCs are near `0.5`; this simple
  attack is close to random guessing in the evaluated setup.

For the complete methodology, full-grid interpretation, limitations, and
references, see `report/bao_cao_dp_sgd_chi_tiet_15_trang.docx` and its Markdown
source beside it.
