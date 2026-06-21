from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

DEFAULT_SEED = 42
DELTA = 1e-5
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE_BASELINE = 1e-3
LEARNING_RATE_DP = 0.05
MOMENTUM = 0.9
MAX_GRAD_NORM = 1.0

NOISE_MULTIPLIERS = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
MAX_GRAD_NORM_VALUES = [0.5, 1.0, 1.5, 2.0]
BATCH_SIZE_VALUES = [64, 128, 256, 512]
SEEDS = [42, 123, 2026]

# Joint stress-test grid used to answer whether utility collapses only at
# larger noise levels and how that interacts with clipping.
NOISE_CLIP_GRID_NOISE_MULTIPLIERS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
NOISE_CLIP_GRID_MAX_GRAD_NORMS = [0.5, 1.0, 2.0, 3.0]

DATASET_NAME = "uci_adult"
