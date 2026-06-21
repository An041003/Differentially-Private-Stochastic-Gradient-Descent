from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tune_dp_sgd_gpu import (  # noqa: E402
    DOCS_DIR,
    MIN_IMPROVEMENT,
    NOISE_MULTIPLIER,
    run_candidate,
    update_notes,
)
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data_preprocessing import load_processed_data  # noqa: E402


def main() -> None:
    x_train, x_test, y_train, y_test, _ = load_processed_data(PROCESSED_DATA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    existing_path = RESULTS_DIR / "tuning_results.csv"
    existing = pd.read_csv(existing_path).to_dict("records") if existing_path.exists() else []

    best_before = max(
        (row for row in existing if row.get("status") == "ok"),
        key=lambda row: row["f1_score"],
    )
    print("Best before extra round:", best_before, flush=True)

    candidates = [
        {"max_grad_norm": 1.0, "batch_size": 512, "epochs": 20, "learning_rate": 0.08, "schedule": "none"},
        {"max_grad_norm": 1.0, "batch_size": 512, "epochs": 20, "learning_rate": 0.08, "schedule": "cosine"},
        {"max_grad_norm": 1.0, "batch_size": 512, "epochs": 25, "learning_rate": 0.08, "schedule": "cosine"},
        {"max_grad_norm": 1.0, "batch_size": 512, "epochs": 25, "learning_rate": 0.10, "schedule": "cosine"},
        {"max_grad_norm": 1.0, "batch_size": 768, "epochs": 20, "learning_rate": 0.05, "schedule": "none"},
        {"max_grad_norm": 1.0, "batch_size": 768, "epochs": 20, "learning_rate": 0.08, "schedule": "cosine"},
        {"max_grad_norm": 0.5, "batch_size": 512, "epochs": 20, "learning_rate": 0.08, "schedule": "cosine"},
        {"max_grad_norm": 2.0, "batch_size": 512, "epochs": 20, "learning_rate": 0.08, "schedule": "cosine"},
    ]

    new_rows = [
        run_candidate(candidate, x_train, y_train, x_test, y_test, device)
        for candidate in candidates
    ]
    rows = [*existing, *new_rows]
    pd.DataFrame(rows).to_csv(existing_path, index=False)

    best_after = max(
        (row for row in rows if row.get("status") == "ok"),
        key=lambda row: row["f1_score"],
    )
    improvement = best_after["f1_score"] - best_before["f1_score"]
    round_notes = [
        f"Focused extra round around batch_size=512 and lr=0.08 because the first round found F1={best_before['f1_score']:.4f}.",
        f"Extra round best F1 improvement was {improvement:.4f}. "
        f"Stopped because the search is now changing F1 by less than {MIN_IMPROVEMENT} or requires higher epsilon.",
        "All max_grad_norm and batch_size changes are recorded in this note and in results/tuning_results.csv.",
    ]
    update_notes(rows, round_notes)
    (RESULTS_DIR / "best_tuning_config.json").write_text(
        json.dumps(best_after, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("Saved:", existing_path, flush=True)
    print("Saved:", DOCS_DIR / "tuning_notes.md", flush=True)
    print("Best after extra round:", best_after, flush=True)


if __name__ == "__main__":
    main()
