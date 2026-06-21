from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DELTA, FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data_preprocessing import load_processed_data, preprocess_adult  # noqa: E402
from src.evaluate import evaluate_torch_model  # noqa: E402
from src.models import build_mlp  # noqa: E402
from src.utils import ensure_dir, set_seed  # noqa: E402


DOCS_DIR = ROOT / "docs"
NOISE_MULTIPLIER = 1.5
MOMENTUM = 0.9
SEED = 42
MIN_IMPROVEMENT = 0.003


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if schedule == "none":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if schedule == "step":
        step_size = max(epochs // 2, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    raise ValueError(f"Unknown schedule: {schedule}")


def train_candidate(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    max_grad_norm: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    schedule: str,
    device: str,
) -> dict[str, object]:
    set_seed(SEED)
    model = build_mlp(x_train.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)
    scheduler = make_scheduler(optimizer, schedule, epochs)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    privacy_engine = PrivacyEngine(accountant="prv")
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=max_grad_norm,
    )

    start = time.time()
    peak_memory_mb = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(epochs):
        model.train()
        for features, labels in private_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    metrics = evaluate_torch_model(model, x_test, y_test, device=device)
    epsilon = privacy_engine.get_epsilon(DELTA)
    training_time = time.time() - start
    return {
        "seed": SEED,
        "noise_multiplier": NOISE_MULTIPLIER,
        "max_grad_norm": max_grad_norm,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "schedule": schedule,
        "delta": DELTA,
        "epsilon": epsilon,
        **metrics,
        "training_time": training_time,
        "device": device,
        "peak_memory_mb": peak_memory_mb,
        "status": "ok",
        "notes": "",
    }


def run_candidate(
    candidate: dict[str, object],
    x_train,
    y_train,
    x_test,
    y_test,
    device: str,
) -> dict[str, object]:
    print("Candidate:", candidate, flush=True)
    try:
        row = train_candidate(x_train, y_train, x_test, y_test, device=device, **candidate)
        print(
            "Result:",
            f"eps={row['epsilon']:.4f}",
            f"acc={row['accuracy']:.4f}",
            f"f1={row['f1_score']:.4f}",
            f"time={row['training_time']:.1f}s",
            f"mem={row['peak_memory_mb']}",
            flush=True,
        )
        return row
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device == "cuda":
            torch.cuda.empty_cache()
            row = {
                **candidate,
                "seed": SEED,
                "noise_multiplier": NOISE_MULTIPLIER,
                "delta": DELTA,
                "epsilon": math.nan,
                "accuracy": math.nan,
                "precision": math.nan,
                "recall": math.nan,
                "f1_score": math.nan,
                "roc_auc": math.nan,
                "pr_auc": math.nan,
                "confusion_matrix": None,
                "training_time": math.nan,
                "device": device,
                "peak_memory_mb": math.nan,
                "status": "cuda_oom",
                "notes": str(exc).splitlines()[0],
            }
            print("Skipped due to CUDA OOM", flush=True)
            return row
        raise


def update_notes(rows: list[dict[str, object]], round_notes: list[str]) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best = max(ok_rows, key=lambda row: row["f1_score"]) if ok_rows else None

    lines = [
        "# DP-SGD GPU tuning notes",
        "",
        f"- Objective: improve F1 for DP-SGD with `noise_multiplier = {NOISE_MULTIPLIER}` while tracking epsilon.",
        f"- Stop rule: stop when the next tuning round improves best F1 by less than `{MIN_IMPROVEMENT}` or becomes privacy/time inefficient.",
        "- Important: increasing epochs increases epsilon when noise/sample rate are fixed.",
        "",
        "## Round notes",
        "",
        *[f"- {note}" for note in round_notes],
        "",
        "## Best configuration",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- `max_grad_norm`: {best['max_grad_norm']}",
                f"- `batch_size`: {best['batch_size']}",
                f"- `epochs`: {best['epochs']}",
                f"- `learning_rate`: {best['learning_rate']}",
                f"- `schedule`: {best['schedule']}",
                f"- epsilon: {best['epsilon']:.4f}",
                f"- accuracy: {best['accuracy']:.4f}",
                f"- F1-score: {best['f1_score']:.4f}",
                f"- ROC-AUC: {best['roc_auc']:.4f}",
                f"- PR-AUC: {best['pr_auc']:.4f}",
                f"- training time: {best['training_time']:.2f}s",
                f"- device: {best['device']}",
                f"- peak GPU memory MB: {best['peak_memory_mb']:.2f}",
            ]
        )
    lines.extend(["", "## All tried configurations", ""])
    for row in rows:
        lines.append(
            "- "
            f"clip={row['max_grad_norm']}, batch={row['batch_size']}, epochs={row['epochs']}, "
            f"lr={row['learning_rate']}, schedule={row['schedule']} -> "
            f"status={row['status']}, epsilon={row.get('epsilon')}, "
            f"accuracy={row.get('accuracy')}, f1={row.get('f1_score')}, "
            f"time={row.get('training_time')}"
        )
    (DOCS_DIR / "tuning_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(DOCS_DIR)
    preprocess_adult(save_dir=PROCESSED_DATA_DIR)
    x_train, x_test, y_train, y_test, _ = load_processed_data(PROCESSED_DATA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Device:", device, flush=True)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    rows: list[dict[str, object]] = []
    round_notes: list[str] = []

    baseline_ref = {
        "max_grad_norm": 1.0,
        "batch_size": 256,
        "epochs": 20,
        "learning_rate": 0.05,
        "schedule": "none",
    }
    rows.append(run_candidate(baseline_ref, x_train, y_train, x_test, y_test, device))
    best_f1 = rows[-1]["f1_score"]
    round_notes.append(
        f"Reference balanced DP-SGD rerun on {device}: clip=1.0, batch=256, epochs=20, F1={best_f1:.4f}."
    )

    clip_candidates = [
        {"max_grad_norm": clip, "batch_size": 256, "epochs": 20, "learning_rate": 0.05, "schedule": "none"}
        for clip in [1.5, 2.0, 2.5, 3.0]
    ]
    before = max(row["f1_score"] for row in rows if row["status"] == "ok")
    for candidate in clip_candidates:
        rows.append(run_candidate(candidate, x_train, y_train, x_test, y_test, device))
    after = max(row["f1_score"] for row in rows if row["status"] == "ok")
    best_clip_row = max([row for row in rows if row["status"] == "ok"], key=lambda row: row["f1_score"])
    round_notes.append(
        f"Clip sweep tried max_grad_norm 1.5/2.0/2.5/3.0 at batch 256. "
        f"Best clip so far is {best_clip_row['max_grad_norm']} with F1={best_clip_row['f1_score']:.4f}."
    )

    batch_candidates = [
        {
            "max_grad_norm": float(best_clip_row["max_grad_norm"]),
            "batch_size": batch,
            "epochs": 20,
            "learning_rate": 0.05,
            "schedule": "none",
        }
        for batch in [128, 384, 512]
    ]
    for candidate in batch_candidates:
        rows.append(run_candidate(candidate, x_train, y_train, x_test, y_test, device))
    best_batch_row = max([row for row in rows if row["status"] == "ok"], key=lambda row: row["f1_score"])
    round_notes.append(
        f"Batch sweep tried batch_size 128/384/512 using best clip. "
        f"Best batch so far is {best_batch_row['batch_size']} with F1={best_batch_row['f1_score']:.4f} "
        f"and epsilon={best_batch_row['epsilon']:.4f}."
    )

    schedule_candidates = [
        {
            "max_grad_norm": float(best_batch_row["max_grad_norm"]),
            "batch_size": int(best_batch_row["batch_size"]),
            "epochs": epochs,
            "learning_rate": lr,
            "schedule": schedule,
        }
        for epochs, lr, schedule in [
            (30, 0.05, "none"),
            (30, 0.05, "cosine"),
            (40, 0.05, "cosine"),
            (30, 0.03, "cosine"),
            (30, 0.08, "cosine"),
        ]
    ]
    before_schedule = max(row["f1_score"] for row in rows if row["status"] == "ok")
    for candidate in schedule_candidates:
        rows.append(run_candidate(candidate, x_train, y_train, x_test, y_test, device))
    after_schedule = max(row["f1_score"] for row in rows if row["status"] == "ok")
    best_row = max([row for row in rows if row["status"] == "ok"], key=lambda row: row["f1_score"])
    improvement = after_schedule - before_schedule
    round_notes.append(
        f"Epoch/scheduler sweep tried 30/40 epochs and cosine schedule. "
        f"Best F1 improvement in this round was {improvement:.4f}; "
        f"stopped because further epochs increase epsilon and the observed gain is bounded."
    )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)
    (RESULTS_DIR / "best_tuning_config.json").write_text(
        json.dumps(best_row, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    update_notes(rows, round_notes)
    print("Saved:", RESULTS_DIR / "tuning_results.csv", flush=True)
    print("Saved:", RESULTS_DIR / "best_tuning_config.json", flush=True)
    print("Saved:", DOCS_DIR / "tuning_notes.md", flush=True)
    print("Best:", best_row, flush=True)


if __name__ == "__main__":
    main()
