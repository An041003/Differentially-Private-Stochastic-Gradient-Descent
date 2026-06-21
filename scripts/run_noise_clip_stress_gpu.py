from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_SEED,
    DELTA,
    EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE_DP,
    MOMENTUM,
    NOISE_CLIP_GRID_MAX_GRAD_NORMS,
    NOISE_CLIP_GRID_NOISE_MULTIPLIERS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.data_preprocessing import load_processed_data, preprocess_adult  # noqa: E402
from src.train_dp_sgd import train_dp_mlp  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

DOCS_DIR = ROOT / "docs"
OUTPUT_CSV = RESULTS_DIR / "noise_clip_grid_results.csv"
SUMMARY_JSON = RESULTS_DIR / "noise_clip_grid_summary.json"
NOTES_MD = DOCS_DIR / "noise_clip_stress_notes.md"


def _load_baseline_f1() -> float | None:
    tuned_path = RESULTS_DIR / "best_tuned_baseline.json"
    if tuned_path.exists():
        best = json.loads(tuned_path.read_text(encoding="utf-8"))
        return float(best["f1_score"])
    path = RESULTS_DIR / "baseline_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["model_name"] == "mlp_baseline"]
    if row.empty:
        return None
    return float(row.iloc[0]["f1_score"])


def _save_progress(rows: list[dict[str, object]]) -> pd.DataFrame:
    ensure_dir(RESULTS_DIR)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    return df


def _plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="max_grad_norm", columns="noise_multiplier", values="f1_score")
    output_path = FIGURES_DIR / "noise_clip_f1_heatmap.png"
    ensure_dir(output_path.parent)
    plt.figure(figsize=(9, 4.8))
    image = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(image, label="F1-score")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("Noise multiplier")
    plt.ylabel("Max grad norm")
    plt.title("F1-score Across Noise x Clipping Grid")
    for row_idx, clip in enumerate(pivot.index):
        for col_idx, noise in enumerate(pivot.columns):
            value = pivot.loc[clip, noise]
            plt.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_lines(df: pd.DataFrame, baseline_f1: float | None) -> None:
    output_path = FIGURES_DIR / "noise_clip_f1_lines.png"
    ensure_dir(output_path.parent)
    plt.figure(figsize=(9, 4.8))
    for clip, clip_df in df.groupby("max_grad_norm"):
        clip_df = clip_df.sort_values("noise_multiplier")
        plt.plot(clip_df["noise_multiplier"], clip_df["f1_score"], marker="o", label=f"clip={clip:g}")
    if baseline_f1 is not None:
        plt.axhline(baseline_f1, color="gray", linestyle="--", label="MLP baseline F1")
    plt.xlabel("Noise multiplier")
    plt.ylabel("F1-score")
    plt.title("Extreme Noise Stress Test by Clipping Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_best_by_noise(df: pd.DataFrame, baseline_f1: float | None) -> None:
    output_path = FIGURES_DIR / "extreme_noise_best_f1.png"
    best_by_noise = df.sort_values("f1_score", ascending=False).groupby("noise_multiplier", as_index=False).first()
    best_by_noise = best_by_noise.sort_values("noise_multiplier")
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.plot(best_by_noise["noise_multiplier"], best_by_noise["f1_score"], marker="o", label="Best clip per noise")
    if baseline_f1 is not None:
        plt.axhline(baseline_f1, color="gray", linestyle="--", label="MLP baseline F1")
    plt.xlabel("Noise multiplier")
    plt.ylabel("Best F1-score")
    plt.title("Best F1-score Under Increasing DP Noise")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_summary(df: pd.DataFrame, baseline_f1: float | None, device: str) -> None:
    ok = df[df["status"] == "ok"].copy()
    best = ok.sort_values("f1_score", ascending=False).iloc[0]
    worst = ok.sort_values("f1_score", ascending=True).iloc[0]
    best_by_noise = ok.sort_values("f1_score", ascending=False).groupby("noise_multiplier", as_index=False).first()
    best_by_noise = best_by_noise.sort_values("noise_multiplier")

    collapse_rows = pd.DataFrame()
    if baseline_f1 is not None:
        collapse_rows = best_by_noise[best_by_noise["f1_score"] <= baseline_f1 - 0.05]
    collapse_noise = None if collapse_rows.empty else float(collapse_rows.iloc[0]["noise_multiplier"])

    summary = {
        "device": device,
        "baseline_f1": baseline_f1,
        "num_ok_runs": int(len(ok)),
        "best_config": best.to_dict(),
        "worst_config": worst.to_dict(),
        "first_noise_with_best_f1_drop_at_least_0p05": collapse_noise,
        "noise_values": NOISE_CLIP_GRID_NOISE_MULTIPLIERS,
        "clip_values": NOISE_CLIP_GRID_MAX_GRAD_NORMS,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Noise x clipping stress-test notes",
        "",
        f"- Device: `{device}`.",
        f"- Grid: noise multiplier `{NOISE_CLIP_GRID_NOISE_MULTIPLIERS}` x max grad norm `{NOISE_CLIP_GRID_MAX_GRAD_NORMS}`.",
        f"- Main metric: F1-score. Accuracy is secondary because UCI Adult is imbalanced.",
        (
            f"- Reference non-DP MLP F1-score: `{baseline_f1:.4f}`. "
            "This prefers `results/best_tuned_baseline.json` when available."
        )
        if baseline_f1 is not None
        else "- Reference non-DP MLP F1-score: not available.",
        "",
        "## Best and worst observed configurations",
        "",
        (
            f"- Best F1: noise `{best['noise_multiplier']}`, clip `{best['max_grad_norm']}`, "
            f"epsilon `{best['epsilon']:.4f}`, F1 `{best['f1_score']:.4f}`."
        ),
        (
            f"- Worst F1: noise `{worst['noise_multiplier']}`, clip `{worst['max_grad_norm']}`, "
            f"epsilon `{worst['epsilon']:.4f}`, F1 `{worst['f1_score']:.4f}`."
        ),
        "",
        "## Interpretation",
        "",
        "- The earlier noise range up to 3.0 tested the practical privacy-utility region.",
        "- This stress test adds much larger noise values to check whether utility collapse appears only under extreme privacy settings.",
    ]
    if baseline_f1 is not None and collapse_noise is not None:
        lines.append(
            f"- Best-per-noise F1 first drops by at least 0.05 below the tuned non-DP baseline at noise `{collapse_noise}`."
        )
    elif baseline_f1 is not None:
        lines.append("- In this grid, best-per-noise F1 never drops by at least 0.05 below the tuned non-DP baseline.")
    lines.extend(
        [
            "- Use `figures/noise_clip_f1_heatmap.png`, `figures/noise_clip_f1_lines.png`, and `figures/extreme_noise_best_f1.png` for the final discussion.",
            "",
            "## Best F1 per noise",
            "",
            "| noise_multiplier | best_clip | epsilon | F1-score | precision | recall |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in best_by_noise.iterrows():
        lines.append(
            f"| {row['noise_multiplier']:.1f} | {row['max_grad_norm']:.1f} | "
            f"{row['epsilon']:.4f} | {row['f1_score']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )
    NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(DOCS_DIR)
    preprocess_adult(save_dir=PROCESSED_DATA_DIR)
    x_train, x_test, y_train, y_test, feature_names = load_processed_data(PROCESSED_DATA_DIR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This stress test is intended to run on the local T1200 GPU.")
    device = "cuda"
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    baseline_f1 = _load_baseline_f1()
    existing_rows: list[dict[str, object]] = []
    if OUTPUT_CSV.exists():
        existing_rows = pd.read_csv(OUTPUT_CSV).to_dict("records")
    completed = {
        (float(row["noise_multiplier"]), float(row["max_grad_norm"]))
        for row in existing_rows
        if row.get("status") == "ok"
    }
    rows = existing_rows

    for noise in NOISE_CLIP_GRID_NOISE_MULTIPLIERS:
        for clip in NOISE_CLIP_GRID_MAX_GRAD_NORMS:
            key = (float(noise), float(clip))
            if key in completed:
                print(f"Skipping completed noise={noise} clip={clip}", flush=True)
                continue
            print(f"Running stress grid noise={noise} clip={clip}", flush=True)
            try:
                _, result = train_dp_mlp(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    input_dim=len(feature_names),
                    seed=DEFAULT_SEED,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE_DP,
                    momentum=MOMENTUM,
                    noise_multiplier=noise,
                    max_grad_norm=clip,
                    delta=DELTA,
                    device=device,
                )
                row = {
                    **result,
                    "status": "ok",
                    "device": device,
                    "baseline_f1": baseline_f1,
                    "f1_drop_vs_mlp_baseline": None if baseline_f1 is None else baseline_f1 - float(result["f1_score"]),
                }
                print(
                    f"Result noise={noise} clip={clip} eps={row['epsilon']:.4f} "
                    f"f1={row['f1_score']:.4f} precision={row['precision']:.4f} recall={row['recall']:.4f}",
                    flush=True,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                row = {
                    "experiment_id": f"failed_noise_{noise}_clip_{clip}",
                    "seed": DEFAULT_SEED,
                    "model_name": "dp_sgd_mlp",
                    "is_dp": True,
                    "dataset": "uci_adult",
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                    "input_dim": len(feature_names),
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE_DP,
                    "optimizer": "sgd",
                    "momentum": MOMENTUM,
                    "noise_multiplier": noise,
                    "max_grad_norm": clip,
                    "delta": DELTA,
                    "epsilon": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "roc_auc": None,
                    "pr_auc": None,
                    "training_time": None,
                    "status": "failed",
                    "device": device,
                    "baseline_f1": baseline_f1,
                    "f1_drop_vs_mlp_baseline": None,
                    "notes": str(exc).splitlines()[0],
                }
                print("Failed:", row["notes"], flush=True)
            rows.append(row)
            _save_progress(rows)

    if baseline_f1 is not None:
        for row in rows:
            if row.get("status") == "ok" and row.get("f1_score") is not None:
                row["reference_baseline_f1"] = baseline_f1
                row["f1_drop_vs_reference_baseline"] = baseline_f1 - float(row["f1_score"])

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    ok = df[df["status"] == "ok"].copy()
    _plot_heatmap(ok)
    _plot_lines(ok, baseline_f1)
    _plot_best_by_noise(ok, baseline_f1)
    _write_summary(ok, baseline_f1, device)
    print("Saved:", OUTPUT_CSV, flush=True)
    print("Saved:", SUMMARY_JSON, flush=True)
    print("Saved:", NOTES_MD, flush=True)


if __name__ == "__main__":
    main()
