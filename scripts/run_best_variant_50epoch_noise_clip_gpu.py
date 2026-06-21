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
OUTPUT_CSV = RESULTS_DIR / "best_variant_50epoch_noise_clip_results.csv"
SUMMARY_JSON = RESULTS_DIR / "best_variant_50epoch_noise_clip_summary.json"
NOTES_MD = DOCS_DIR / "best_variant_50epoch_noise_clip_notes.md"

MODEL_VARIANT = {
    "name": "mlp_64_32_d0p1",
    "hidden_dims": (64, 32),
    "dropout": 0.1,
}
EPOCHS_50 = 50


def load_reference_f1() -> float | None:
    for path in [
        RESULTS_DIR / "best_boosted_feature_baseline.json",
        RESULTS_DIR / "best_strong_baseline_full_train.json",
        RESULTS_DIR / "best_tuned_baseline.json",
    ]:
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8"))["f1_score"])
    return None


def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="max_grad_norm", columns="noise_multiplier", values="f1_score")
    output_path = FIGURES_DIR / "best_variant_50epoch_f1_heatmap.png"
    ensure_dir(output_path.parent)
    plt.figure(figsize=(9, 4.8))
    image = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(image, label="F1-score")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("Noise multiplier")
    plt.ylabel("Max grad norm")
    plt.title("Best Variant 50-Epoch F1 Across Noise x Clip")
    for row_idx, clip in enumerate(pivot.index):
        for col_idx, noise in enumerate(pivot.columns):
            value = pivot.loc[clip, noise]
            plt.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_lines(df: pd.DataFrame, reference_f1: float | None) -> None:
    output_path = FIGURES_DIR / "best_variant_50epoch_f1_lines.png"
    ensure_dir(output_path.parent)
    plt.figure(figsize=(9, 4.8))
    for clip, group in df.groupby("max_grad_norm"):
        group = group.sort_values("noise_multiplier")
        plt.plot(group["noise_multiplier"], group["f1_score"], marker="o", label=f"clip={clip:g}")
    if reference_f1 is not None:
        plt.axhline(reference_f1, color="gray", linestyle="--", label="Best non-DP reference F1")
    plt.xlabel("Noise multiplier")
    plt.ylabel("F1-score")
    plt.title("Best Variant 50-Epoch Noise x Clip Sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary(df: pd.DataFrame, reference_f1: float | None, device: str) -> None:
    ok = df[df["status"] == "ok"].copy()
    best = ok.sort_values("f1_score", ascending=False).iloc[0]
    worst = ok.sort_values("f1_score", ascending=True).iloc[0]
    best_by_noise = (
        ok.sort_values("f1_score", ascending=False)
        .groupby("noise_multiplier", as_index=False)
        .first()
        .sort_values("noise_multiplier")
    )
    summary = {
        "device": device,
        "model_variant": MODEL_VARIANT,
        "epochs": EPOCHS_50,
        "reference_f1": reference_f1,
        "num_ok_runs": int(len(ok)),
        "num_failed_runs": int((df["status"] != "ok").sum()),
        "best_config": best.to_dict(),
        "worst_config": worst.to_dict(),
        "noise_values": NOISE_CLIP_GRID_NOISE_MULTIPLIERS,
        "clip_values": NOISE_CLIP_GRID_MAX_GRAD_NORMS,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Best variant 50-epoch noise x clipping notes",
        "",
        f"- Device: `{device}`.",
        f"- Model variant: `{MODEL_VARIANT['name']}`.",
        f"- Hidden dims: `{MODEL_VARIANT['hidden_dims']}`.",
        f"- Dropout: `{MODEL_VARIANT['dropout']}`.",
        f"- Epochs: `{EPOCHS_50}`.",
        "- Early stopping: disabled; every run uses the full 50 epochs.",
        f"- Completed runs: `{len(ok)}`.",
        f"- Reference non-DP F1-score: `{reference_f1:.4f}`." if reference_f1 is not None else "- Reference non-DP F1-score: not available.",
        "",
        "## Best and worst",
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
        "## Best per noise",
        "",
        "| noise | best_clip | epsilon | F1-score | precision | recall | accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_by_noise.iterrows():
        lines.append(
            f"| {row['noise_multiplier']:.1f} | {row['max_grad_norm']:.1f} | {row['epsilon']:.4f} | "
            f"{row['f1_score']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['accuracy']:.4f} |"
        )
    NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(DOCS_DIR)
    preprocess_adult(save_dir=PROCESSED_DATA_DIR)
    x_train, x_test, y_train, y_test, feature_names = load_processed_data(PROCESSED_DATA_DIR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This sweep is intended to run on the local T1200 GPU.")
    device = "cuda"
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    reference_f1 = load_reference_f1()

    rows: list[dict[str, object]] = []
    if OUTPUT_CSV.exists():
        rows = pd.read_csv(OUTPUT_CSV).to_dict("records")
    completed = {
        (float(row["noise_multiplier"]), float(row["max_grad_norm"]))
        for row in rows
        if row.get("status") == "ok"
    }

    for noise in NOISE_CLIP_GRID_NOISE_MULTIPLIERS:
        for clip in NOISE_CLIP_GRID_MAX_GRAD_NORMS:
            key = (float(noise), float(clip))
            if key in completed:
                print(f"Skipping completed noise={noise} clip={clip}", flush=True)
                continue
            print(f"Running {MODEL_VARIANT['name']} epochs={EPOCHS_50} noise={noise} clip={clip}", flush=True)
            try:
                _, result = train_dp_mlp(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    input_dim=len(feature_names),
                    seed=DEFAULT_SEED,
                    epochs=EPOCHS_50,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE_DP,
                    momentum=MOMENTUM,
                    noise_multiplier=noise,
                    max_grad_norm=clip,
                    delta=DELTA,
                    device=device,
                    hidden_dims=MODEL_VARIANT["hidden_dims"],
                    dropout=MODEL_VARIANT["dropout"],
                    model_variant=MODEL_VARIANT["name"],
                )
                row = {
                    **result,
                    "status": "ok",
                    "device": device,
                    "reference_f1": reference_f1,
                    "f1_drop_vs_reference": None if reference_f1 is None else reference_f1 - float(result["f1_score"]),
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
                    "model_variant": MODEL_VARIANT["name"],
                    "hidden_dims": "-".join(str(dim) for dim in MODEL_VARIANT["hidden_dims"]),
                    "dropout": MODEL_VARIANT["dropout"],
                    "epochs": EPOCHS_50,
                    "noise_multiplier": noise,
                    "max_grad_norm": clip,
                    "status": "failed",
                    "device": device,
                    "reference_f1": reference_f1,
                    "notes": str(exc).splitlines()[0],
                }
                print("Failed:", row["notes"], flush=True)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    plot_heatmap(ok)
    plot_lines(ok, reference_f1)
    write_summary(df, reference_f1, device)
    print("Saved:", OUTPUT_CSV, flush=True)
    print("Saved:", SUMMARY_JSON, flush=True)
    print("Saved:", NOTES_MD, flush=True)


if __name__ == "__main__":
    main()
