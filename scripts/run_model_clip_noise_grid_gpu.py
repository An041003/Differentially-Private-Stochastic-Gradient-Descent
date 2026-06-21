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
OUTPUT_CSV = RESULTS_DIR / "model_clip_noise_grid_results.csv"
SUMMARY_JSON = RESULTS_DIR / "model_clip_noise_grid_summary.json"
NOTES_MD = DOCS_DIR / "model_clip_noise_grid_notes.md"

MODEL_VARIANTS = [
    {"name": "mlp_64_32_d0p1", "hidden_dims": (64, 32), "dropout": 0.1},
    {"name": "mlp_128_64_d0p1", "hidden_dims": (128, 64), "dropout": 0.1},
    {"name": "mlp_128_64_d0p0", "hidden_dims": (128, 64), "dropout": 0.0},
    {"name": "mlp_256_128_d0p1", "hidden_dims": (256, 128), "dropout": 0.1},
]


def load_reference_f1() -> float | None:
    for path in [
        RESULTS_DIR / "best_boosted_feature_baseline.json",
        RESULTS_DIR / "best_strong_baseline_full_train.json",
        RESULTS_DIR / "best_tuned_baseline.json",
    ]:
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8"))["f1_score"])
    return None


def plot_best_by_noise_model(df: pd.DataFrame, reference_f1: float | None) -> None:
    output_path = FIGURES_DIR / "model_clip_noise_best_f1.png"
    best = (
        df.sort_values("f1_score", ascending=False)
        .groupby(["model_variant", "noise_multiplier"], as_index=False)
        .first()
        .sort_values(["model_variant", "noise_multiplier"])
    )
    ensure_dir(output_path.parent)
    plt.figure(figsize=(10, 5.2))
    for model_variant, group in best.groupby("model_variant"):
        plt.plot(group["noise_multiplier"], group["f1_score"], marker="o", label=model_variant)
    if reference_f1 is not None:
        plt.axhline(reference_f1, color="gray", linestyle="--", label="Best non-DP reference F1")
    plt.xlabel("Noise multiplier")
    plt.ylabel("Best F1-score over clipping")
    plt.title("DP-SGD Model x Clip x Noise Grid")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary(df: pd.DataFrame, reference_f1: float | None, device: str) -> None:
    ok = df[df["status"] == "ok"].copy()
    best = ok.sort_values("f1_score", ascending=False).iloc[0]
    worst = ok.sort_values("f1_score", ascending=True).iloc[0]
    by_model = (
        ok.sort_values("f1_score", ascending=False)
        .groupby("model_variant", as_index=False)
        .first()
        .sort_values("f1_score", ascending=False)
    )
    by_noise = (
        ok.sort_values("f1_score", ascending=False)
        .groupby("noise_multiplier", as_index=False)
        .first()
        .sort_values("noise_multiplier")
    )
    summary = {
        "device": device,
        "reference_f1": reference_f1,
        "num_ok_runs": int(len(ok)),
        "num_failed_runs": int((df["status"] != "ok").sum()),
        "best_config": best.to_dict(),
        "worst_config": worst.to_dict(),
        "model_variants": MODEL_VARIANTS,
        "noise_values": NOISE_CLIP_GRID_NOISE_MULTIPLIERS,
        "clip_values": NOISE_CLIP_GRID_MAX_GRAD_NORMS,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Model x clipping x noise grid notes",
        "",
        f"- Device: `{device}`.",
        f"- Completed runs: `{len(ok)}`.",
        f"- Reference non-DP F1-score: `{reference_f1:.4f}`." if reference_f1 is not None else "- Reference non-DP F1-score: not available.",
        "- Grid: 4 MLP model variants x 4 clipping norms x 9 noise multipliers.",
        "",
        "## Best and worst",
        "",
        (
            f"- Best DP F1: model `{best['model_variant']}`, noise `{best['noise_multiplier']}`, "
            f"clip `{best['max_grad_norm']}`, epsilon `{best['epsilon']:.4f}`, F1 `{best['f1_score']:.4f}`."
        ),
        (
            f"- Worst DP F1: model `{worst['model_variant']}`, noise `{worst['noise_multiplier']}`, "
            f"clip `{worst['max_grad_norm']}`, epsilon `{worst['epsilon']:.4f}`, F1 `{worst['f1_score']:.4f}`."
        ),
        "",
        "## Best per model",
        "",
        "| model_variant | noise | clip | epsilon | F1-score | precision | recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_model.iterrows():
        lines.append(
            f"| {row['model_variant']} | {row['noise_multiplier']:.1f} | {row['max_grad_norm']:.1f} | "
            f"{row['epsilon']:.4f} | {row['f1_score']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )
    lines.extend(["", "## Best per noise", "", "| noise | model_variant | clip | epsilon | F1-score |", "|---:|---|---:|---:|---:|"])
    for _, row in by_noise.iterrows():
        lines.append(
            f"| {row['noise_multiplier']:.1f} | {row['model_variant']} | {row['max_grad_norm']:.1f} | "
            f"{row['epsilon']:.4f} | {row['f1_score']:.4f} |"
        )
    NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(DOCS_DIR)
    preprocess_adult(save_dir=PROCESSED_DATA_DIR)
    x_train, x_test, y_train, y_test, feature_names = load_processed_data(PROCESSED_DATA_DIR)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This grid is intended to run on the local T1200 GPU.")
    device = "cuda"
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    reference_f1 = load_reference_f1()

    rows: list[dict[str, object]] = []
    if OUTPUT_CSV.exists():
        rows = pd.read_csv(OUTPUT_CSV).to_dict("records")
    completed = {
        (str(row["model_variant"]), float(row["noise_multiplier"]), float(row["max_grad_norm"]))
        for row in rows
        if row.get("status") == "ok"
    }

    for variant in MODEL_VARIANTS:
        for noise in NOISE_CLIP_GRID_NOISE_MULTIPLIERS:
            for clip in NOISE_CLIP_GRID_MAX_GRAD_NORMS:
                key = (variant["name"], float(noise), float(clip))
                if key in completed:
                    print(f"Skipping completed model={variant['name']} noise={noise} clip={clip}", flush=True)
                    continue
                print(f"Running model={variant['name']} noise={noise} clip={clip}", flush=True)
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
                        hidden_dims=variant["hidden_dims"],
                        dropout=variant["dropout"],
                        model_variant=variant["name"],
                    )
                    row = {
                        **result,
                        "status": "ok",
                        "device": device,
                        "reference_f1": reference_f1,
                        "f1_drop_vs_reference": None if reference_f1 is None else reference_f1 - float(result["f1_score"]),
                    }
                    print(
                        f"Result model={variant['name']} noise={noise} clip={clip} "
                        f"eps={row['epsilon']:.4f} f1={row['f1_score']:.4f}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                    row = {
                        "model_variant": variant["name"],
                        "hidden_dims": "-".join(str(dim) for dim in variant["hidden_dims"]),
                        "dropout": variant["dropout"],
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
    plot_best_by_noise_model(ok, reference_f1)
    write_summary(df, reference_f1, device)
    print("Saved:", OUTPUT_CSV, flush=True)
    print("Saved:", SUMMARY_JSON, flush=True)
    print("Saved:", NOTES_MD, flush=True)


if __name__ == "__main__":
    main()
