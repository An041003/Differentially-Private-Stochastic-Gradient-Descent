from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def _save_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: str | Path,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_noise_vs_epsilon(results_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(results_csv)
    _save_line_plot(df, "noise_multiplier", "epsilon", output_path, "Noise Multiplier vs Epsilon")


def plot_noise_vs_accuracy(results_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(results_csv)
    _save_line_plot(df, "noise_multiplier", "accuracy", output_path, "Noise Multiplier vs Accuracy")


def plot_noise_vs_f1(results_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(results_csv)
    _save_line_plot(df, "noise_multiplier", "f1_score", output_path, "Noise Multiplier vs F1-score")


def plot_privacy_utility_tradeoff(results_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(results_csv)
    _save_line_plot(df, "epsilon", "accuracy", output_path, "Privacy Utility Tradeoff", "Epsilon", "Accuracy")


def plot_max_grad_norm_vs_metric(results_csv: str | Path, metric: str, output_path: str | Path) -> None:
    df = pd.read_csv(results_csv)
    _save_line_plot(df, "max_grad_norm", metric, output_path, f"Max Grad Norm vs {metric}")


def plot_batch_size_vs_metric(results_csv: str | Path, metric: str, output_path: str | Path) -> None:
    df = pd.read_csv(results_csv)
    _save_line_plot(df, "batch_size", metric, output_path, f"Batch Size vs {metric}")


def plot_attack_auc_comparison(mia_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(mia_csv)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.figure(figsize=(7, 4))
    labels = df["model_name"].astype(str)
    plt.bar(labels, df["attack_auc"])
    plt.axhline(0.5, color="gray", linestyle="--", label="Random guessing")
    plt.ylabel("Attack AUC")
    plt.title("Membership Inference Attack AUC")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_multi_seed_errorbar(summary_csv: str | Path, metric: str, output_path: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"
    plt.figure(figsize=(8, 4))
    plt.errorbar(df["model_config"], df[mean_col], yerr=df[std_col], marker="o", capsize=4)
    plt.ylabel(metric)
    plt.title(f"Multi-seed {metric}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
