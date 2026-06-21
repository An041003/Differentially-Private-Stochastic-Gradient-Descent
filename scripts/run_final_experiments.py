from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BATCH_SIZE,
    BATCH_SIZE_VALUES,
    DEFAULT_SEED,
    DELTA,
    EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE_BASELINE,
    LEARNING_RATE_DP,
    MAX_GRAD_NORM,
    MAX_GRAD_NORM_VALUES,
    MOMENTUM,
    NOISE_MULTIPLIERS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SEEDS,
)
from src.data_preprocessing import load_processed_data, preprocess_adult  # noqa: E402
from src.plotting import (  # noqa: E402
    plot_attack_auc_comparison,
    plot_batch_size_vs_metric,
    plot_max_grad_norm_vs_metric,
    plot_multi_seed_errorbar,
    plot_noise_vs_accuracy,
    plot_noise_vs_epsilon,
    plot_noise_vs_f1,
    plot_privacy_utility_tradeoff,
)
from src.privacy_attack import run_confidence_mia  # noqa: E402
from src.train_baseline import train_logistic_regression, train_mlp_baseline  # noqa: E402
from src.train_dp_sgd import train_dp_mlp  # noqa: E402
from src.utils import ensure_dir, get_device, save_json  # noqa: E402


def write_csv(rows: list[dict[str, object]], path: Path) -> pd.DataFrame:
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print("Saved:", path, flush=True)
    return df


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    metadata = preprocess_adult(save_dir=PROCESSED_DATA_DIR)
    x_train, x_test, y_train, y_test, feature_names = load_processed_data(PROCESSED_DATA_DIR)
    input_dim = len(feature_names)
    device = get_device()

    print("Device:", device, flush=True)
    print("Preprocessing metadata:", metadata, flush=True)

    baseline_rows: list[dict[str, object]] = []
    logreg_result = train_logistic_regression(x_train, y_train, x_test, y_test, seed=DEFAULT_SEED)
    baseline_rows.append(logreg_result)
    print("LogReg:", logreg_result["accuracy"], logreg_result["f1_score"], flush=True)

    mlp_model, mlp_result = train_mlp_baseline(
        x_train,
        y_train,
        x_test,
        y_test,
        input_dim=input_dim,
        seed=DEFAULT_SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE_BASELINE,
        device=device,
    )
    baseline_rows.append(mlp_result)
    print("MLP baseline:", mlp_result["accuracy"], mlp_result["f1_score"], flush=True)
    baseline_df = write_csv(baseline_rows, RESULTS_DIR / "baseline_results.csv")

    noise_rows: list[dict[str, object]] = []
    mia_rows: list[dict[str, object]] = []
    mia_rows.append(
        run_confidence_mia(
            mlp_model,
            x_train,
            x_test,
            device=device,
            model_name="mlp_baseline",
            is_dp=False,
        )
    )

    selected_dp_models: dict[float, tuple[object, dict[str, object]]] = {}
    for noise in NOISE_MULTIPLIERS:
        print(f"Running noise sweep noise={noise}", flush=True)
        model, result = train_dp_mlp(
            x_train,
            y_train,
            x_test,
            y_test,
            input_dim=input_dim,
            seed=DEFAULT_SEED,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE_DP,
            momentum=MOMENTUM,
            noise_multiplier=noise,
            max_grad_norm=MAX_GRAD_NORM,
            delta=DELTA,
            device=device,
        )
        noise_rows.append(result)
        print("Noise result:", noise, result["epsilon"], result["accuracy"], result["f1_score"], flush=True)
        if noise in {1.5, 3.0}:
            selected_dp_models[noise] = (model, result)
    noise_df = write_csv(noise_rows, RESULTS_DIR / "dp_sgd_noise_results.csv")

    for noise, (model, result) in selected_dp_models.items():
        mia_rows.append(
            run_confidence_mia(
                model,
                x_train,
                x_test,
                device=device,
                model_name=f"dp_sgd_noise_{noise}",
                is_dp=True,
                noise_multiplier=noise,
                epsilon=float(result["epsilon"]),
            )
        )
    write_csv(mia_rows, RESULTS_DIR / "mia_results.csv")

    max_norm_rows: list[dict[str, object]] = []
    for max_grad_norm in MAX_GRAD_NORM_VALUES:
        print(f"Running max_grad_norm sweep norm={max_grad_norm}", flush=True)
        _, result = train_dp_mlp(
            x_train,
            y_train,
            x_test,
            y_test,
            input_dim=input_dim,
            seed=DEFAULT_SEED,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE_DP,
            momentum=MOMENTUM,
            noise_multiplier=1.5,
            max_grad_norm=max_grad_norm,
            delta=DELTA,
            device=device,
        )
        max_norm_rows.append(result)
        print("Max norm result:", max_grad_norm, result["epsilon"], result["accuracy"], result["f1_score"], flush=True)
    write_csv(max_norm_rows, RESULTS_DIR / "max_grad_norm_results.csv")

    batch_rows: list[dict[str, object]] = []
    for batch_size in BATCH_SIZE_VALUES:
        print(f"Running batch size sweep batch_size={batch_size}", flush=True)
        _, result = train_dp_mlp(
            x_train,
            y_train,
            x_test,
            y_test,
            input_dim=input_dim,
            seed=DEFAULT_SEED,
            epochs=EPOCHS,
            batch_size=batch_size,
            learning_rate=LEARNING_RATE_DP,
            momentum=MOMENTUM,
            noise_multiplier=1.5,
            max_grad_norm=MAX_GRAD_NORM,
            delta=DELTA,
            device=device,
        )
        batch_rows.append(result)
        print("Batch result:", batch_size, result["epsilon"], result["accuracy"], result["f1_score"], flush=True)
    write_csv(batch_rows, RESULTS_DIR / "batch_size_results.csv")

    multi_rows: list[dict[str, object]] = []
    multi_rows.append({**mlp_result, "model_config": "mlp_baseline"})
    for row in noise_rows:
        if row["noise_multiplier"] in {1.0, 1.5, 3.0}:
            multi_rows.append({**row, "model_config": f"dp_noise_{row['noise_multiplier']}"})

    for seed in [seed for seed in SEEDS if seed != DEFAULT_SEED]:
        print(f"Running multi-seed baseline seed={seed}", flush=True)
        _, result = train_mlp_baseline(
            x_train,
            y_train,
            x_test,
            y_test,
            input_dim=input_dim,
            seed=seed,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE_BASELINE,
            device=device,
        )
        multi_rows.append({**result, "model_config": "mlp_baseline"})
        for noise in [1.0, 1.5, 3.0]:
            print(f"Running multi-seed DP seed={seed} noise={noise}", flush=True)
            _, result = train_dp_mlp(
                x_train,
                y_train,
                x_test,
                y_test,
                input_dim=input_dim,
                seed=seed,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE_DP,
                momentum=MOMENTUM,
                noise_multiplier=noise,
                max_grad_norm=MAX_GRAD_NORM,
                delta=DELTA,
                device=device,
            )
            multi_rows.append({**result, "model_config": f"dp_noise_{noise}"})
    multi_df = write_csv(multi_rows, RESULTS_DIR / "multi_seed_runs.csv")
    summary_df = (
        multi_df.groupby("model_config")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1_score=("f1_score", "mean"),
            std_f1_score=("f1_score", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_epsilon=("epsilon", "mean"),
            std_epsilon=("epsilon", "std"),
        )
        .reset_index()
    )
    summary_df.to_csv(RESULTS_DIR / "multi_seed_results.csv", index=False)
    print("Saved:", RESULTS_DIR / "multi_seed_results.csv", flush=True)

    plot_noise_vs_epsilon(RESULTS_DIR / "dp_sgd_noise_results.csv", FIGURES_DIR / "noise_vs_epsilon.png")
    plot_noise_vs_accuracy(RESULTS_DIR / "dp_sgd_noise_results.csv", FIGURES_DIR / "noise_vs_accuracy.png")
    plot_noise_vs_f1(RESULTS_DIR / "dp_sgd_noise_results.csv", FIGURES_DIR / "noise_vs_f1.png")
    plot_privacy_utility_tradeoff(RESULTS_DIR / "dp_sgd_noise_results.csv", FIGURES_DIR / "privacy_utility_tradeoff.png")
    plot_max_grad_norm_vs_metric(RESULTS_DIR / "max_grad_norm_results.csv", "accuracy", FIGURES_DIR / "max_grad_norm_vs_accuracy.png")
    plot_max_grad_norm_vs_metric(RESULTS_DIR / "max_grad_norm_results.csv", "f1_score", FIGURES_DIR / "max_grad_norm_vs_f1.png")
    plot_batch_size_vs_metric(RESULTS_DIR / "batch_size_results.csv", "epsilon", FIGURES_DIR / "batch_size_vs_epsilon.png")
    plot_batch_size_vs_metric(RESULTS_DIR / "batch_size_results.csv", "accuracy", FIGURES_DIR / "batch_size_vs_accuracy.png")
    plot_attack_auc_comparison(RESULTS_DIR / "mia_results.csv", FIGURES_DIR / "attack_auc_comparison.png")
    plot_multi_seed_errorbar(RESULTS_DIR / "multi_seed_results.csv", "accuracy", FIGURES_DIR / "multi_seed_errorbar_accuracy.png")
    plot_multi_seed_errorbar(RESULTS_DIR / "multi_seed_results.csv", "f1_score", FIGURES_DIR / "multi_seed_errorbar_f1.png")

    best_noise = noise_df.sort_values(["epsilon", "accuracy"], ascending=[True, False]).iloc[0]
    best_clip = pd.DataFrame(max_norm_rows).sort_values("f1_score", ascending=False).iloc[0]
    mia_df = pd.DataFrame(mia_rows)
    final_summary = [
        {
            "section": "noise_sweep",
            "main_finding": "Higher noise reduces epsilon with modest utility changes.",
            "best_config": "noise_multiplier=1.5 remains the balanced presentation setting",
            "epsilon": float(noise_df.loc[noise_df["noise_multiplier"] == 1.5, "epsilon"].iloc[0]),
            "accuracy": float(noise_df.loc[noise_df["noise_multiplier"] == 1.5, "accuracy"].iloc[0]),
            "f1_score": float(noise_df.loc[noise_df["noise_multiplier"] == 1.5, "f1_score"].iloc[0]),
            "attack_auc": None,
            "interpretation": "Balanced privacy-utility choice for final discussion.",
        },
        {
            "section": "strongest_privacy_noise",
            "main_finding": "The smallest epsilon in the sweep comes from the largest noise.",
            "best_config": f"noise_multiplier={best_noise['noise_multiplier']}",
            "epsilon": float(best_noise["epsilon"]),
            "accuracy": float(best_noise["accuracy"]),
            "f1_score": float(best_noise["f1_score"]),
            "attack_auc": None,
            "interpretation": "Best formal privacy, not necessarily best utility.",
        },
        {
            "section": "max_grad_norm",
            "main_finding": "Clipping norm mainly affects utility when other DP parameters are fixed.",
            "best_config": f"max_grad_norm={best_clip['max_grad_norm']}",
            "epsilon": float(best_clip["epsilon"]),
            "accuracy": float(best_clip["accuracy"]),
            "f1_score": float(best_clip["f1_score"]),
            "attack_auc": None,
            "interpretation": "Select by F1-score because epsilon is unchanged under fixed sampling/noise.",
        },
    ]
    for _, row in mia_df.iterrows():
        final_summary.append(
            {
                "section": "membership_inference",
                "main_finding": "Confidence-based attack result for trained model.",
                "best_config": row["model_name"],
                "epsilon": row["epsilon"],
                "accuracy": None,
                "f1_score": None,
                "attack_auc": row["attack_auc"],
                "interpretation": "Attack AUC closer to 0.5 means lower membership signal.",
            }
        )
    pd.DataFrame(final_summary).to_csv(RESULTS_DIR / "final_summary.csv", index=False)
    save_json({"device": device, "metadata": metadata}, RESULTS_DIR / "run_metadata.json")
    print("Saved:", RESULTS_DIR / "final_summary.csv", flush=True)


if __name__ == "__main__":
    main()
