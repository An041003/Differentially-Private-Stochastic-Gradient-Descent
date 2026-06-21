from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_SEED, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data_preprocessing import load_processed_data, preprocess_adult  # noqa: E402
from src.evaluate import evaluate_predictions  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

OUTPUT_CSV = RESULTS_DIR / "strong_baseline_results.csv"
BEST_JSON = RESULTS_DIR / "best_strong_baseline.json"


def find_best_threshold(y_true: np.ndarray, positive_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (positive_proba >= threshold).astype("int64")
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def make_candidates() -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    for learning_rate in [0.03, 0.05, 0.08]:
        for max_leaf_nodes in [15, 31, 63]:
            for l2_regularization in [0.0, 0.01, 0.1]:
                candidates.append(
                    (
                        f"hgb_lr{learning_rate}_leaf{max_leaf_nodes}_l2{l2_regularization}",
                        HistGradientBoostingClassifier(
                            learning_rate=learning_rate,
                            max_iter=500,
                            max_leaf_nodes=max_leaf_nodes,
                            l2_regularization=l2_regularization,
                            early_stopping=True,
                            validation_fraction=0.1,
                            random_state=DEFAULT_SEED,
                        ),
                    )
                )
    for c_value in [0.5, 1.0, 2.0]:
        for class_weight in [None, "balanced"]:
            candidates.append(
                (
                    f"logreg_C{c_value}_cw{class_weight or 'none'}",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2000,
                        class_weight=class_weight,
                        random_state=DEFAULT_SEED,
                    ),
                )
            )
    for class_weight in [None, "balanced"]:
        for max_depth in [12, 18, None]:
            candidates.append(
                (
                    f"extra_trees_depth{max_depth or 'none'}_cw{class_weight or 'none'}",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=max_depth,
                        min_samples_leaf=2,
                        class_weight=class_weight,
                        n_jobs=-1,
                        random_state=DEFAULT_SEED,
                    ),
                )
            )
            candidates.append(
                (
                    f"random_forest_depth{max_depth or 'none'}_cw{class_weight or 'none'}",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=max_depth,
                        min_samples_leaf=2,
                        class_weight=class_weight,
                        n_jobs=-1,
                        random_state=DEFAULT_SEED,
                    ),
                )
            )
    return candidates


def evaluate_candidate(name: str, model, x_fit, y_fit, x_val, y_val, x_test, y_test) -> dict[str, object]:
    start = time.time()
    model.fit(x_fit, y_fit)
    training_time = time.time() - start
    val_proba = model.predict_proba(x_val)[:, 1]
    threshold, val_f1 = find_best_threshold(y_val, val_proba)

    test_proba = model.predict_proba(x_test)
    default_pred = (test_proba[:, 1] >= 0.5).astype("int64")
    tuned_pred = (test_proba[:, 1] >= threshold).astype("int64")
    default_metrics = evaluate_predictions(y_test, default_pred, test_proba)
    tuned_metrics = evaluate_predictions(y_test, tuned_pred, test_proba)

    row: dict[str, object] = {
        "model_name": name,
        "threshold": threshold,
        "validation_f1_at_threshold": val_f1,
        "default_accuracy": default_metrics["accuracy"],
        "default_precision": default_metrics["precision"],
        "default_recall": default_metrics["recall"],
        "default_f1_score": default_metrics["f1_score"],
        "accuracy": tuned_metrics["accuracy"],
        "precision": tuned_metrics["precision"],
        "recall": tuned_metrics["recall"],
        "f1_score": tuned_metrics["f1_score"],
        "roc_auc": tuned_metrics["roc_auc"],
        "pr_auc": tuned_metrics["pr_auc"],
        "confusion_matrix": tuned_metrics["confusion_matrix"],
        "training_time": training_time,
        "threshold_source": "train_validation_split",
        "status": "ok",
    }
    return row


def main() -> None:
    ensure_dir(RESULTS_DIR)
    preprocess_adult(save_dir=PROCESSED_DATA_DIR)
    x_train, x_test, y_train, y_test, _ = load_processed_data(PROCESSED_DATA_DIR)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=DEFAULT_SEED,
    )

    rows: list[dict[str, object]] = []
    for name, model in make_candidates():
        print("Candidate:", name, flush=True)
        try:
            row = evaluate_candidate(name, model, x_fit, y_fit, x_val, y_val, x_test, y_test)
            print(
                f"Result f1={row['f1_score']:.4f} acc={row['accuracy']:.4f} "
                f"default_f1={row['default_f1_score']:.4f} threshold={row['threshold']:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "model_name": name,
                "status": "failed",
                "notes": str(exc).splitlines()[0],
            }
            print("Failed:", row["notes"], flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    best = ok.sort_values("f1_score", ascending=False).iloc[0].to_dict()
    BEST_JSON.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    print("Saved:", OUTPUT_CSV, flush=True)
    print("Saved:", BEST_JSON, flush=True)
    print("Best:", best, flush=True)


if __name__ == "__main__":
    main()
