from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_SEED, RAW_DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data_preprocessing import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    load_adult_file,
)
from src.evaluate import evaluate_predictions  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

OUTPUT_CSV = RESULTS_DIR / "boosted_feature_baseline_results.csv"
BEST_JSON = RESULTS_DIR / "best_boosted_feature_baseline.json"


def load_raw_split() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_df = load_adult_file(RAW_DATA_DIR / "adult.data")
    test_df = load_adult_file(RAW_DATA_DIR / "adult.test")
    x_train = train_df.drop(columns=TARGET_COLUMN)
    x_test = test_df.drop(columns=TARGET_COLUMN)
    y_train = (train_df[TARGET_COLUMN] == ">50K").astype("int64").to_numpy()
    y_test = (test_df[TARGET_COLUMN] == ">50K").astype("int64").to_numpy()
    return x_train, x_test, y_train, y_test


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    out["has_capital_gain"] = (out["capital-gain"] > 0).astype("int64")
    out["has_capital_loss"] = (out["capital-loss"] > 0).astype("int64")
    out["hours_age_ratio"] = out["hours-per-week"] / out["age"].clip(lower=1)
    out["education_hours"] = out["education-num"] * out["hours-per-week"]
    out["age_bin"] = pd.cut(
        out["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", ">65"],
    ).astype(str)
    out["hours_bin"] = pd.cut(
        out["hours-per-week"],
        bins=[0, 20, 35, 40, 50, 60, 100],
        labels=["<=20", "21-35", "36-40", "41-50", "51-60", ">60"],
    ).astype(str)
    out["education_bin"] = pd.cut(
        out["education-num"],
        bins=[0, 8, 10, 12, 14, 16],
        labels=["low", "hs", "some_college", "college", "grad"],
        include_lowest=True,
    ).astype(str)
    out["occupation_education"] = out["occupation"].astype(str) + "__" + out["education"].astype(str)
    out["marital_relationship"] = out["marital-status"].astype(str) + "__" + out["relationship"].astype(str)
    out["sex_marital"] = out["sex"].astype(str) + "__" + out["marital-status"].astype(str)
    out["workclass_occupation"] = out["workclass"].astype(str) + "__" + out["occupation"].astype(str)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = [
        *NUMERIC_COLUMNS,
        "capital_net",
        "has_capital_gain",
        "has_capital_loss",
        "hours_age_ratio",
        "education_hours",
    ]
    categorical = [
        *CATEGORICAL_COLUMNS,
        "age_bin",
        "hours_bin",
        "education_bin",
        "occupation_education",
        "marital_relationship",
        "sex_marital",
        "workclass_occupation",
    ]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def find_best_threshold(y_true: np.ndarray, positive_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (positive_proba >= threshold).astype("int64")
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def make_candidates(pos_weight: float) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    for max_depth in [3, 4, 5]:
        for learning_rate in [0.03, 0.05, 0.08]:
            candidates.append(
                (
                    f"xgb_depth{max_depth}_lr{learning_rate}",
                    XGBClassifier(
                        n_estimators=700,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        min_child_weight=3,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        scale_pos_weight=pos_weight,
                        n_jobs=-1,
                        random_state=DEFAULT_SEED,
                    ),
                )
            )
            candidates.append(
                (
                    f"lgbm_leaves31_depth{max_depth}_lr{learning_rate}",
                    LGBMClassifier(
                        n_estimators=700,
                        max_depth=max_depth,
                        num_leaves=31,
                        learning_rate=learning_rate,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        class_weight="balanced",
                        random_state=DEFAULT_SEED,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                )
            )
    for depth in [4, 6, 8]:
        for learning_rate in [0.03, 0.05]:
            candidates.append(
                (
                    f"cat_depth{depth}_lr{learning_rate}",
                    CatBoostClassifier(
                        iterations=700,
                        depth=depth,
                        learning_rate=learning_rate,
                        loss_function="Logloss",
                        auto_class_weights="Balanced",
                        random_seed=DEFAULT_SEED,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
    for learning_rate in [0.03, 0.05, 0.08]:
        candidates.append(
            (
                f"hgb_fe_lr{learning_rate}",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=700,
                    max_leaf_nodes=31,
                    l2_regularization=0.01,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=DEFAULT_SEED,
                ),
            )
        )
    return candidates


def evaluate_candidate(name: str, model, x_fit, y_fit, x_val, y_val, x_train, y_train, x_test, y_test) -> dict[str, object]:
    pipe = Pipeline(
        [
            ("features", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
    start = time.time()
    pipe.fit(x_fit, y_fit)
    training_time = time.time() - start
    val_proba = pipe.predict_proba(x_val)[:, 1]
    threshold, val_f1 = find_best_threshold(y_val, val_proba)

    full_pipe = Pipeline(
        [
            ("features", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("model", model.__class__(**model.get_params())),
        ]
    )
    full_start = time.time()
    full_pipe.fit(x_train, y_train)
    full_training_time = time.time() - full_start
    test_proba = full_pipe.predict_proba(x_test)
    default_pred = (test_proba[:, 1] >= 0.5).astype("int64")
    tuned_pred = (test_proba[:, 1] >= threshold).astype("int64")
    default_metrics = evaluate_predictions(y_test, default_pred, test_proba)
    tuned_metrics = evaluate_predictions(y_test, tuned_pred, test_proba)
    return {
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
        "validation_training_time": training_time,
        "full_training_time": full_training_time,
        "threshold_source": "train_validation_split",
        "features": "numeric_scaling_onehot_bins_interactions",
        "status": "ok",
    }


def main() -> None:
    ensure_dir(RESULTS_DIR)
    x_train, x_test, y_train, y_test = load_raw_split()
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=DEFAULT_SEED,
    )
    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    pos_weight = neg / pos
    rows: list[dict[str, object]] = []
    for name, model in make_candidates(pos_weight):
        print("Candidate:", name, flush=True)
        try:
            row = evaluate_candidate(name, model, x_fit, y_fit, x_val, y_val, x_train, y_train, x_test, y_test)
            print(
                f"Result default_acc={row['default_accuracy']:.4f} "
                f"default_f1={row['default_f1_score']:.4f} "
                f"f1={row['f1_score']:.4f} auc={row['roc_auc']:.4f}",
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
