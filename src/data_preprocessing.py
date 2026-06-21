from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .utils import ensure_dir, save_json


RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_COLUMN = "income"


def load_adult_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        names=RAW_COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|",
    )
    return clean_adult_dataframe(df)


def clean_adult_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace(".", "", regex=False)
    return df.dropna().reset_index(drop=True)


def preprocess_adult(
    raw_dir: str | Path = RAW_DATA_DIR,
    save_dir: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    save_dir = ensure_dir(save_dir)

    train_df = load_adult_file(raw_dir / "adult.data")
    test_df = load_adult_file(raw_dir / "adult.test")

    x_train_raw = train_df.drop(columns=TARGET_COLUMN)
    x_test_raw = test_df.drop(columns=TARGET_COLUMN)
    y_train = (train_df[TARGET_COLUMN] == ">50K").astype("int64").to_numpy()
    y_test = (test_df[TARGET_COLUMN] == ">50K").astype("int64").to_numpy()

    combined = pd.concat([x_train_raw, x_test_raw], axis=0)
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLUMNS, dtype="float32")

    x_train = combined.iloc[: len(train_df)].copy()
    x_test = combined.iloc[len(train_df) :].copy()

    means = x_train[NUMERIC_COLUMNS].mean()
    stds = x_train[NUMERIC_COLUMNS].std().replace(0, 1)
    x_train[NUMERIC_COLUMNS] = (x_train[NUMERIC_COLUMNS] - means) / stds
    x_test[NUMERIC_COLUMNS] = (x_test[NUMERIC_COLUMNS] - means) / stds

    feature_names = list(x_train.columns)
    x_train_np = x_train.to_numpy(dtype="float32")
    x_test_np = x_test.to_numpy(dtype="float32")

    np.save(save_dir / "X_train.npy", x_train_np)
    np.save(save_dir / "X_test.npy", x_test_np)
    np.save(save_dir / "y_train.npy", y_train)
    np.save(save_dir / "y_test.npy", y_test)
    save_json({"feature_names": feature_names}, save_dir / "feature_names.json")

    metadata = {
        "dataset": "UCI Adult",
        "num_train_samples": int(x_train_np.shape[0]),
        "num_test_samples": int(x_test_np.shape[0]),
        "num_features": int(x_train_np.shape[1]),
        "target_mapping": {"<=50K": 0, ">50K": 1},
        "missing_value_strategy": "drop_rows",
        "categorical_encoding": "one_hot",
        "numeric_scaling": "train_standardization",
        "split_strategy": "original_adult_train_test_files",
    }
    save_json(metadata, save_dir / "preprocessing_metadata.json")
    return metadata


def load_processed_data(
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    processed_dir = Path(processed_dir)
    x_train = np.load(processed_dir / "X_train.npy")
    x_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    feature_names = pd.read_json(processed_dir / "feature_names.json")["feature_names"].tolist()
    return x_train, x_test, y_train, y_test, feature_names
