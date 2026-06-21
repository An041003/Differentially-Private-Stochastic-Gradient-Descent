from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "adult"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 20
LR_DP = 0.05
MOMENTUM = 0.9
DELTA = 1e-5
NOISE_MULTIPLIER = 1.5
MAX_GRAD_NORM_LIST = [0.5, 1.0, 1.5, 2.0]
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

numeric_cols = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def load_adult_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    df = pd.read_csv(
        path,
        names=columns,
        skipinitialspace=True,
        na_values="?",
        comment="|",
    )
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df.dropna().reset_index(drop=True)


def load_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_df = load_adult_csv(DATA_DIR / "adult.data")
    test_df = load_adult_csv(DATA_DIR / "adult.test")

    x_train_raw = train_df.drop(columns="income")
    x_test_raw = test_df.drop(columns="income")
    y_train = (train_df["income"] == ">50K").astype("float32")
    y_test = (test_df["income"] == ">50K").astype("float32")

    combined = pd.concat([x_train_raw, x_test_raw], axis=0)
    categorical_cols = [col for col in combined.columns if col not in numeric_cols]
    combined = pd.get_dummies(combined, columns=categorical_cols, dtype="float32")

    x_train = combined.iloc[: len(train_df)].copy()
    x_test = combined.iloc[len(train_df) :].copy()

    means = x_train[numeric_cols].mean()
    stds = x_train[numeric_cols].std().replace(0, 1)
    x_train[numeric_cols] = (x_train[numeric_cols] - means) / stds
    x_test[numeric_cols] = (x_test[numeric_cols] - means) / stds

    print("Train shape:", train_df.shape, flush=True)
    print("Test shape:", test_df.shape, flush=True)
    print("Input features:", x_train.shape[1], flush=True)

    return (
        torch.tensor(x_train.values, dtype=torch.float32),
        torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32),
        torch.tensor(x_test.values, dtype=torch.float32),
        torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32),
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


criterion = nn.BCEWithLogitsLoss()


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            preds = (torch.sigmoid(logits) >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.numel()
            true_positive += ((preds == 1) & (labels == 1)).sum().item()
            false_positive += ((preds == 1) & (labels == 0)).sum().item()
            false_negative += ((preds == 0) & (labels == 1)).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def train_private(
    max_grad_norm: float,
    x_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    test_loader: DataLoader,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    seed_everything(RANDOM_SEED)
    model = MLP(input_dim=x_train_tensor.shape[1]).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_DP, momentum=MOMENTUM)
    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    privacy_engine = PrivacyEngine(accountant="prv")
    model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=max_grad_norm,
    )

    history: list[dict[str, float]] = []
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, labels in private_train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            total_rows += features.size(0)

        metrics = evaluate(model, test_loader)
        epsilon = privacy_engine.get_epsilon(DELTA)
        epoch_row = {
            "max_grad_norm": max_grad_norm,
            "epoch": epoch,
            "epsilon": epsilon,
            "train_loss": total_loss / max(total_rows, 1),
            **metrics,
        }
        history.append(epoch_row)
        print(
            f"norm={max_grad_norm:.1f} epoch={epoch:02d} "
            f"epsilon={epsilon:.4f} acc={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1_score']:.4f}",
            flush=True,
        )

    elapsed = time.time() - start_time
    final_metrics = evaluate(model, test_loader)
    final_row = {
        "max_grad_norm": max_grad_norm,
        "epsilon": privacy_engine.get_epsilon(DELTA),
        **final_metrics,
        "training_time": elapsed,
    }
    return final_row, history


def save_plots(df: pd.DataFrame) -> None:
    plots = [
        ("accuracy", "Accuracy", "max_grad_norm_vs_accuracy.png", "tab:blue"),
        ("f1_score", "F1-score", "max_grad_norm_vs_f1_score.png", "tab:green"),
        ("training_time", "Training time (seconds)", "max_grad_norm_vs_training_time.png", "tab:orange"),
    ]
    for column, ylabel, filename, color in plots:
        plt.figure(figsize=(7, 4))
        plt.plot(df["max_grad_norm"], df[column], marker="o", color=color)
        plt.xlabel("max_grad_norm")
        plt.ylabel(ylabel)
        plt.title(f"Effect of Clipping Norm on {ylabel}")
        plt.grid(True)
        plt.tight_layout()
        path = OUTPUT_DIR / filename
        plt.savefig(path, dpi=200)
        plt.close()
        print("Saved:", path, flush=True)


def main() -> None:
    print("Device:", DEVICE, flush=True)
    print("Torch:", torch.__version__, flush=True)
    seed_everything(RANDOM_SEED)

    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = load_tensors()
    test_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_results: list[dict[str, float]] = []
    all_history: list[dict[str, float]] = []
    for max_grad_norm in MAX_GRAD_NORM_LIST:
        print("\n" + "=" * 72, flush=True)
        print(f"Training DP-SGD with max_grad_norm={max_grad_norm}", flush=True)
        print("=" * 72, flush=True)
        final_row, history = train_private(
            max_grad_norm=max_grad_norm,
            x_train_tensor=x_train_tensor,
            y_train_tensor=y_train_tensor,
            test_loader=test_loader,
        )
        all_results.append(final_row)
        all_history.extend(history)

    df_results = pd.DataFrame(all_results)
    df_history = pd.DataFrame(all_history)

    result_csv = OUTPUT_DIR / "uci_adult_dp_sgd_max_grad_norm_sweep.csv"
    history_csv = OUTPUT_DIR / "uci_adult_dp_sgd_max_grad_norm_history.csv"
    summary_json = OUTPUT_DIR / "uci_adult_dp_sgd_max_grad_norm_sweep_summary.json"

    df_results.to_csv(result_csv, index=False)
    df_history.to_csv(history_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "device": str(DEVICE),
                "torch": torch.__version__,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LR_DP,
                "momentum": MOMENTUM,
                "delta": DELTA,
                "noise_multiplier": NOISE_MULTIPLIER,
                "results": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_plots(df_results)

    print("\nFinal sweep results:", flush=True)
    print(df_results.to_string(index=False), flush=True)
    print("Saved:", result_csv, flush=True)
    print("Saved:", history_csv, flush=True)
    print("Saved:", summary_json, flush=True)


if __name__ == "__main__":
    main()
