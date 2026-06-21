from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_SEED, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data_preprocessing import load_processed_data, preprocess_adult  # noqa: E402
from src.evaluate import evaluate_torch_model  # noqa: E402
from src.models import AdultMLP  # noqa: E402
from src.utils import ensure_dir, set_seed  # noqa: E402

OUTPUT_CSV = RESULTS_DIR / "tuned_baseline_results.csv"
BEST_JSON = RESULTS_DIR / "best_tuned_baseline.json"


def class_weights(y_train, mode: str, device: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = pd.Series(y_train).value_counts().sort_index()
    neg = float(counts.loc[0])
    pos = float(counts.loc[1])
    if mode == "balanced":
        total = neg + pos
        return torch.tensor([total / (2 * neg), total / (2 * pos)], dtype=torch.float32, device=device)
    if mode == "mild":
        return torch.tensor([1.0, (neg / pos) ** 0.5], dtype=torch.float32, device=device)
    raise ValueError(f"Unknown class weight mode: {mode}")


def train_candidate(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    hidden_dims: tuple[int, ...],
    dropout: float,
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_mode: str,
    device: str,
) -> dict[str, object]:
    set_seed(DEFAULT_SEED)
    model = AdultMLP(input_dim=x_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, weight_mode, device))
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    start = time.time()
    for _ in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
    metrics = evaluate_torch_model(model, x_test, y_test, device=device)
    return {
        "seed": DEFAULT_SEED,
        "model_name": "mlp_tuned_non_dp",
        "is_dp": False,
        "hidden_dims": "-".join(str(dim) for dim in hidden_dims),
        "dropout": dropout,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "class_weight": weight_mode,
        "epsilon": None,
        **metrics,
        "training_time": time.time() - start,
        "device": device,
        "status": "ok",
    }


def main() -> None:
    ensure_dir(RESULTS_DIR)
    preprocess_adult(save_dir=PROCESSED_DATA_DIR)
    x_train, x_test, y_train, y_test, _ = load_processed_data(PROCESSED_DATA_DIR)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Baseline tuning is intended to run on the local T1200 GPU.")
    device = "cuda"
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    candidates = []
    for hidden_dims in [(128, 64), (256, 128), (256, 128, 64)]:
        for dropout in [0.0, 0.1, 0.2]:
            for weight_mode in ["none", "mild", "balanced"]:
                candidates.append(
                    {
                        "hidden_dims": hidden_dims,
                        "dropout": dropout,
                        "optimizer_name": "adam",
                        "learning_rate": 0.001,
                        "batch_size": 256,
                        "epochs": 40,
                        "weight_mode": weight_mode,
                    }
                )
    candidates.extend(
        [
            {
                "hidden_dims": (128, 64),
                "dropout": 0.1,
                "optimizer_name": "sgd",
                "learning_rate": lr,
                "batch_size": 256,
                "epochs": 40,
                "weight_mode": weight_mode,
            }
            for lr in [0.03, 0.05, 0.08]
            for weight_mode in ["none", "mild", "balanced"]
        ]
    )

    rows = []
    for candidate in candidates:
        print("Candidate:", candidate, flush=True)
        row = train_candidate(x_train, y_train, x_test, y_test, device=device, **candidate)
        rows.append(row)
        print(
            f"Result f1={row['f1_score']:.4f} acc={row['accuracy']:.4f} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(rows)
    best = df.sort_values("f1_score", ascending=False).iloc[0].to_dict()
    BEST_JSON.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    print("Saved:", OUTPUT_CSV, flush=True)
    print("Saved:", BEST_JSON, flush=True)
    print("Best:", best, flush=True)


if __name__ == "__main__":
    main()
