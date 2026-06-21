from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DATASET_NAME
from .evaluate import evaluate_predictions, evaluate_torch_model
from .models import build_logistic_regression, build_mlp
from .utils import generate_experiment_id, set_seed


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    start = time.time()
    model = build_logistic_regression(seed)
    model.fit(x_train, y_train)
    training_time = time.time() - start
    y_proba = model.predict_proba(x_test)
    y_pred = model.predict(x_test)
    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    return {
        "experiment_id": generate_experiment_id("logreg", {"seed": seed}),
        "seed": seed,
        "model_name": "logistic_regression",
        "is_dp": False,
        "dataset": DATASET_NAME,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "input_dim": x_train.shape[1],
        "epochs": None,
        "batch_size": None,
        "learning_rate": None,
        "optimizer": "lbfgs",
        "momentum": None,
        "noise_multiplier": None,
        "max_grad_norm": None,
        "delta": None,
        "epsilon": None,
        **metrics,
        "training_time": training_time,
        "notes": "sklearn LogisticRegression baseline",
    }


def train_mlp_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    set_seed(seed)
    model = build_mlp(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
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
    training_time = time.time() - start

    metrics = evaluate_torch_model(model, x_test, y_test, device=device)
    result = {
        "experiment_id": generate_experiment_id("mlp_baseline", {"seed": seed}),
        "seed": seed,
        "model_name": "mlp_baseline",
        "is_dp": False,
        "dataset": DATASET_NAME,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "input_dim": input_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "adam",
        "momentum": None,
        "noise_multiplier": None,
        "max_grad_norm": None,
        "delta": None,
        "epsilon": None,
        **metrics,
        "training_time": training_time,
        "notes": "non-private PyTorch MLP baseline",
    }
    return model, result
