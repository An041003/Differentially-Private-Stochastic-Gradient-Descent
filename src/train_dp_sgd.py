from __future__ import annotations

import time

import numpy as np
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DATASET_NAME
from .evaluate import evaluate_torch_model
from .models import build_mlp
from .utils import generate_experiment_id, set_seed


def train_dp_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.05,
    momentum: float = 0.9,
    noise_multiplier: float = 1.5,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    device: str = "cpu",
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.1,
    model_variant: str = "mlp_128_64_dropout_0p1",
) -> tuple[nn.Module, dict[str, object]]:
    set_seed(seed)
    model = build_mlp(input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    privacy_engine = PrivacyEngine(accountant="prv")
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    start = time.time()
    for _ in range(epochs):
        model.train()
        for features, labels in private_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start

    epsilon = privacy_engine.get_epsilon(delta)
    metrics = evaluate_torch_model(model, x_test, y_test, device=device)
    result = {
        "experiment_id": generate_experiment_id(
            "dp_mlp",
            {
                "seed": seed,
                "noise": noise_multiplier,
                "clip": max_grad_norm,
                "bs": batch_size,
            },
        ),
        "seed": seed,
        "model_name": "dp_sgd_mlp",
        "model_variant": model_variant,
        "is_dp": True,
        "dataset": DATASET_NAME,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "input_dim": input_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "sgd",
        "momentum": momentum,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "hidden_dims": "-".join(str(dim) for dim in hidden_dims),
        "dropout": dropout,
        "delta": delta,
        "epsilon": epsilon,
        **metrics,
        "training_time": training_time,
        "notes": "DP-SGD with Opacus PrivacyEngine",
    }
    return model, result
