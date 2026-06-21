from __future__ import annotations

from collections.abc import Sequence

import torch
from sklearn.linear_model import LogisticRegression
from torch import nn


class AdultMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.1,
) -> AdultMLP:
    return AdultMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


def build_logistic_regression(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=seed)
