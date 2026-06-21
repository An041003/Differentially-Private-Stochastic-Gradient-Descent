from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_proba_torch(
    model: torch.nn.Module,
    x: np.ndarray,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            logits = model(batch)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, object]:
    positive_proba = y_proba[:, 1]
    result: dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        result["roc_auc"] = roc_auc_score(y_true, positive_proba)
    except ValueError:
        result["roc_auc"] = None
    try:
        result["pr_auc"] = average_precision_score(y_true, positive_proba)
    except ValueError:
        result["pr_auc"] = None
    return result


def evaluate_torch_model(
    model: torch.nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict[str, object]:
    y_proba = predict_proba_torch(model, x_test, device=device)
    y_pred = np.argmax(y_proba, axis=1)
    return evaluate_predictions(y_test, y_pred, y_proba)
