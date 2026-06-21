from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .evaluate import predict_proba_torch


def get_confidence_scores(model: torch.nn.Module, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    proba = predict_proba_torch(model, x, device=device)
    return np.max(proba, axis=1)


def build_mia_dataset(
    train_confidence: np.ndarray,
    test_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.concatenate([train_confidence, test_confidence])
    membership_label = np.concatenate(
        [np.ones(len(train_confidence)), np.zeros(len(test_confidence))]
    )
    return confidence, membership_label


def compute_attack_auc(train_confidence: np.ndarray, test_confidence: np.ndarray) -> float:
    confidence, membership_label = build_mia_dataset(train_confidence, test_confidence)
    return float(roc_auc_score(membership_label, confidence))


def run_confidence_mia(
    model: torch.nn.Module,
    x_train: np.ndarray,
    x_test: np.ndarray,
    device: str = "cpu",
    model_name: str = "model",
    is_dp: bool = False,
    noise_multiplier: float | None = None,
    epsilon: float | None = None,
) -> dict[str, object]:
    train_confidence = get_confidence_scores(model, x_train, device=device)
    test_confidence = get_confidence_scores(model, x_test, device=device)
    return {
        "experiment_id": f"mia_{model_name}",
        "model_name": model_name,
        "is_dp": is_dp,
        "noise_multiplier": noise_multiplier,
        "epsilon": epsilon,
        "attack_type": "confidence_threshold_auc",
        "attack_auc": compute_attack_auc(train_confidence, test_confidence),
        "mean_train_confidence": float(np.mean(train_confidence)),
        "mean_test_confidence": float(np.mean(test_confidence)),
        "confidence_gap": float(np.mean(train_confidence) - np.mean(test_confidence)),
        "num_train_samples": len(x_train),
        "num_test_samples": len(x_test),
        "notes": "AUC of confidence score for train-vs-test membership",
    }
