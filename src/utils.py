from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_experiment_id(prefix: str, params: dict[str, Any]) -> str:
    parts = [prefix]
    for key in sorted(params):
        value = params[key]
        if value is None:
            continue
        parts.append(f"{key}_{value}")
    return "_".join(str(part).replace(".", "p").replace(" ", "_") for part in parts)
