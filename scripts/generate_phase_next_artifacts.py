from __future__ import annotations

import json
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]


NOISE_RESULTS = [
    ("Baseline", None, None, 0.8501, 0.6567, None),
    ("DP-SGD", 0.5, 15.5548, 0.8491, 0.6683, "Accuracy cao nhat nhung privacy yeu"),
    ("DP-SGD", 0.8, 3.5038, 0.8485, 0.6523, "Privacy tot hon"),
    ("DP-SGD", 1.0, 2.1115, 0.8479, 0.6572, "Can bang kha tot"),
    ("DP-SGD", 1.5, 1.0946, 0.8464, 0.6544, "Cau hinh can bang nen uu tien"),
    ("DP-SGD", 2.0, 0.7501, 0.8460, 0.6446, "Privacy manh hon"),
    ("DP-SGD", 3.0, 0.4637, 0.8425, 0.6518, "Privacy manh nhat"),
]

MAX_GRAD_NORM_RESULTS = [
    (0.5, 1.2143, 0.851859, 0.760000, 0.580270, 0.658084, 84.809753, "Accuracy tốt, F1 thấp hơn norm 1.5"),
    (1.0, 1.2143, 0.851926, 0.756098, 0.586486, 0.660578, 82.720123, "Mốc hiện tại, kết quả ổn định"),
    (1.5, 1.2143, 0.852258, 0.750424, 0.597297, 0.665162, 80.983586, "F1-score cao nhất trong sweep"),
    (2.0, 1.2143, 0.850199, 0.748281, 0.588108, 0.658596, 80.674731, "Training nhanh hơn nhưng utility giảm nhẹ"),
]


def dedent(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def nb_cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": dedent(source).splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def create_sweep_notebook() -> None:
    cells = [
        nb_cell(
            "markdown",
            """
            # DP-SGD max_grad_norm sweep on UCI Adult

            Notebook này bổ sung thí nghiệm nhỏ cho phase tiếp theo: giữ cố định `noise_multiplier = 1.5` và quét `max_grad_norm = [0.5, 1.0, 1.5, 2.0]`.

            Mục tiêu không phải tăng accuracy bằng mọi giá, mà là quan sát clipping norm ảnh hưởng thế nào tới privacy-utility tradeoff của DP-SGD.
            """,
        ),
        nb_cell(
            "code",
            """
            # Run this cell if Opacus is not installed in the current notebook runtime.
            # In Colab, torch is usually preinstalled; Opacus is the only extra library needed.
            import importlib.util
            import sys
            import subprocess

            if importlib.util.find_spec("opacus") is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opacus"])
            """,
        ),
        nb_cell(
            "code",
            """
            from collections.abc import Sequence
            from pathlib import Path
            import random
            import time

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from opacus import PrivacyEngine
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            DATA_DIR = Path("adult")
            TRAIN_PATH = DATA_DIR / "adult.data"
            TEST_PATH = DATA_DIR / "adult.test"

            BATCH_SIZE = 256
            EPOCHS = 20
            LR_DP = 0.05
            MOMENTUM = 0.9
            DELTA = 1e-5
            NOISE_MULTIPLIER = 1.5
            MAX_GRAD_NORM_LIST = [0.5, 1.0, 1.5, 2.0]
            RANDOM_SEED = 42

            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)

            print("Device:", DEVICE)
            print("Train path:", TRAIN_PATH)
            print("Test path:", TEST_PATH)
            """,
        ),
        nb_cell(
            "markdown",
            """
            ## Load and preprocess data

            Notebook dùng trực tiếp thư mục `adult/` trong repo. Test file của Adult có dòng metadata ở đầu và nhãn kết thúc bằng dấu chấm, nên loader xử lý cả hai chi tiết này.
            """,
        ),
        nb_cell(
            "code",
            """
            columns = [
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week",
                "native_country", "income",
            ]

            numeric_cols = [
                "age", "fnlwgt", "education_num", "capital_gain",
                "capital_loss", "hours_per_week",
            ]


            def load_adult_csv(path: Path) -> pd.DataFrame:
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing {path}. Put the UCI Adult files under an adult/ folder."
                    )
                df = pd.read_csv(
                    path,
                    names=columns,
                    skipinitialspace=True,
                    na_values="?",
                    comment="|",
                )
                df["income"] = df["income"].str.replace(".", "", regex=False)
                return df.dropna().reset_index(drop=True)


            train_df = load_adult_csv(TRAIN_PATH)
            test_df = load_adult_csv(TEST_PATH)

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

            x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
            x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

            test_loader = DataLoader(
                TensorDataset(x_test_tensor, y_test_tensor),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            print("Train shape:", train_df.shape)
            print("Test shape:", test_df.shape)
            print("Input features:", x_train_tensor.shape[1])
            """,
        ),
        nb_cell(
            "code",
            """
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
                tp = 0
                fp = 0
                fn = 0

                with torch.no_grad():
                    for features, labels in loader:
                        features = features.to(DEVICE)
                        labels = labels.to(DEVICE)
                        logits = model(features)
                        preds = (torch.sigmoid(logits) >= 0.5).float()

                        correct += (preds == labels).sum().item()
                        total += labels.numel()
                        tp += ((preds == 1) & (labels == 1)).sum().item()
                        fp += ((preds == 1) & (labels == 0)).sum().item()
                        fn += ((preds == 0) & (labels == 1)).sum().item()

                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                return {
                    "accuracy": correct / total,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }


            def train_private(max_grad_norm: float) -> dict[str, float]:
                torch.manual_seed(RANDOM_SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(RANDOM_SEED)

                model = MLP(input_dim=x_train_tensor.shape[1]).to(DEVICE)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=LR_DP,
                    momentum=MOMENTUM,
                )
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

                start_time = time.time()
                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    for features, labels in private_train_loader:
                        features = features.to(DEVICE)
                        labels = labels.to(DEVICE)
                        optimizer.zero_grad()
                        loss = criterion(model(features), labels)
                        loss.backward()
                        optimizer.step()

                    metrics = evaluate(model, test_loader)
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    print(
                        f"norm={max_grad_norm:.1f} epoch={epoch:02d} "
                        f"epsilon={epsilon:.4f} acc={metrics['accuracy']:.4f} "
                        f"f1={metrics['f1_score']:.4f}"
                    )

                elapsed = time.time() - start_time
                final_metrics = evaluate(model, test_loader)
                return {
                    "max_grad_norm": max_grad_norm,
                    "epsilon": privacy_engine.get_epsilon(DELTA),
                    **final_metrics,
                    "training_time": elapsed,
                }
            """,
        ),
        nb_cell(
            "code",
            """
            sweep_results = []

            for max_grad_norm in MAX_GRAD_NORM_LIST:
                print("\\n" + "=" * 72)
                print(f"Training DP-SGD with max_grad_norm={max_grad_norm}")
                print("=" * 72)
                sweep_results.append(train_private(max_grad_norm))

            df_sweep = pd.DataFrame(sweep_results)
            df_sweep.to_csv("uci_adult_dp_sgd_max_grad_norm_sweep.csv", index=False)
            display(df_sweep)
            print("Saved: uci_adult_dp_sgd_max_grad_norm_sweep.csv")
            """,
        ),
        nb_cell(
            "code",
            """
            plt.figure(figsize=(7, 4))
            plt.plot(df_sweep["max_grad_norm"], df_sweep["accuracy"], marker="o")
            plt.xlabel("max_grad_norm")
            plt.ylabel("Accuracy")
            plt.title("Effect of Clipping Norm on Accuracy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("max_grad_norm_vs_accuracy.png", dpi=200)
            plt.show()

            plt.figure(figsize=(7, 4))
            plt.plot(df_sweep["max_grad_norm"], df_sweep["f1_score"], marker="o", color="tab:green")
            plt.xlabel("max_grad_norm")
            plt.ylabel("F1-score")
            plt.title("Effect of Clipping Norm on F1-score")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("max_grad_norm_vs_f1_score.png", dpi=200)
            plt.show()

            plt.figure(figsize=(7, 4))
            plt.plot(df_sweep["max_grad_norm"], df_sweep["training_time"], marker="o", color="tab:orange")
            plt.xlabel("max_grad_norm")
            plt.ylabel("Training time (seconds)")
            plt.title("Effect of Clipping Norm on Training Time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("max_grad_norm_vs_training_time.png", dpi=200)
            plt.show()
            """,
        ),
        nb_cell(
            "code",
            """
            def comment_for_norm(row: pd.Series) -> str:
                norm = row["max_grad_norm"]
                if norm == 0.5:
                    return "Có thể clip quá mạnh, cần xem accuracy/F1 có giảm không."
                if norm == 1.0:
                    return "Mốc hiện tại, dùng làm baseline của clipping norm."
                if norm == 1.5:
                    return "Có thể giữ nhiều tín hiệu gradient hơn."
                return "Có thể tăng ảnh hưởng gradient trước khi thêm noise."


            report_table = df_sweep.copy()
            report_table["Nhận xét"] = report_table.apply(comment_for_norm, axis=1)
            report_table = report_table.rename(
                columns={
                    "max_grad_norm": "Max grad norm",
                    "epsilon": "Epsilon",
                    "accuracy": "Accuracy",
                    "f1_score": "F1-score",
                    "training_time": "Training time",
                }
            )
            display(report_table)
            """,
        ),
    ]
    write_notebook(ROOT / "uci_adult_dp_sgd_max_grad_norm_sweep.ipynb", cells)


def create_demo_notebook() -> None:
    cells = [
        nb_cell(
            "markdown",
            """
            # DP-SGD UCI Adult demo

            Demo gọn cho thuyết trình: load dữ liệu Adult, train baseline, train DP-SGD với `noise_multiplier = 1.5`, in bảng so sánh và vẽ các biểu đồ tradeoff từ kết quả đã báo cáo.
            """,
        ),
        nb_cell(
            "code",
            """
            import importlib.util
            import sys
            import subprocess

            if importlib.util.find_spec("opacus") is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opacus"])
            """,
        ),
        nb_cell(
            "code",
            """
            from collections.abc import Sequence
            from pathlib import Path
            import random
            import time

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from opacus import PrivacyEngine
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            DATA_DIR = Path("adult")
            BATCH_SIZE = 256
            EPOCHS = 20
            BASELINE_LR = 1e-3
            DP_LR = 0.05
            DELTA = 1e-5
            NOISE_MULTIPLIER = 1.5
            MAX_GRAD_NORM = 1.0
            RANDOM_SEED = 42

            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            """,
        ),
        nb_cell(
            "code",
            """
            columns = [
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
            ]
            numeric_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


            def load_adult_csv(path: Path) -> pd.DataFrame:
                df = pd.read_csv(path, names=columns, skipinitialspace=True, na_values="?", comment="|")
                df["income"] = df["income"].str.replace(".", "", regex=False)
                return df.dropna().reset_index(drop=True)


            train_df = load_adult_csv(DATA_DIR / "adult.data")
            test_df = load_adult_csv(DATA_DIR / "adult.test")

            x_train_raw = train_df.drop(columns="income")
            x_test_raw = test_df.drop(columns="income")
            y_train = (train_df["income"] == ">50K").astype("float32")
            y_test = (test_df["income"] == ">50K").astype("float32")

            combined = pd.concat([x_train_raw, x_test_raw], axis=0)
            combined = pd.get_dummies(
                combined,
                columns=[col for col in combined.columns if col not in numeric_cols],
                dtype="float32",
            )
            x_train = combined.iloc[: len(train_df)].copy()
            x_test = combined.iloc[len(train_df) :].copy()

            means = x_train[numeric_cols].mean()
            stds = x_train[numeric_cols].std().replace(0, 1)
            x_train[numeric_cols] = (x_train[numeric_cols] - means) / stds
            x_test[numeric_cols] = (x_test[numeric_cols] - means) / stds

            x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
            x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

            train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
            print("Input features:", x_train_tensor.shape[1])
            """,
        ),
        nb_cell(
            "code",
            """
            class MLP(nn.Module):
                def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64, 32)) -> None:
                    super().__init__()
                    layers: list[nn.Module] = []
                    previous_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
                        previous_dim = hidden_dim
                    layers.append(nn.Linear(previous_dim, 1))
                    self.network = nn.Sequential(*layers)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.network(x)


            criterion = nn.BCEWithLogitsLoss()


            def evaluate(model: nn.Module) -> dict[str, float]:
                model.eval()
                correct = total = tp = fp = fn = 0
                with torch.no_grad():
                    for features, labels in test_loader:
                        features = features.to(DEVICE)
                        labels = labels.to(DEVICE)
                        preds = (torch.sigmoid(model(features)) >= 0.5).float()
                        correct += (preds == labels).sum().item()
                        total += labels.numel()
                        tp += ((preds == 1) & (labels == 1)).sum().item()
                        fp += ((preds == 1) & (labels == 0)).sum().item()
                        fn += ((preds == 0) & (labels == 1)).sum().item()
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                return {"accuracy": correct / total, "precision": precision, "recall": recall, "f1_score": f1}
            """,
        ),
        nb_cell(
            "code",
            """
            def train_baseline() -> dict[str, float]:
                torch.manual_seed(RANDOM_SEED)
                model = MLP(x_train_tensor.shape[1]).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE_LR)
                start = time.time()
                for epoch in range(EPOCHS):
                    model.train()
                    for features, labels in train_loader:
                        features = features.to(DEVICE)
                        labels = labels.to(DEVICE)
                        optimizer.zero_grad()
                        loss = criterion(model(features), labels)
                        loss.backward()
                        optimizer.step()
                return {**evaluate(model), "training_time": time.time() - start}


            baseline_metrics = train_baseline()
            print("Baseline accuracy:", round(baseline_metrics["accuracy"], 4))
            print("Baseline F1-score:", round(baseline_metrics["f1_score"], 4))
            """,
        ),
        nb_cell(
            "code",
            """
            def train_dp_sgd() -> dict[str, float]:
                torch.manual_seed(RANDOM_SEED)
                model = MLP(x_train_tensor.shape[1]).to(DEVICE)
                optimizer = torch.optim.SGD(model.parameters(), lr=DP_LR, momentum=0.9)
                dp_train_loader = DataLoader(
                    TensorDataset(x_train_tensor, y_train_tensor),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )
                privacy_engine = PrivacyEngine(accountant="prv")
                model, optimizer, private_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=dp_train_loader,
                    noise_multiplier=NOISE_MULTIPLIER,
                    max_grad_norm=MAX_GRAD_NORM,
                )
                start = time.time()
                for epoch in range(EPOCHS):
                    model.train()
                    for features, labels in private_loader:
                        features = features.to(DEVICE)
                        labels = labels.to(DEVICE)
                        optimizer.zero_grad()
                        loss = criterion(model(features), labels)
                        loss.backward()
                        optimizer.step()
                return {
                    **evaluate(model),
                    "epsilon": privacy_engine.get_epsilon(DELTA),
                    "training_time": time.time() - start,
                }


            dp_metrics = train_dp_sgd()
            print("DP-SGD epsilon:", round(dp_metrics["epsilon"], 4))
            print("DP-SGD accuracy:", round(dp_metrics["accuracy"], 4))
            print("DP-SGD F1-score:", round(dp_metrics["f1_score"], 4))
            print("Accuracy drop:", round(baseline_metrics["accuracy"] - dp_metrics["accuracy"], 4))
            """,
        ),
        nb_cell(
            "code",
            """
            comparison = pd.DataFrame(
                [
                    {"method": "Baseline", "epsilon": None, **baseline_metrics},
                    {"method": "DP-SGD noise=1.5", **dp_metrics},
                ]
            )
            display(comparison)
            """,
        ),
        nb_cell(
            "code",
            """
            reported_noise_results = pd.DataFrame(
                [
                    {"noise_multiplier": 0.5, "epsilon": 15.5548, "accuracy": 0.8491},
                    {"noise_multiplier": 0.8, "epsilon": 3.5038, "accuracy": 0.8485},
                    {"noise_multiplier": 1.0, "epsilon": 2.1115, "accuracy": 0.8479},
                    {"noise_multiplier": 1.5, "epsilon": 1.0946, "accuracy": 0.8464},
                    {"noise_multiplier": 2.0, "epsilon": 0.7501, "accuracy": 0.8460},
                    {"noise_multiplier": 3.0, "epsilon": 0.4637, "accuracy": 0.8425},
                ]
            )

            plt.figure(figsize=(7, 4))
            plt.plot(reported_noise_results["epsilon"], reported_noise_results["accuracy"], marker="o")
            plt.xlabel("Epsilon")
            plt.ylabel("Accuracy")
            plt.title("Privacy Utility Tradeoff")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 4))
            plt.plot(reported_noise_results["noise_multiplier"], reported_noise_results["epsilon"], marker="o")
            plt.xlabel("Noise multiplier")
            plt.ylabel("Epsilon")
            plt.title("Noise Multiplier vs Epsilon")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 4))
            plt.plot(reported_noise_results["noise_multiplier"], reported_noise_results["accuracy"], marker="o")
            plt.xlabel("Noise multiplier")
            plt.ylabel("Accuracy")
            plt.title("Noise Multiplier vs Accuracy")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            """,
        ),
    ]
    write_notebook(ROOT / "dp_sgd_uci_adult_demo.ipynb", cells)


def docx_paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    lines = text.splitlines() or [""]
    runs = []
    for i, line in enumerate(lines):
        if i:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(f"<w:r><w:t>{escape(line)}</w:t></w:r>")
    return f"<w:p>{style_xml}{''.join(runs)}</w:p>"


def docx_bullet(text: str) -> str:
    return docx_paragraph("• " + text)


def docx_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object, bold: bool = False) -> str:
        value_text = "" if value is None else str(value)
        bold_xml = "<w:rPr><w:b/></w:rPr>" if bold else ""
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:r>{bold_xml}<w:t>{escape(value_text)}</w:t></w:r></w:p></w:tc>"
        )

    header_xml = "<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>"
    row_xml = "".join("<w:tr>" + "".join(cell(v) for v in row) + "</w:tr>" for row in rows)
    return "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>" + header_xml + row_xml + "</w:tbl>"


def write_docx(path: Path, body_parts: list[str]) -> None:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body_parts)}
    <w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
  </w:body>
</w:document>"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
</w:styles>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)


def create_report_docx() -> None:
    body: list[str] = []
    body.append(docx_paragraph("Privacy-Preserving Data with Differential Privacy: DP-SGD on UCI Adult Dataset", "Title"))
    body.append(docx_paragraph("Báo cáo cập nhật cho phase tiếp theo", "Heading1"))
    body.append(docx_paragraph("Chủ đề này minh họa cách tích hợp Differential Privacy vào quá trình huấn luyện mô hình, thay vì xem privacy là bước xử lý sau cùng. Dữ liệu UCI Adult chứa nhiều thuộc tính cá nhân như tuổi, học vấn, nghề nghiệp, giới tính và số giờ làm việc, nên phù hợp với pillar Privacy in Development."))

    body.append(docx_paragraph("1. Vấn đề nghiên cứu", "Heading1"))
    body.append(docx_paragraph("Mô hình học máy có thể vô tình ghi nhớ một số mẫu huấn luyện cụ thể. Khi dữ liệu chứa thông tin cá nhân hoặc nhạy cảm, việc ghi nhớ này có thể dẫn tới rò rỉ thông tin qua tham số mô hình hoặc qua đầu ra dự đoán."))
    body.append(docx_paragraph("DP-SGD được dùng như một cơ chế bảo vệ privacy trong training pipeline: tính gradient theo từng mẫu, clip gradient theo chuẩn L2, thêm Gaussian noise, cập nhật mô hình và theo dõi privacy budget bằng accountant."))

    body.append(docx_paragraph("2. Threat Model", "Heading1"))
    body.append(docx_paragraph("Giả định kẻ tấn công có thể truy cập mô hình đã huấn luyện hoặc truy vấn mô hình để quan sát đầu ra dự đoán. Từ đó, kẻ tấn công có thể cố suy luận liệu một cá nhân cụ thể có xuất hiện trong tập huấn luyện hay không, hoặc khai thác thông tin nhạy cảm mà mô hình đã ghi nhớ."))
    body.append(docx_paragraph("Mục tiêu của DP-SGD là giới hạn đóng góp của từng cá nhân vào mô hình. Khi thêm hoặc loại bỏ một bản ghi khỏi tập huấn luyện, phân phối đầu ra của thuật toán huấn luyện không thay đổi đáng kể theo định nghĩa Differential Privacy."))
    body.append(docx_paragraph("Trong phạm vi thực nghiệm này, nhóm chưa triển khai membership inference attack trực tiếp. Vì vậy, epsilon được dùng để đánh giá định lượng privacy guarantee của quá trình huấn luyện, nhưng không nên khẳng định mô hình chống được mọi kiểu tấn công privacy."))

    body.append(docx_paragraph("3. So sánh với Abadi et al.", "Heading1"))
    body.append(docx_table(
        ["Tiêu chí", "Abadi et al.", "Thực nghiệm của nhóm"],
        [
            ["Dataset", "MNIST, CIFAR-10", "UCI Adult"],
            ["Kiểu dữ liệu", "Ảnh", "Dữ liệu bảng cá nhân"],
            ["Mô hình", "Neural network/CNN", "MLP"],
            ["Cơ chế riêng tư", "DP-SGD", "DP-SGD qua Opacus"],
            ["Privacy accounting", "Moments Accountant", "Opacus PrivacyEngine"],
            ["Trục phân tích", "Epsilon, delta, accuracy", "Epsilon, accuracy, F1-score, training time"],
            ["Mục tiêu", "Benchmark deep learning có DP", "Minh họa Privacy in Development với dữ liệu cá nhân"],
        ],
    ))

    body.append(docx_paragraph("4. Thiết kế thực nghiệm", "Heading1"))
    for item in [
        "Baseline: MLP không áp dụng Differential Privacy.",
        "DP-SGD: MLP huấn luyện bằng Opacus với per-example gradient, clipping và Gaussian noise.",
        "Noise multiplier sweep: 0.5, 0.8, 1.0, 1.5, 2.0, 3.0.",
        "Max grad norm sweep mới: 0.5, 1.0, 1.5, 2.0 với noise_multiplier cố định bằng 1.5.",
        "Chỉ số: accuracy, precision, recall, F1-score, epsilon và training time.",
    ]:
        body.append(docx_bullet(item))

    body.append(docx_paragraph("5. Kết quả noise multiplier", "Heading1"))
    body.append(docx_table(
        ["Method", "Noise", "Epsilon", "Accuracy", "F1-score", "Ghi chú"],
        [[m, "" if n is None else n, "" if e is None else e, a, f, "" if note is None else note] for m, n, e, a, f, note in NOISE_RESULTS],
    ))
    body.append(docx_paragraph("Kết quả cho thấy khi noise multiplier tăng, epsilon giảm rõ rệt, nghĩa là privacy mạnh hơn. Accuracy giảm nhẹ so với baseline, thể hiện đúng privacy-utility tradeoff."))

    body.append(docx_paragraph("6. Cấu hình cân bằng được chọn", "Heading1"))
    body.append(docx_paragraph("Cấu hình được ưu tiên để thảo luận chính là noise_multiplier = 1.5, epsilon = 1.0946, accuracy = 0.8464 và F1-score = 0.6544. Accuracy chỉ giảm khoảng 0.0036 so với baseline 0.8501, trong khi epsilon thấp hơn đáng kể so với noise 0.5 và 1.0."))

    body.append(docx_paragraph("7. Thí nghiệm max_grad_norm", "Heading1"))
    body.append(docx_paragraph("Thí nghiệm đã được chạy thật bằng .venv trên CPU với PyTorch 2.12.0+cpu và Opacus 1.6.0. Cấu hình cố định gồm noise_multiplier = 1.5, batch_size = 256, epochs = 20, learning_rate = 0.05, momentum = 0.9 và delta = 1e-5."))
    body.append(docx_table(
        ["Max grad norm", "Epsilon", "Accuracy", "Precision", "Recall", "F1-score", "Training time", "Nhận xét"],
        [
            [norm, eps, f"{acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{seconds:.2f}s", note]
            for norm, eps, acc, precision, recall, f1, seconds, note in MAX_GRAD_NORM_RESULTS
        ],
    ))
    body.append(docx_paragraph("Vì noise_multiplier, sample rate, số epoch và delta được giữ cố định, epsilon cuối cùng giống nhau giữa các giá trị max_grad_norm. Điều này cho thấy sweep này chủ yếu phân tích utility, còn privacy budget không đổi theo clipping norm trong cấu hình Opacus đã dùng. Trong lần chạy này, max_grad_norm = 1.5 đạt F1-score cao nhất 0.6652 và accuracy cao nhất 0.8523, nên có thể xem là lựa chọn tốt hơn mốc 1.0 cho thí nghiệm bổ sung. Tuy vậy, chênh lệch giữa các cấu hình nhỏ, nên kết luận nên được trình bày thận trọng: MLP nhỏ trên UCI Adult khá ổn định trong phạm vi clipping norm được thử."))

    body.append(docx_paragraph("8. Hạn chế và hướng phát triển", "Heading1"))
    for item in [
        "Chưa triển khai membership inference attack trực tiếp.",
        "Mới dùng MLP nhỏ trên dữ liệu bảng.",
        "Chưa sweep đầy đủ batch size, learning rate và target epsilon.",
        "Có thể mở rộng bằng so sánh Logistic Regression, MLP lớn hơn hoặc chạy MNIST để đối chiếu trực tiếp với paper gốc.",
    ]:
        body.append(docx_bullet(item))

    body.append(docx_paragraph("9. Kết luận", "Heading1"))
    body.append(docx_paragraph("Thực nghiệm cho thấy DP-SGD có thể được tích hợp vào pipeline huấn luyện mô hình học máy nhằm bảo vệ quyền riêng tư của dữ liệu huấn luyện. Trên UCI Adult Dataset, khi tăng noise multiplier, privacy budget epsilon giảm rõ rệt trong khi accuracy chỉ giảm nhẹ so với baseline. Cấu hình noise_multiplier = 1.5 đạt epsilon 1.0946 với accuracy 0.8464, thể hiện một điểm cân bằng hợp lý giữa privacy và utility."))

    body.append(docx_paragraph("10. Câu hỏi phản biện ngắn", "Heading1"))
    qa = [
        ("Vì sao chọn UCI Adult thay vì MNIST?", "UCI Adult chứa thuộc tính cá nhân như tuổi, nghề nghiệp, học vấn, giới tính và thu nhập, nên phù hợp hơn với Privacy in Development."),
        ("Epsilon càng nhỏ nghĩa là gì?", "Đầu ra thuật toán ít thay đổi hơn khi thêm hoặc bỏ một cá nhân khỏi tập huấn luyện, nên privacy guarantee mạnh hơn."),
        ("DP-SGD có chống mọi tấn công privacy không?", "Không. DP-SGD cung cấp bảo đảm hình thức theo Differential Privacy, nhưng thí nghiệm này chưa kiểm tra trực tiếp membership inference hoặc model inversion."),
        ("Opacus có vai trò gì?", "Opacus hỗ trợ per-example gradient, clipping, thêm Gaussian noise và privacy accounting cho PyTorch."),
    ]
    for question, answer in qa:
        body.append(docx_paragraph(question, "Heading2"))
        body.append(docx_paragraph(answer))

    write_docx(ROOT / "bao_cao_dp_sgd_uci_adult_updated.docx", body)


def ppt_text_shape(shape_id: int, x: int, y: int, cx: int, cy: int, lines: list[str], font_size: int = 2600, bold_first: bool = False) -> str:
    paragraphs = []
    for idx, line in enumerate(lines):
        bold = "<a:b/>" if bold_first and idx == 0 else ""
        paragraphs.append(
            f"""<a:p><a:r><a:rPr lang="vi-VN" sz="{font_size}">{bold}</a:rPr><a:t>{escape(line)}</a:t></a:r><a:endParaRPr lang="vi-VN" sz="{font_size}"/></a:p>"""
        )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{''.join(paragraphs)}</p:txBody>
</p:sp>"""


def ppt_slide_xml(title: str, bullets: list[str], slide_no: int) -> str:
    title_shape = ppt_text_shape(2, 600000, 450000, 11000000, 900000, [title], 3400, True)
    bullet_lines = [f"• {bullet}" for bullet in bullets]
    body_shape = ppt_text_shape(3, 850000, 1500000, 11000000, 5000000, bullet_lines, 2300)
    footer = ppt_text_shape(4, 9700000, 6350000, 2200000, 400000, [f"{slide_no}"], 1400)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {title_shape}
      {body_shape}
      {footer}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def write_pptx(path: Path, slides: list[tuple[str, list[str]]]) -> None:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    for idx in range(1, len(slides) + 1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {' '.join(overrides)}
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""
    slide_id_list = []
    pres_rels = []
    for idx in range(1, len(slides) + 1):
        rel_id = f"rId{idx}"
        slide_id_list.append(f'<p:sldId id="{255 + idx}" r:id="{rel_id}"/>')
        pres_rels.append(f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx}.xml"/>')
    master_rel_id = f"rId{len(slides) + 1}"
    pres_rels.append(f'<Relationship Id="{master_rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="{master_rel_id}"/></p:sldMasterIdLst>
  <p:sldIdLst>{''.join(slide_id_list)}</p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""
    presentation_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(pres_rels)}
</Relationships>"""
    slide_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""
    master_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""
    master_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""
    layout_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" type="blank">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""
    layout_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""
    theme_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="DPSGD">
  <a:themeElements>
    <a:clrScheme name="DPSGD">
      <a:dk1><a:srgbClr val="0F172A"/></a:dk1><a:lt1><a:srgbClr val="F8FAFC"/></a:lt1>
      <a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="E2E8F0"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="059669"/></a:accent2>
      <a:accent3><a:srgbClr val="D97706"/></a:accent3><a:accent4><a:srgbClr val="DC2626"/></a:accent4>
      <a:accent5><a:srgbClr val="7C3AED"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="DPSGD"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="DPSGD"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
  </a:themeElements>
</a:theme>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("ppt/presentation.xml", presentation)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        zf.writestr("ppt/slideMasters/slideMaster1.xml", master_xml)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels)
        zf.writestr("ppt/theme/theme1.xml", theme_xml)
        for idx, (title, bullets) in enumerate(slides, 1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", ppt_slide_xml(title, bullets, idx))
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels)


def create_presentation() -> None:
    slides = [
        ("Privacy-Preserving Data with Differential Privacy", ["DP-SGD on UCI Adult Dataset", "Pillar: Privacy in Development", "Method: Opacus + PyTorch", "Source: Abadi et al."]),
        ("The Problem", ["Models may memorize training examples", "Adult data contains personal attributes", "Privacy should be integrated during training"]),
        ("Differential Privacy", ["Adding or removing one person should not change outputs much", "Smaller epsilon means stronger privacy", "Delta is a small failure probability"]),
        ("DP-SGD Workflow", ["Per-example gradient", "L2 gradient clipping", "Gaussian noise", "Model update", "Privacy accounting"]),
        ("Source Paper", ["Abadi et al. proposed DP-SGD for deep learning", "Experiments on MNIST and CIFAR-10", "Focus: privacy accountant and privacy-utility tradeoff"]),
        ("Dataset", ["UCI Adult Dataset", "48,842 records", "14 input attributes", "Predict income >50K or <=50K", "Contains personal tabular attributes"]),
        ("Experimental Design", ["Baseline: non-private MLP", "DP-SGD: MLP with Opacus", "Noise multiplier: 0.5 to 3.0", "Delta: 1e-5", "Batch size: 256, epochs: 20"]),
        ("Main Results", ["Baseline accuracy: 0.8501, F1: 0.6567", "Noise 1.5: epsilon 1.0946", "Noise 1.5 accuracy: 0.8464", "Accuracy drop: 0.0036"]),
        ("Privacy Utility Tradeoff", ["Noise increases, epsilon usually decreases", "Smaller epsilon means stronger privacy", "Accuracy decreases slightly on UCI Adult"]),
        ("Max Grad Norm Sweep", ["Fixed noise multiplier = 1.5", "All settings end at epsilon = 1.2143", "Best F1: norm 1.5, F1 = 0.6652", "Differences are small, so present cautiously"]),
        ("Recommended Configuration", ["Choose noise multiplier 1.5", "For clipping sweep, norm 1.5 performed best", "Accuracy = 0.8523 in sweep", "F1-score = 0.6652 in sweep", "Balanced privacy and utility"]),
        ("Limitations", ["No direct membership inference attack yet", "Only a small MLP", "No full sweep for batch size and learning rate", "Tabular task is simpler than image benchmarks"]),
        ("Conclusion", ["DP-SGD can be integrated into training", "Opacus makes PyTorch DP-SGD practical", "UCI Adult shows low accuracy drop with meaningful privacy", "Good example of Privacy in Development"]),
    ]
    write_pptx(ROOT / "dp_sgd_uci_adult_presentation.pptx", slides)


def main() -> None:
    create_sweep_notebook()
    create_demo_notebook()
    create_report_docx()
    create_presentation()
    print("Generated phase-next artifacts:")
    for name in [
        "uci_adult_dp_sgd_max_grad_norm_sweep.ipynb",
        "dp_sgd_uci_adult_demo.ipynb",
        "bao_cao_dp_sgd_uci_adult_updated.docx",
        "dp_sgd_uci_adult_presentation.pptx",
    ]:
        print(" -", ROOT / name)


if __name__ == "__main__":
    main()
