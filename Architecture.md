# ARCHITECTURE.md

# Final Project Architecture – DP-SGD Privacy-Preserving ML

## 1. Architecture Overview

This document describes the final architecture for the project:

**Privacy-Preserving Data with Differential Privacy using DP-SGD**

The architecture is designed for a 5-person research-style project. It supports:

- reproducible preprocessing
- baseline model training
- DP-SGD training with Opacus
- hyperparameter sweeps
- privacy accounting
- utility evaluation
- membership inference attack evaluation
- result logging
- figure generation
- report and slide production

The project follows a layered architecture:

```text
Raw Data
   ↓
Preprocessing Layer
   ↓
Processed Dataset
   ↓
Model Layer
   ├── Baseline Models
   └── DP-SGD Models
   ↓
Training Layer
   ├── Non-DP Training
   └── DP-SGD Training with PrivacyEngine
   ↓
Evaluation Layer
   ├── Utility Metrics
   ├── Privacy Budget
   └── Membership Inference Attack
   ↓
Experiment Management Layer
   ├── Result CSV Files
   ├── Figures
   └── Final Summary
   ↓
Deliverables
   ├── Report
   ├── Slides
   └── Demo Notebook
```

---

## 2. System Goals

The architecture must satisfy these goals:

1. **Reproducibility**
   - Same seed and same configuration should produce comparable results.
   - All result tables should be saved as CSV files.
   - Figures should be generated from saved results.

2. **Modularity**
   - Data preprocessing, models, training, evaluation, plotting, and attacks should be separated.
   - Notebooks should call reusable functions from `src/`.

3. **Experiment traceability**
   - Every experiment should have an `experiment_id`.
   - Hyperparameters and metrics should be stored in result files.
   - Reported numbers should come from saved CSV files.

4. **Privacy-specific evaluation**
   - DP-SGD runs must report epsilon and delta.
   - Membership inference attack must be evaluated separately from formal epsilon.

5. **Presentation readiness**
   - The final output must be easy to explain through diagrams, tables, and graphs.

---

## 3. Directory Architecture

Use this final directory structure.

```text
dp_sgd_privacy_project/
│
├── AGENT.md
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── adult.data
│   │   ├── adult.test
│   │   └── adult.names
│   │
│   ├── processed/
│   │   ├── X_train.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   ├── y_test.npy
│   │   ├── train_indices.npy
│   │   ├── test_indices.npy
│   │   ├── feature_names.json
│   │   └── preprocessing_metadata.json
│   │
│   └── data.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train_baseline.py
│   ├── train_dp_sgd.py
│   ├── evaluate.py
│   ├── privacy_attack.py
│   ├── plotting.py
│   └── utils.py
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_dp_sgd_noise_sweep.ipynb
│   ├── 04_dp_sgd_max_grad_norm_sweep.ipynb
│   ├── 05_dp_sgd_batch_size_sweep.ipynb
│   ├── 06_multi_seed_experiment.ipynb
│   ├── 07_membership_inference_attack.ipynb
│   └── 08_demo.ipynb
│
├── results/
│   ├── baseline_results.csv
│   ├── dp_sgd_noise_results.csv
│   ├── max_grad_norm_results.csv
│   ├── batch_size_results.csv
│   ├── multi_seed_results.csv
│   ├── mia_results.csv
│   └── final_summary.csv
│
├── figures/
│   ├── dp_sgd_workflow.png
│   ├── baseline_confusion_matrix.png
│   ├── privacy_utility_tradeoff.png
│   ├── noise_vs_epsilon.png
│   ├── noise_vs_accuracy.png
│   ├── noise_vs_f1.png
│   ├── max_grad_norm_vs_accuracy.png
│   ├── max_grad_norm_vs_f1.png
│   ├── batch_size_vs_epsilon.png
│   ├── batch_size_vs_accuracy.png
│   ├── multi_seed_errorbar_accuracy.png
│   ├── multi_seed_errorbar_f1.png
│   ├── confidence_distribution_baseline.png
│   ├── confidence_distribution_dp_sgd.png
│   └── attack_auc_comparison.png
│
├── docs/
│   ├── theory_summary.md
│   ├── paper_comparison.md
│   ├── threat_model.md
│   ├── experiment_plan.md
│   └── defense_questions.md
│
├── report/
│   ├── report.md
│   ├── report.docx
│   └── report.pdf
│
└── slides/
    └── dp_sgd_final_presentation.pptx
```

---

## 4. Data Architecture

## 4.1 Input Dataset

The project uses the UCI Adult Dataset.

Raw files:

```text
data/raw/adult.data
data/raw/adult.test
data/raw/adult.names
```

The prediction task is binary classification:

```text
income <=50K -> 0
income >50K  -> 1
```

## 4.2 Data Schema

Expected raw columns:

```python
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
```

Numerical columns:

```python
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
```

Categorical columns:

```python
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
```

Target column:

```python
TARGET_COLUMN = "income"
```

## 4.3 Preprocessing Pipeline

The preprocessing layer performs:

```text
1. Load adult.data and adult.test.
2. Assign column names.
3. Remove trailing periods from adult.test labels.
4. Strip whitespace from all string columns.
5. Replace "?" with missing values.
6. Drop missing rows or apply a consistent imputation strategy.
7. Convert target label to binary.
8. Split features and label.
9. One-hot encode categorical columns.
10. Standardize numerical columns.
11. Save processed arrays.
12. Save feature names and metadata.
```

Recommended approach:

- Use `adult.data` and `adult.test` as original split if desired.
- Or combine and use stratified 80/20 split for consistency with existing experiment.
- Whichever method is chosen, document it clearly.

## 4.4 Processed Output

Save processed files:

```text
data/processed/X_train.npy
data/processed/X_test.npy
data/processed/y_train.npy
data/processed/y_test.npy
data/processed/feature_names.json
data/processed/preprocessing_metadata.json
```

Metadata should include:

```json
{
  "dataset": "UCI Adult",
  "num_train_samples": 0,
  "num_test_samples": 0,
  "num_features": 0,
  "target_mapping": {
    "<=50K": 0,
    ">50K": 1
  },
  "missing_value_strategy": "drop_rows",
  "categorical_encoding": "one_hot",
  "numeric_scaling": "standard_scaler",
  "split_strategy": "stratified_80_20",
  "random_seed": 42
}
```

---

## 5. Source Code Architecture

## 5.1 `src/config.py`

Purpose:

Central place for default configuration.

Recommended contents:

```python
DEFAULT_SEED = 42
DELTA = 1e-5
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE_BASELINE = 1e-3
LEARNING_RATE_DP = 0.05
MOMENTUM = 0.9
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIERS = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
MAX_GRAD_NORM_VALUES = [0.5, 1.0, 1.5, 2.0]
BATCH_SIZE_VALUES = [64, 128, 256, 512]
SEEDS = [42, 123, 2026]
```

Also define paths:

```python
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
```

---

## 5.2 `src/utils.py`

Purpose:

Common utilities.

Functions:

```python
def set_seed(seed: int) -> None:
    pass

def ensure_dir(path: str) -> None:
    pass

def get_device() -> str:
    pass

def save_json(obj: dict, path: str) -> None:
    pass

def load_json(path: str) -> dict:
    pass

def generate_experiment_id(prefix: str, params: dict) -> str:
    pass
```

Expected behavior:

- `set_seed` sets Python, NumPy, and PyTorch seeds.
- `get_device` returns `cuda` if available, otherwise `cpu`.
- `generate_experiment_id` creates names such as:

```text
dp_noise_1.5_clip_1.0_bs_256_seed_42
```

---

## 5.3 `src/data_preprocessing.py`

Purpose:

Load and preprocess UCI Adult.

Functions:

```python
def load_adult_raw(raw_dir: str) -> pd.DataFrame:
    pass

def clean_adult_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    pass

def build_preprocessor() -> ColumnTransformer:
    pass

def preprocess_adult(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    save_dir: str = "data/processed"
) -> dict:
    pass

def load_processed_data(processed_dir: str = "data/processed") -> tuple:
    pass
```

Return format of `load_processed_data`:

```python
X_train, X_test, y_train, y_test, feature_names
```

Important implementation notes:

- Make sure labels are clean:
  - `>50K.`
  - `<=50K.`
  - `>50K`
  - `<=50K`
- Make sure the train and test feature dimensions match.
- Save feature names after one-hot encoding.

---

## 5.4 `src/models.py`

Purpose:

Define ML models.

Functions/classes:

```python
class AdultMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout=0.1):
        pass

    def forward(self, x):
        pass

def build_mlp(input_dim: int) -> AdultMLP:
    pass

def build_logistic_regression():
    pass
```

Recommended MLP architecture:

```text
Input dimension
↓
Linear(input_dim, 128)
ReLU
Dropout(0.1)
Linear(128, 64)
ReLU
Dropout(0.1)
Linear(64, 2)
```

Loss:

```python
CrossEntropyLoss
```

For binary classification, output two logits.

---

## 5.5 `src/train_baseline.py`

Purpose:

Train non-private baselines.

Functions:

```python
def train_logistic_regression(X_train, y_train, X_test, y_test, seed: int) -> dict:
    pass

def train_mlp_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    input_dim: int,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3
) -> tuple:
    pass
```

Return for `train_mlp_baseline`:

```python
model, result_dict
```

`result_dict` must contain:

```text
experiment_id
seed
model_name
is_dp
accuracy
precision
recall
f1_score
roc_auc
pr_auc
training_time
epsilon
delta
noise_multiplier
max_grad_norm
batch_size
epochs
```

For non-DP models:

```text
epsilon = None
delta = None
noise_multiplier = None
max_grad_norm = None
```

---

## 5.6 `src/train_dp_sgd.py`

Purpose:

Train MLP using DP-SGD with Opacus.

Functions:

```python
def train_dp_mlp(
    X_train,
    y_train,
    X_test,
    y_test,
    input_dim: int,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.05,
    momentum: float = 0.9,
    noise_multiplier: float = 1.5,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5
) -> tuple:
    pass
```

Return:

```python
model, result_dict
```

Important Opacus logic:

```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm,
)
```

At the end:

```python
epsilon = privacy_engine.get_epsilon(delta)
```

Important notes:

- Use `ModuleValidator` if Opacus complains about unsupported layers.
- Avoid BatchNorm in DP models.
- Use simple Linear/ReLU/Dropout layers.
- Do not use unsupported operations.
- Make sure labels are integer class labels for `CrossEntropyLoss`.

---

## 5.7 `src/evaluate.py`

Purpose:

Evaluate classification models.

Functions:

```python
def predict_proba_torch(model, X, device="cpu") -> np.ndarray:
    pass

def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    pass

def evaluate_torch_model(model, X_test, y_test, device="cpu") -> dict:
    pass

def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    pass
```

Required metrics:

```text
accuracy
precision
recall
f1_score
roc_auc
pr_auc
confusion_matrix
```

For binary classification:

- `y_proba_positive = probability of class 1`
- ROC-AUC uses positive class probability.
- PR-AUC uses positive class probability.

Use safe behavior:

- If a metric fails due to one-class issue, return `None` and log a warning.

---

## 5.8 `src/privacy_attack.py`

Purpose:

Run confidence-based membership inference attack.

Functions:

```python
def get_confidence_scores(model, X, device="cpu") -> np.ndarray:
    pass

def build_mia_dataset(
    train_confidence: np.ndarray,
    test_confidence: np.ndarray
) -> tuple:
    pass

def compute_attack_auc(
    train_confidence: np.ndarray,
    test_confidence: np.ndarray
) -> float:
    pass

def run_confidence_mia(
    model,
    X_train,
    X_test,
    device="cpu",
    model_name: str = "model"
) -> dict:
    pass
```

Attack data construction:

```python
confidence = np.concatenate([train_confidence, test_confidence])
membership_label = np.concatenate([
    np.ones(len(train_confidence)),
    np.zeros(len(test_confidence))
])
```

Metric:

```python
attack_auc = roc_auc_score(membership_label, confidence)
```

Result dictionary:

```text
model_name
attack_type
attack_auc
mean_train_confidence
mean_test_confidence
confidence_gap
num_train_samples
num_test_samples
```

Interpretation:

```text
attack_auc close to 0.5 -> lower membership leakage
attack_auc higher than 0.5 -> stronger membership signal
```

---

## 5.9 `src/plotting.py`

Purpose:

Generate all final figures.

Functions:

```python
def plot_noise_vs_epsilon(results_csv: str, output_path: str) -> None:
    pass

def plot_noise_vs_accuracy(results_csv: str, output_path: str) -> None:
    pass

def plot_privacy_utility_tradeoff(results_csv: str, output_path: str) -> None:
    pass

def plot_max_grad_norm_vs_metric(results_csv: str, metric: str, output_path: str) -> None:
    pass

def plot_batch_size_vs_metric(results_csv: str, metric: str, output_path: str) -> None:
    pass

def plot_confidence_distribution(
    train_confidence,
    test_confidence,
    output_path: str,
    title: str
) -> None:
    pass

def plot_attack_auc_comparison(mia_csv: str, output_path: str) -> None:
    pass
```

Plot standards:

- Save PNG files to `figures/`.
- Use clear titles and axis labels.
- Use consistent figure size.
- Do not rely on screenshots from notebooks.
- Figures must be directly reusable in report and slides.

---

## 6. Experiment Architecture

## 6.1 Experiment Runner Pattern

Each experiment notebook should follow this pattern:

```text
1. Import libraries and src functions.
2. Load processed data.
3. Define experiment grid.
4. For each configuration:
   4.1. Set seed.
   4.2. Train model.
   4.3. Evaluate model.
   4.4. Record result dict.
5. Convert results to DataFrame.
6. Save CSV to results/.
7. Generate figures into figures/.
8. Print summary table.
```

---

## 6.2 Baseline Experiment

Input:

```text
X_train, X_test, y_train, y_test
```

Models:

```text
Logistic Regression
MLP baseline
```

Output:

```text
results/baseline_results.csv
```

Rows:

```text
logistic_regression_seed_42
mlp_baseline_seed_42
```

---

## 6.2.1 Strong Non-DP Baseline Search

Goal:

```text
Establish a stronger non-private upper reference before arguing that DP-SGD
reduces utility.
```

Models:

```text
HistGradientBoostingClassifier
XGBoost
LightGBM
CatBoost
RandomForestClassifier
ExtraTreesClassifier
LogisticRegression
Tuned PyTorch MLP
```

Protocol:

- Use the original Adult train/test files.
- Split the training file internally into fit/validation for threshold tuning.
- Select classification threshold by validation F1-score, not by test labels.
- Evaluate the selected model on the held-out Adult test file.
- Report both default-threshold accuracy and threshold-tuned F1-score.

Outputs:

```text
results/tuned_baseline_results.csv
results/best_tuned_baseline.json
results/strong_baseline_results.csv
results/best_strong_baseline.json
results/strong_baseline_full_train.csv
results/best_strong_baseline_full_train.json
results/boosted_feature_baseline_results.csv
results/best_boosted_feature_baseline.json
```

Important interpretation:

```text
A clean baseline near 0.87 accuracy / 0.72 F1 is already strong for this split.
Targets around 0.90 should be treated as stretch goals or leak checks unless a
new model family or feature strategy proves otherwise.
```

Feature-engineered boosted baseline:

```text
Add numeric interaction features, numeric bins, and selected categorical
interaction features. Compare XGBoost, LightGBM, CatBoost, and
HistGradientBoosting with validation-selected F1 threshold.
```

---

## 6.3 Noise Sweep Experiment

Grid:

```python
noise_multipliers = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
```

Fixed:

```python
max_grad_norm = 1.0
batch_size = 256
epochs = 20
delta = 1e-5
seed = 42
```

Output:

```text
results/dp_sgd_noise_results.csv
```

Key figures:

```text
figures/noise_vs_epsilon.png
figures/noise_vs_accuracy.png
figures/noise_vs_f1.png
figures/privacy_utility_tradeoff.png
```

---

## 6.4 Max Grad Norm Sweep Experiment

Grid:

```python
max_grad_norm_values = [0.5, 1.0, 1.5, 2.0]
```

Fixed:

```python
noise_multiplier = 1.5
batch_size = 256
epochs = 20
delta = 1e-5
seed = 42
```

Output:

```text
results/max_grad_norm_results.csv
```

Key figures:

```text
figures/max_grad_norm_vs_accuracy.png
figures/max_grad_norm_vs_f1.png
```

---

## 6.4.1 Joint Noise x Clipping Stress Test

Goal:

```text
Answer whether the original noise range was too small to show a clear utility
collapse, and test whether the F1-score trend depends on the clipping norm.
```

Grid:

```python
noise_multiplier = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
max_grad_norm = [0.5, 1.0, 2.0, 3.0]
```

Fixed:

```python
batch_size = 256
epochs = 20
delta = 1e-5
learning_rate = 0.05
seed = 42
```

Primary metric:

```text
F1-score
```

Reason:

- UCI Adult is class-imbalanced, so F1-score is more informative than accuracy.
- Very large noise values are used as a stress test, not necessarily as practical
  final configurations.
- The experiment helps distinguish "the tested noise was not large enough" from
  "DP-SGD was implemented incorrectly".

Output:

```text
results/noise_clip_grid_results.csv
results/noise_clip_grid_summary.json
docs/noise_clip_stress_notes.md
figures/noise_clip_f1_heatmap.png
figures/noise_clip_f1_lines.png
figures/extreme_noise_best_f1.png
```

Execution:

```powershell
.\.venv\Scripts\python.exe scripts\run_noise_clip_stress_gpu.py
```

---

## 6.4.2 Joint Model x Noise x Clipping Grid

Goal:

```text
Test whether the apparent DP utility behavior depends on the MLP architecture,
not only on the noise multiplier and clipping norm.
```

Model variants:

```python
model_variants = [
    {"name": "mlp_64_32_d0p1", "hidden_dims": (64, 32), "dropout": 0.1},
    {"name": "mlp_128_64_d0p1", "hidden_dims": (128, 64), "dropout": 0.1},
    {"name": "mlp_128_64_d0p0", "hidden_dims": (128, 64), "dropout": 0.0},
    {"name": "mlp_256_128_d0p1", "hidden_dims": (256, 128), "dropout": 0.1},
]
```

Grid:

```python
noise_multiplier = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
max_grad_norm = [0.5, 1.0, 2.0, 3.0]
```

Total:

```text
4 model variants x 9 noise values x 4 clipping values = 144 DP-SGD runs
```

Outputs:

```text
results/model_clip_noise_grid_results.csv
results/model_clip_noise_grid_summary.json
docs/model_clip_noise_grid_notes.md
figures/model_clip_noise_best_f1.png
```

Execution:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_clip_noise_grid_gpu.py
```

---

## 6.4.3 Best Variant 50-Epoch Noise x Clipping Grid

Goal:

```text
Keep only the best model variant from the full model x clipping x noise grid
and rerun noise x clipping with longer fixed training.
```

Selected model:

```python
model_variant = "mlp_64_32_d0p1"
hidden_dims = (64, 32)
dropout = 0.1
```

Fixed:

```python
epochs = 50
early_stopping = False
batch_size = 256
learning_rate = 0.05
momentum = 0.9
delta = 1e-5
```

Grid:

```python
noise_multiplier = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
max_grad_norm = [0.5, 1.0, 2.0, 3.0]
```

Outputs:

```text
results/best_variant_50epoch_noise_clip_results.csv
results/best_variant_50epoch_noise_clip_summary.json
docs/best_variant_50epoch_noise_clip_notes.md
figures/best_variant_50epoch_f1_heatmap.png
figures/best_variant_50epoch_f1_lines.png
```

Execution:

```powershell
.\.venv\Scripts\python.exe scripts\run_best_variant_50epoch_noise_clip_gpu.py
```

---

## 6.5 Batch Size Sweep Experiment

Grid:

```python
batch_size_values = [64, 128, 256, 512]
```

Fixed:

```python
noise_multiplier = 1.5
max_grad_norm = 1.0
epochs = 20
delta = 1e-5
seed = 42
```

Output:

```text
results/batch_size_results.csv
```

Key figures:

```text
figures/batch_size_vs_epsilon.png
figures/batch_size_vs_accuracy.png
```

---

## 6.6 Multiple Seed Experiment

Grid:

```python
seeds = [42, 123, 2026]
configs = [
    {"model": "mlp_baseline"},
    {"model": "dp_sgd", "noise_multiplier": 1.0},
    {"model": "dp_sgd", "noise_multiplier": 1.5},
    {"model": "dp_sgd", "noise_multiplier": 3.0},
]
```

Output:

```text
results/multi_seed_results.csv
```

Aggregate output:

```text
mean_accuracy
std_accuracy
mean_f1_score
std_f1_score
mean_roc_auc
std_roc_auc
mean_pr_auc
std_pr_auc
mean_epsilon
std_epsilon
```

Key figures:

```text
figures/multi_seed_errorbar_accuracy.png
figures/multi_seed_errorbar_f1.png
```

---

## 6.7 Membership Inference Attack Experiment

Models:

```text
MLP baseline
DP-SGD noise 1.5
DP-SGD noise 3.0
```

Attack feature:

```text
confidence = max predicted probability
```

Attack target:

```text
1 if sample is from train set
0 if sample is from test set
```

Metric:

```text
attack_auc
```

Output:

```text
results/mia_results.csv
```

Figures:

```text
figures/confidence_distribution_baseline.png
figures/confidence_distribution_dp_sgd.png
figures/attack_auc_comparison.png
```

---

## 7. Result Schema

All result CSV files should use consistent columns as much as possible.

### 7.1 Common result columns

```text
experiment_id
seed
model_name
is_dp
dataset
train_size
test_size
input_dim
epochs
batch_size
learning_rate
optimizer
momentum
noise_multiplier
max_grad_norm
delta
epsilon
accuracy
precision
recall
f1_score
roc_auc
pr_auc
training_time
notes
```

### 7.2 MIA result columns

```text
experiment_id
model_name
is_dp
noise_multiplier
epsilon
attack_type
attack_auc
mean_train_confidence
mean_test_confidence
confidence_gap
num_train_samples
num_test_samples
notes
```

### 7.3 Final summary columns

```text
section
main_finding
best_config
epsilon
accuracy
f1_score
attack_auc
interpretation
```

---

## 8. Data Flow Diagrams

## 8.1 Training Data Flow

```text
adult.data / adult.test
        ↓
load_adult_raw()
        ↓
clean_adult_dataframe()
        ↓
ColumnTransformer
        ├── StandardScaler for numeric features
        └── OneHotEncoder for categorical features
        ↓
X_train, X_test, y_train, y_test
        ↓
Baseline training / DP-SGD training
        ↓
Evaluation metrics
        ↓
CSV results + figures
```

## 8.2 DP-SGD Training Flow

```text
Mini-batch samples
        ↓
Compute per-example gradients
        ↓
Clip each gradient to max_grad_norm
        ↓
Add Gaussian noise controlled by noise_multiplier
        ↓
Average sanitized gradients
        ↓
Update model parameters
        ↓
PrivacyEngine updates privacy accountant
        ↓
Report epsilon for fixed delta
```

## 8.3 Membership Inference Flow

```text
Trained model
   ├── Predict on train samples
   │       ↓
   │   train confidence scores
   │
   └── Predict on test samples
           ↓
       test confidence scores

train confidence -> membership label 1
test confidence  -> membership label 0
        ↓
ROC-AUC(confidence, membership label)
        ↓
attack_auc
```

---

## 9. Dependency Architecture

Recommended `requirements.txt`:

```text
numpy
pandas
scikit-learn
matplotlib
torch
opacus
jupyter
notebook
tqdm
```

Optional:

```text
seaborn
python-docx
nbconvert
```

If using Colab, install Opacus in the first cell:

```python
!pip install opacus
```

---

## 10. Configuration Architecture

Every experiment should be controlled by a configuration dictionary.

Example:

```python
config = {
    "experiment_name": "dp_sgd_noise_sweep",
    "seed": 42,
    "dataset": "uci_adult",
    "model": "mlp",
    "is_dp": True,
    "epochs": 20,
    "batch_size": 256,
    "learning_rate": 0.05,
    "optimizer": "sgd",
    "momentum": 0.9,
    "noise_multiplier": 1.5,
    "max_grad_norm": 1.0,
    "delta": 1e-5,
}
```

This config should be stored or copied into result rows.

---

## 11. Model Architecture

## 11.1 Logistic Regression Baseline

Purpose:

- Simple baseline.
- Helps show whether MLP is necessary.
- Non-DP reference.

Implementation:

```python
sklearn.linear_model.LogisticRegression
```

Recommended settings:

```python
max_iter=1000
class_weight=None
random_state=seed
```

## 11.2 MLP Baseline and DP-SGD MLP

Use the same architecture for non-DP and DP-SGD experiments.

Recommended architecture:

```text
Input layer: input_dim
Hidden layer 1: Linear(input_dim, 128), ReLU, Dropout(0.1)
Hidden layer 2: Linear(128, 64), ReLU, Dropout(0.1)
Output layer: Linear(64, 2)
```

Reason:

- Simple enough for Opacus.
- No BatchNorm.
- Works with tabular data.
- Fast enough for repeated experiments.
- Comparable between DP and non-DP.

Loss:

```text
CrossEntropyLoss
```

Prediction:

```text
softmax(logits)
class = argmax(probability)
```

---

## 12. Privacy Architecture

## 12.1 Formal DP Component

Formal privacy is provided by:

```text
Opacus PrivacyEngine
```

Mechanism:

```text
per-example gradient calculation
gradient clipping
Gaussian noise addition
privacy accounting
```

Reported values:

```text
epsilon
delta
```

In this project, delta is fixed:

```text
delta = 1e-5
```

## 12.2 Empirical Privacy-Risk Component

Empirical risk is evaluated by:

```text
confidence-based membership inference attack
```

This is not a formal DP proof. It is an intuitive attack evaluation to show whether the model leaks membership information through prediction confidence.

Reported value:

```text
attack_auc
```

Interpretation:

```text
attack_auc = 0.5 -> random guessing
attack_auc > 0.5 -> possible membership leakage
```

## 12.3 Relationship Between Epsilon and MIA

Important explanation:

- Epsilon is a formal privacy budget computed from the DP training process.
- MIA attack AUC is an empirical test of one possible privacy risk.
- They are related but not identical.
- A smaller epsilon should generally reduce membership leakage, but simple attack results may depend on overfitting, dataset, model, and attack strength.

---

## 13. Evaluation Architecture

## 13.1 Utility Metrics

Use these metrics for all models:

```text
accuracy
precision
recall
f1_score
roc_auc
pr_auc
confusion_matrix
training_time
```

Why not only accuracy?

- UCI Adult is imbalanced.
- Accuracy can hide poor performance on the positive class.
- F1-score, ROC-AUC, and PR-AUC provide better insight.

## 13.2 Privacy Metrics

Use:

```text
epsilon at delta = 1e-5
attack_auc
confidence_gap
```

Definitions:

```text
epsilon: formal privacy loss
attack_auc: empirical membership inference risk
confidence_gap: mean_train_confidence - mean_test_confidence
```

## 13.3 Tradeoff Metrics

Compute:

```text
accuracy_drop = baseline_accuracy - dp_accuracy
f1_drop = baseline_f1 - dp_f1
epsilon_reduction = epsilon_low_noise - epsilon_current
```

Use these to explain practical tradeoffs.

---

## 14. Reproducibility Architecture

## 14.1 Seed Control

Every experiment must call:

```python
set_seed(seed)
```

before:

- train/test split
- model initialization
- DataLoader creation
- training loop

## 14.2 Data Split Control

The final report must state exactly which split strategy was used:

Option A:

```text
Use original adult.data as train and adult.test as test
```

Option B:

```text
Combine data and use stratified 80/20 split
```

Do not mix both without explanation.

## 14.3 Result Versioning

If results are rerun, do not overwrite important final files without saving old version.

Recommended:

```text
results/archive/
```

or add timestamp:

```text
dp_sgd_noise_results_2026_06_07.csv
```

For final report, use stable filenames:

```text
results/final_summary.csv
```

---

## 15. Report Integration Architecture

The report should not manually invent result values. It should draw from:

```text
results/final_summary.csv
results/baseline_results.csv
results/dp_sgd_noise_results.csv
results/max_grad_norm_results.csv
results/batch_size_results.csv
results/multi_seed_results.csv
results/mia_results.csv
```

Figures should be inserted from:

```text
figures/
```

Report mapping:

| Report section | Source |
|---|---|
| Dataset | `data/data.md`, preprocessing metadata |
| Baseline results | `baseline_results.csv` |
| Noise sweep | `dp_sgd_noise_results.csv` |
| Clipping analysis | `max_grad_norm_results.csv` |
| Batch size analysis | `batch_size_results.csv` |
| Robustness | `multi_seed_results.csv` |
| MIA | `mia_results.csv` |
| Final recommendation | `final_summary.csv` |

---

## 16. Slide Integration Architecture

Slides should use only final cleaned figures and tables.

Do not screenshot notebook cells.

Slide assets:

```text
figures/dp_sgd_workflow.png
figures/privacy_utility_tradeoff.png
figures/noise_vs_epsilon.png
figures/noise_vs_accuracy.png
figures/max_grad_norm_vs_f1.png
figures/batch_size_vs_epsilon.png
figures/attack_auc_comparison.png
```

Recommended slide logic:

```text
Problem
↓
DP-SGD method
↓
Dataset
↓
Experiments
↓
Results
↓
Attack evaluation
↓
Recommended configuration
↓
Conclusion
```

---

## 17. Demo Architecture

The demo should avoid long training if presentation time is short.

Recommended demo modes:

### Mode 1 – Safe demo

Load saved CSV results and figures.

```text
Fast, reliable, best for presentation.
```

### Mode 2 – Partial live demo

Train only one DP-SGD configuration:

```text
noise_multiplier = 1.5
epochs = 3 to 5 for demo
```

Then show that full experiment used 20 epochs.

### Mode 3 – Full live demo

Train baseline and DP-SGD for 20 epochs.

```text
Only use this if hardware is reliable and enough time is available.
```

Recommended final choice:

```text
Mode 1 + briefly show code from Mode 2
```

---

## 18. Risk Management

## 18.1 Technical Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Opacus error with model layer | DP training fails | Avoid BatchNorm, use simple MLP |
| Training too slow | Cannot finish sweeps | Reduce epochs temporarily, use Colab |
| Results differ from current report | Confusion | Explain seed/split/library difference |
| MIA result not strong | Weak privacy story | Explain formal epsilon and simple attack limitation |
| Batch size sweep unstable | Hard to interpret | Report honestly and focus on observed trend |

## 18.2 Presentation Risks

| Risk | Mitigation |
|---|---|
| Audience does not understand epsilon | Use simple explanation: smaller epsilon = stronger privacy |
| Asked why not MNIST | Explain UCI Adult better matches personal data |
| Asked if DP-SGD prevents all attacks | Say no, it provides formal bounded influence |
| Asked why accuracy drop is small | Explain dataset simplicity and moderate noise |
| Asked whether attack proves DP | Say no, MIA is empirical and complementary |

---

## 19. Development Roadmap

## Phase 1 – Stabilize Current Project

```text
1. Create final folder structure.
2. Move existing notebooks and report.
3. Reproduce current baseline and DP-SGD noise sweep.
4. Save results and figures.
```

## Phase 2 – Add Research Depth

```text
1. Run max grad norm sweep.
2. Run batch size sweep.
3. Run multiple seed evaluation.
4. Add ROC-AUC and PR-AUC to all result tables.
```

## Phase 3 – Add Privacy Attack

```text
1. Implement confidence-based MIA.
2. Compare baseline and DP models.
3. Save confidence distribution plots.
4. Save attack AUC comparison.
```

## Phase 4 – Finalize Deliverables

```text
1. Write final report.
2. Create final slides.
3. Create demo notebook.
4. Cross-check all numbers.
5. Prepare defense answers.
```

---

## 20. Final Architecture Summary

The final architecture turns the project into a complete privacy-preserving ML research pipeline:

```text
UCI Adult personal tabular data
        ↓
Preprocessing and feature engineering
        ↓
Baseline non-private models
        ↓
DP-SGD models with Opacus
        ↓
Noise, clipping, batch size, seed experiments
        ↓
Formal privacy budget evaluation using epsilon
        ↓
Empirical membership inference risk evaluation
        ↓
Final report, slides, and demo
```

The most important architectural principle is separation of concerns:

- `data/` stores input and processed data.
- `src/` stores reusable implementation.
- `notebooks/` run experiments.
- `results/` stores numerical outputs.
- `figures/` stores visual outputs.
- `docs/`, `report/`, and `slides/` store final communication materials.

---

## 21. Current Implementation Status

The final architecture has been implemented in this workspace.

### Synchronized guide files

- `Agent.md` has been synchronized from `AGENT(10).md`.
- `Architecture.md` has been synchronized from `ARCHITECTURE(2).md`.
- The temporary files `AGENT(10).md` and `ARCHITECTURE(2).md` were deleted
  after synchronization.

### Environment

- Local virtual environment: `.venv/`
- Python: 3.12.10
- PyTorch: `2.11.0+cu128`
- Opacus: `1.6.0`
- Execution device: NVIDIA T1200 Laptop GPU through CUDA

### Completed experiment files

- `results/baseline_results.csv`
- `results/dp_sgd_noise_results.csv`
- `results/max_grad_norm_results.csv`
- `results/noise_clip_grid_results.csv`
- `results/tuned_baseline_results.csv`
- `results/best_tuned_baseline.json`
- `results/strong_baseline_results.csv`
- `results/best_strong_baseline.json`
- `results/strong_baseline_full_train.csv`
- `results/best_strong_baseline_full_train.json`
- `results/boosted_feature_baseline_results.csv`
- `results/best_boosted_feature_baseline.json`
- `results/model_clip_noise_grid_results.csv`
- `results/model_clip_noise_grid_summary.json`
- `results/batch_size_results.csv`
- `results/multi_seed_runs.csv`
- `results/multi_seed_results.csv`
- `results/mia_results.csv`
- `results/final_summary.csv`

### Completed figures

- `figures/noise_vs_epsilon.png`
- `figures/noise_vs_accuracy.png`
- `figures/noise_vs_f1.png`
- `figures/privacy_utility_tradeoff.png`
- `figures/max_grad_norm_vs_accuracy.png`
- `figures/max_grad_norm_vs_f1.png`
- `figures/noise_clip_f1_heatmap.png`
- `figures/noise_clip_f1_lines.png`
- `figures/extreme_noise_best_f1.png`
- `figures/model_clip_noise_best_f1.png`
- `figures/batch_size_vs_epsilon.png`
- `figures/batch_size_vs_accuracy.png`
- `figures/multi_seed_errorbar_accuracy.png`
- `figures/multi_seed_errorbar_f1.png`
- `figures/attack_auc_comparison.png`

### Key final numbers

- Logistic Regression baseline: accuracy `0.8480`, F1-score `0.6611`.
- MLP baseline: accuracy `0.8487`, F1-score `0.6645`.
- Tuned non-DP MLP reference: accuracy `0.8438`, F1-score `0.6951`
  using `hidden_dims = 128-64`, `dropout = 0.1`, SGD learning rate `0.03`,
  40 epochs, and mild class weighting. This is the stricter F1 reference for
  judging whether DP noise hurts utility.
- Strong clean classical baseline: HistGradientBoosting with validation-tuned
  threshold reaches default accuracy `0.8708`, threshold-tuned accuracy
  `0.8586`, F1-score `0.7261`, ROC-AUC `0.9266`, and PR-AUC `0.8288`.
- Boosted + feature-engineered baseline tried XGBoost `3.2.0`, LightGBM
  `4.6.0`, CatBoost `1.2.10`, and HistGradientBoosting with numeric bins and
  interaction features. Best result: HistGradientBoosting with engineered
  features, default accuracy `0.8689`, default F1-score `0.7128`,
  threshold-tuned F1-score `0.7273`, ROC-AUC `0.9259`, and PR-AUC `0.8278`.
- No clean local baseline run reached `0.90` accuracy or F1 on the original
  Adult test split. A `0.90` target should therefore be treated as a stretch
  goal that may require a new model family, more feature engineering, or a
  leak audit if it appears suddenly.
- Full DP model x clipping x noise grid completed `144` GPU runs with no
  failures. Best DP F1-score is `0.6760` from `mlp_64_32_d0p1`,
  `noise_multiplier = 0.5`, and `max_grad_norm = 1.0`, still about `0.0513`
  below the boosted feature baseline F1-score `0.7273`.
- The worst DP grid result is `mlp_64_32_d0p1`, `noise_multiplier = 15.0`,
  `max_grad_norm = 3.0`, epsilon `0.0934`, accuracy `0.7365`, and F1-score
  `0.2413`, showing a clear utility collapse under extreme noise/clipping.
- DP-SGD noise `1.5`: epsilon `1.2143`, accuracy `0.8497`,
  F1-score `0.6571`.
- DP-SGD noise `3.0`: epsilon `0.5124`, accuracy `0.8465`,
  F1-score `0.6526`.
- Best clipping F1-score in the final two-logit pipeline:
  `max_grad_norm = 2.0`, F1-score `0.6619`.
- Joint noise x clipping stress test completed 36 GPU runs:
  noise values `[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]`
  and clipping values `[0.5, 1.0, 2.0, 3.0]`.
- Compared with the tuned non-DP baseline, the best DP result in the joint grid
  is lower: noise `0.5`, `max_grad_norm = 3.0`, epsilon `16.8660`, and
  F1-score `0.6667`.
- Best-per-noise F1 first drops at least `0.05` below the tuned non-DP baseline
  at noise `3.0`, with epsilon `0.5124` and F1-score `0.6387`.
- At extreme noise `20.0`, the best clipping setting is `max_grad_norm = 0.5`,
  epsilon `0.0707`, accuracy `0.7803`, and F1-score `0.5135`.
- Confidence-based MIA AUC values are close to `0.5`:
  baseline `0.5021`, DP noise `1.5` `0.5021`, DP noise `3.0` `0.5016`.
- GPU tuning was run after installing CUDA PyTorch (`torch 2.11.0+cu128`).
  The best tuned DP-SGD configuration for `noise_multiplier = 1.5` is
  `max_grad_norm = 2.0`, `batch_size = 512`, `epochs = 20`,
  `learning_rate = 0.08`, `schedule = cosine`, with epsilon `1.7991`,
  accuracy `0.8531`, and F1-score `0.6680`.
- Detailed tuning notes are stored in `docs/tuning_notes.md`.

### Execution commands

```powershell
.\.venv\Scripts\python.exe scripts\run_final_experiments.py
.\.venv\Scripts\python.exe scripts\tune_non_dp_baseline_gpu.py
.\.venv\Scripts\python.exe scripts\tune_strong_classical_baselines.py
.\.venv\Scripts\python.exe scripts\tune_boosted_feature_baselines.py
.\.venv\Scripts\python.exe scripts\run_noise_clip_stress_gpu.py
.\.venv\Scripts\python.exe scripts\run_model_clip_noise_grid_gpu.py
.\.venv\Scripts\python.exe scripts\tune_dp_sgd_gpu.py
.\.venv\Scripts\python.exe scripts\tune_dp_sgd_gpu_extra.py
.\.venv\Scripts\python.exe scripts\generate_final_deliverables.py
```

This architecture is sufficient for a 5-person group project and supports a strong final presentation: not only showing that DP-SGD works, but also analyzing its privacy-utility tradeoff and empirically evaluating membership inference risk.
