# AGENT.md

# Privacy-Preserving Data with Differential Privacy – DP-SGD Final Project Agent Guide

## 0. Project Identity

**Project title:** Privacy-Preserving Data with Differential Privacy using DP-SGD  
**Pillar:** Privacy in Development  
**Core method:** Differentially Private Stochastic Gradient Descent  
**Main library:** PyTorch + Opacus  
**Main dataset:** UCI Adult / Census Income Dataset  
**Primary task:** Binary classification of income group, `>50K` vs `<=50K`  
**Source paper:** Deep Learning with Differential Privacy – Abadi et al.  
**Final project level:** Group project for 5 members, research-style implementation and evaluation

This file defines how every agent, team member, or coding assistant should contribute to the final project. The goal is not only to train a DP-SGD model, but to build a complete privacy-preserving machine learning project with reproducible experiments, privacy-utility analysis, hyperparameter studies, and a simple privacy attack evaluation.

---

## 1. Final Project Goal

The final project must answer the following central question:

> Can DP-SGD be integrated into a normal PyTorch training pipeline to reduce privacy risk while preserving useful model performance on a personal tabular dataset?

The project must demonstrate three layers:

1. **Core DP-SGD implementation**
   - Train baseline models without Differential Privacy.
   - Train MLP models with DP-SGD using Opacus.
   - Measure epsilon, accuracy, F1-score, ROC-AUC, PR-AUC, and training time.

2. **Privacy-utility and hyperparameter analysis**
   - Analyze how `noise_multiplier` changes privacy budget and utility.
   - Analyze how `max_grad_norm` changes model performance.
   - Analyze how `batch_size` affects epsilon and utility.
   - Run selected configurations with multiple random seeds.

3. **Privacy attack evaluation**
   - Implement a simple membership inference attack based on model confidence.
   - Compare attack risk between baseline non-DP model and DP-SGD models.
   - Use attack AUC as the main attack metric.

The project should be strong enough for a 5-person group and should look like a mini research project, not just a simple library demo.

---

## 2. Research Questions

### RQ1 – Privacy Budget vs Utility

How does increasing `noise_multiplier` affect epsilon, accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and training time?

Expected direction:

- Higher `noise_multiplier` → lower epsilon → stronger privacy.
- Higher `noise_multiplier` → possible decrease in utility.

### RQ2 – Effect of Gradient Clipping

How does changing `max_grad_norm` affect DP-SGD performance?

Expected discussion:

- Too small `max_grad_norm` clips too many gradients and may remove useful learning signal.
- Too large `max_grad_norm` allows larger per-example influence and may make noisy updates less stable.
- A middle value may provide a better privacy-utility balance.

### RQ3 – Effect of Batch Size

How does changing `batch_size` affect privacy and utility?

Expected discussion:

- Batch size affects sampling rate.
- Sampling rate affects privacy accounting.
- Batch size also changes optimization stability.
- Larger batch is not always better under DP-SGD.

### RQ4 – Robustness Across Seeds

Are the results stable across random seeds?

Expected discussion:

- Report mean and standard deviation.
- Avoid relying on a single lucky run.
- Use multiple seeds for selected important configurations.

### RQ5 – Membership Inference Risk

Does DP-SGD reduce membership inference risk compared with non-DP training?

Expected discussion:

- Non-DP models may produce higher confidence on train samples than test samples.
- A confidence-based attacker may exploit this gap.
- DP-SGD should reduce overconfident memorization and move attack AUC closer to 0.5.
- Attack AUC = 0.5 means nearly random guessing.
- Attack AUC > 0.5 means membership information may be leaked.

---

## 3. Final Scope

### 3.1 Must-have components

The final project must include:

- UCI Adult preprocessing pipeline
- Baseline Logistic Regression
- Baseline MLP without DP
- DP-SGD MLP with Opacus
- Noise multiplier sweep
- Max grad norm sweep
- Batch size sweep
- Multiple-seed evaluation
- Membership inference attack
- Figures and CSV result files
- Final report
- Final slides
- Demo notebook
- README with execution instructions

### 3.2 Optional components

Only add these if the required work is complete:

- MNIST replication as comparison with Abadi et al.
- Target epsilon training instead of manually selecting noise.
- Different Opacus accountants if supported by environment.
- Larger MLP architecture comparison.
- Fairness analysis across sensitive attributes.

### 3.3 Out of scope

Do not spend time on these unless specifically requested:

- Implementing DP-SGD from scratch.
- Proving Differential Privacy mathematically from first principles.
- Implementing full model inversion attack.
- Training large CNNs.
- Using multiple datasets without finishing UCI Adult.
- Building a web application.

---

## 4. Team Roles for 5 Members

## Member 1 – Theory and Paper Lead

### Responsibilities

- Read and summarize Deep Learning with Differential Privacy.
- Explain Differential Privacy, epsilon, delta, DP-SGD, clipping, Gaussian noise, and privacy accountant.
- Write the paper comparison section.
- Prepare theory slides.

### Required outputs

```text
docs/theory_summary.md
docs/paper_comparison.md
figures/dp_sgd_workflow.png
```

### Content requirements

The theory section must explain:

- Why ML models may memorize training records.
- Why memorization is a privacy risk.
- What Differential Privacy guarantees.
- Why epsilon is important.
- Why DP-SGD uses per-example gradients.
- Why gradient clipping is necessary.
- Why Gaussian noise is added.
- How privacy accounting tracks cumulative privacy loss.
- How this project differs from Abadi et al.

### Quality bar

The explanation must be simple enough for presentation but technically correct. Avoid claiming that DP-SGD prevents every privacy attack. Say that it provides a formal bound on the influence of individual training records.

---

## Member 2 – Dataset and Baseline Lead

### Responsibilities

- Load UCI Adult Dataset.
- Clean missing values represented by `?`.
- Normalize label values.
- Encode categorical features.
- Scale numerical features.
- Split train/test data.
- Train Logistic Regression baseline.
- Train non-DP MLP baseline.

### Required outputs

```text
notebooks/01_preprocessing.ipynb
notebooks/02_baseline_models.ipynb
results/baseline_results.csv
results/preprocessing_summary.json
figures/baseline_confusion_matrix.png
```

### Required metrics

For every baseline model, report:

```text
model_name
accuracy
precision
recall
f1_score
roc_auc
pr_auc
training_time
confusion_matrix
```

### Baseline models

Minimum:

1. Logistic Regression
2. MLP non-DP

The MLP baseline is the main comparison point for DP-SGD.

### Recommended preprocessing

Numerical columns:

```text
age
fnlwgt
education-num
capital-gain
capital-loss
hours-per-week
```

Categorical columns:

```text
workclass
education
marital-status
occupation
relationship
race
sex
native-country
```

Target column:

```text
income
```

Recommended transformations:

- Replace `?` with missing.
- Drop missing rows or impute consistently.
- One-hot encode categorical columns.
- Standardize numerical columns.
- Convert labels:
  - `<=50K` → 0
  - `>50K` → 1
- Use stratified train/test split.

---

## Member 3 – DP-SGD Experiment Lead

### Responsibilities

- Implement DP-SGD training using Opacus.
- Run noise multiplier sweep.
- Record epsilon and utility metrics.
- Export result CSV and figures.

### Required outputs

```text
notebooks/03_dp_sgd_noise_sweep.ipynb
results/dp_sgd_noise_results.csv
figures/noise_vs_epsilon.png
figures/noise_vs_accuracy.png
figures/privacy_utility_tradeoff.png
figures/noise_vs_f1.png
```

### Main noise sweep configuration

```python
noise_multipliers = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
delta = 1e-5
max_grad_norm = 1.0
batch_size = 256
epochs = 20
optimizer = "SGD"
momentum = 0.9
learning_rate = 0.05
```

### Required metrics per run

```text
experiment_id
model_name
noise_multiplier
max_grad_norm
batch_size
epochs
delta
epsilon
accuracy
precision
recall
f1_score
roc_auc
pr_auc
training_time
accuracy_drop_vs_mlp_baseline
f1_drop_vs_mlp_baseline
```

### Current known reference results

Use these as the initial benchmark from the current project:

| Method | Noise | Epsilon | Accuracy | F1-score | Training time |
|---|---:|---:|---:|---:|---:|
| Baseline MLP | N/A | N/A | 0.8501 | 0.6567 | 19.59 |
| DP-SGD | 0.5 | 15.5548 | 0.8491 | 0.6683 | 32.26 |
| DP-SGD | 0.8 | 3.5038 | 0.8485 | 0.6523 | 25.01 |
| DP-SGD | 1.0 | 2.1115 | 0.8479 | 0.6572 | 23.80 |
| DP-SGD | 1.5 | 1.0946 | 0.8464 | 0.6544 | 25.03 |
| DP-SGD | 2.0 | 0.7501 | 0.8460 | 0.6446 | 23.64 |
| DP-SGD | 3.0 | 0.4637 | 0.8425 | 0.6518 | 23.57 |

The final project can rerun and update these results, but if new values differ, explain the reason: seed, library version, preprocessing, train/test split, or hardware.

### Recommended configuration for discussion

Use this as the main balanced configuration unless later experiments prove otherwise:

```text
noise_multiplier = 1.5
epsilon ≈ 1.0946
accuracy ≈ 0.8464
f1_score ≈ 0.6544
```

Reason:

- Much stronger privacy than noise 0.5.
- Accuracy only slightly lower than baseline.
- Easy to defend in presentation.

---

## Member 4 – Hyperparameter and Robustness Lead

### Responsibilities

- Run max grad norm sweep.
- Run batch size sweep.
- Run multiple-seed evaluation.
- Analyze robustness and stability.

### Required outputs

```text
notebooks/04_dp_sgd_max_grad_norm_sweep.ipynb
notebooks/05_dp_sgd_batch_size_sweep.ipynb
notebooks/06_multi_seed_experiment.ipynb
results/max_grad_norm_results.csv
results/batch_size_results.csv
results/multi_seed_results.csv
figures/max_grad_norm_vs_accuracy.png
figures/max_grad_norm_vs_f1.png
figures/batch_size_vs_epsilon.png
figures/batch_size_vs_accuracy.png
figures/multi_seed_errorbar_accuracy.png
figures/multi_seed_errorbar_f1.png
```

### Max grad norm sweep

Fixed:

```python
noise_multiplier = 1.5
batch_size = 256
epochs = 20
delta = 1e-5
learning_rate = 0.05
```

Vary:

```python
max_grad_norm_values = [0.5, 1.0, 1.5, 2.0]
```

Report:

```text
max_grad_norm
epsilon
accuracy
precision
recall
f1_score
roc_auc
pr_auc
training_time
```

### Batch size sweep

Fixed:

```python
noise_multiplier = 1.5
max_grad_norm = 1.0
epochs = 20
delta = 1e-5
learning_rate = 0.05
```

Vary:

```python
batch_sizes = [64, 128, 256, 512]
```

Report:

```text
batch_size
epsilon
accuracy
precision
recall
f1_score
roc_auc
pr_auc
training_time
```

### Multiple seed evaluation

Seeds:

```python
seeds = [42, 123, 2026]
```

Configurations:

```text
MLP baseline
DP-SGD noise 1.0
DP-SGD noise 1.5
DP-SGD noise 3.0
```

Report:

```text
model_config
mean_accuracy
std_accuracy
mean_f1
std_f1
mean_roc_auc
std_roc_auc
mean_pr_auc
std_pr_auc
mean_epsilon
std_epsilon
```

### Quality bar

Do not only dump tables. Explain what the results mean:

- Which hyperparameter is most sensitive?
- Which configuration is stable?
- Which setting gives the best privacy-utility tradeoff?
- Are results stable across seeds?

---

## Member 5 – Privacy Attack, Report, and Presentation Lead

### Responsibilities

- Implement simple membership inference attack.
- Compare baseline and DP-SGD models.
- Create final figures for privacy risk.
- Assemble report and presentation.
- Prepare defense answers.

### Required outputs

```text
notebooks/07_membership_inference_attack.ipynb
results/mia_results.csv
figures/confidence_distribution_baseline.png
figures/confidence_distribution_dp_sgd.png
figures/attack_auc_comparison.png
report/report.md
report/report.docx
slides/dp_sgd_final_presentation.pptx
```

### Membership inference attack design

The attack is confidence-based.

For each trained model:

1. Compute predicted probability for every sample in train set.
2. Compute predicted probability for every sample in test set.
3. For each sample, define confidence as:

```text
confidence = max predicted class probability
```

4. Construct attack dataset:

```text
train samples -> membership label = 1
test samples -> membership label = 0
feature = confidence
```

5. Compute attack AUC:

```text
attack_auc = ROC-AUC(confidence, membership_label)
```

Interpretation:

```text
attack_auc ≈ 0.5 -> attacker is close to random guessing
attack_auc > 0.5 -> membership information may be leaked
higher attack_auc -> higher privacy risk
```

### Models to compare

Minimum:

```text
MLP baseline non-DP
DP-SGD noise 1.5
DP-SGD noise 3.0
```

Optional:

```text
DP-SGD noise 1.0
Logistic Regression baseline
```

### Expected conclusion

The best case is:

```text
attack_auc_baseline > attack_auc_dp_sgd_noise_1_5 > attack_auc_dp_sgd_noise_3_0 ≈ 0.5
```

If the result is not perfectly monotonic, explain honestly:

- UCI Adult is not a highly overfit dataset.
- The attack is simple.
- DP-SGD still provides formal privacy accounting even if confidence-based MIA is weak.
- More sophisticated attacks could be tested in future work.

---

## 5. Required Project Directory Structure

Use this structure.

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
│   ├── processed/
│   │   ├── X_train.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   ├── y_test.npy
│   │   └── feature_names.json
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
│   └── report.docx
│
└── slides/
    └── dp_sgd_final_presentation.pptx
```

---

## 6. Coding Standards

### 6.1 General style

All code should be reproducible, modular, clear, light enough to run on a normal laptop or Colab, written in Python, and compatible with PyTorch and Opacus.

### 6.2 Random seed

Every experiment must set seed:

```python
def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### 6.3 Result logging

Every experiment must write CSV output. Do not rely only on notebook cell outputs.

Minimum row fields:

```text
experiment_id
seed
model_name
is_dp
noise_multiplier
max_grad_norm
batch_size
epochs
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

### 6.4 Figure export

Every final plot must be saved to the `figures/` directory with clear file names.

Required format:

```text
.png
```

Use readable labels:

- x-axis label
- y-axis label
- title
- legend if needed
- grid if helpful

### 6.5 Notebook rule

Notebooks are for exploration and demonstration. Reusable functions should be moved to `src/`.

Recommended pattern:

```text
src/ contains reusable functions
notebooks/ call functions from src/
results/ stores output CSV files
figures/ stores output images
```

---

## 7. Core Experiment Plan

## 7.1 Experiment A – Baseline Models

### Goal

Establish non-private performance.

### Models

```text
Logistic Regression
MLP non-DP
```

### Output

```text
results/baseline_results.csv
figures/baseline_confusion_matrix.png
```

### Expected analysis

- Logistic Regression is a simple interpretable baseline.
- MLP non-DP is the direct utility comparison for DP-SGD.
- Baseline MLP accuracy around 0.85 is acceptable.

---

## 7.2 Experiment B – Noise Multiplier Sweep

### Goal

Analyze privacy-utility tradeoff.

### Configuration

```text
noise_multiplier = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
max_grad_norm = 1.0
batch_size = 256
epochs = 20
delta = 1e-5
```

### Output

```text
results/dp_sgd_noise_results.csv
figures/noise_vs_epsilon.png
figures/noise_vs_accuracy.png
figures/noise_vs_f1.png
figures/privacy_utility_tradeoff.png
```

### Expected analysis

- Epsilon should decrease strongly as noise increases.
- Accuracy may decrease slightly.
- Noise 1.5 is likely the balanced configuration.
- Noise 3.0 is likely the strongest privacy configuration.

---

## 7.3 Experiment C – Max Grad Norm Sweep

### Goal

Analyze clipping effect.

### Configuration

```text
max_grad_norm = [0.5, 1.0, 1.5, 2.0]
noise_multiplier = 1.5
batch_size = 256
epochs = 20
delta = 1e-5
```

### Output

```text
results/max_grad_norm_results.csv
figures/max_grad_norm_vs_accuracy.png
figures/max_grad_norm_vs_f1.png
```

### Expected analysis

- Too small clipping may reduce learning signal.
- Too large clipping may reduce the stabilizing effect of clipping.
- The best value is empirical.

---

## 7.4 Experiment D – Batch Size Sweep

### Goal

Analyze effect of batch size on privacy and utility.

### Configuration

```text
batch_size = [64, 128, 256, 512]
noise_multiplier = 1.5
max_grad_norm = 1.0
epochs = 20
delta = 1e-5
```

### Output

```text
results/batch_size_results.csv
figures/batch_size_vs_epsilon.png
figures/batch_size_vs_accuracy.png
```

### Expected analysis

- Batch size changes privacy accounting.
- Batch size also affects optimization.
- There may be no universally best value.

---

## 7.5 Experiment E – Multiple Seed Evaluation

### Goal

Evaluate stability.

### Configuration

```text
seeds = [42, 123, 2026]
configs = [
    "MLP baseline",
    "DP-SGD noise 1.0",
    "DP-SGD noise 1.5",
    "DP-SGD noise 3.0"
]
```

### Output

```text
results/multi_seed_results.csv
figures/multi_seed_errorbar_accuracy.png
figures/multi_seed_errorbar_f1.png
```

### Expected analysis

- Report mean ± std.
- If standard deviation is small, results are stable.
- If standard deviation is large, mention DP training sensitivity.

---

## 7.6 Experiment F – Membership Inference Attack

### Goal

Evaluate privacy risk using a simple attack.

### Configuration

Compare:

```text
MLP baseline non-DP
DP-SGD noise 1.5
DP-SGD noise 3.0
```

### Output

```text
results/mia_results.csv
figures/confidence_distribution_baseline.png
figures/confidence_distribution_dp_sgd.png
figures/attack_auc_comparison.png
```

### Expected analysis

- Baseline may show higher confidence gap between train and test.
- DP-SGD may reduce confidence gap.
- Attack AUC closer to 0.5 indicates lower membership inference risk.
- If attack AUC is already close to 0.5 for baseline, explain that the model may not overfit much.

---

## 8. Final Report Structure

The final report should follow this structure.

```text
1. Introduction
   1.1. Model memorization and privacy risk
   1.2. Privacy in Development
   1.3. Project objectives
   1.4. Research questions

2. Background
   2.1. Differential Privacy
   2.2. Epsilon and delta
   2.3. DP-SGD
   2.4. Opacus
   2.5. Membership inference attack
   2.6. Source paper: Abadi et al.

3. Threat Model
   3.1. Attacker assumptions
   3.2. Protected asset
   3.3. DP-SGD defense scope
   3.4. Out-of-scope attacks

4. Dataset and Preprocessing
   4.1. UCI Adult Dataset
   4.2. Personal attributes
   4.3. Cleaning missing values
   4.4. Encoding and scaling
   4.5. Train/test split

5. Methodology
   5.1. Baseline Logistic Regression
   5.2. Baseline MLP
   5.3. DP-SGD MLP
   5.4. Noise multiplier sweep
   5.5. Max grad norm sweep
   5.6. Batch size sweep
   5.7. Multiple seed evaluation
   5.8. Membership inference attack

6. Results
   6.1. Baseline results
   6.2. Noise multiplier results
   6.3. Privacy-utility tradeoff
   6.4. Max grad norm results
   6.5. Batch size results
   6.6. Multiple seed stability
   6.7. Membership inference attack results

7. Discussion
   7.1. What configuration is best?
   7.2. Why noise reduces epsilon
   7.3. Why accuracy drops only slightly
   7.4. How DP-SGD changes privacy risk
   7.5. Comparison with Abadi et al.
   7.6. Practical meaning for Privacy in Development

8. Limitations
   8.1. Simple tabular dataset
   8.2. Simple MLP architecture
   8.3. Simple confidence-based MIA
   8.4. No model inversion attack
   8.5. Limited hardware and experiment budget

9. Future Work
   9.1. Stronger membership inference attacks
   9.2. Model inversion attack evaluation
   9.3. Fairness and privacy tradeoff
   9.4. Additional datasets
   9.5. Target epsilon training

10. Conclusion

11. References
```

---

## 9. Final Slide Structure

Use 16–18 slides.

```text
1. Title
2. Motivation: Why privacy in machine learning?
3. Problem: Model memorization
4. Differential Privacy overview
5. DP-SGD workflow
6. Source paper: Abadi et al.
7. Dataset: UCI Adult
8. Threat model
9. Experimental pipeline
10. Baseline models
11. Noise multiplier sweep
12. Privacy-utility tradeoff
13. Max grad norm analysis
14. Batch size analysis
15. Multiple seed robustness
16. Membership inference attack setup
17. Membership inference result
18. Recommended configuration and conclusion
```

Slide style:

- Academic and modern
- White background
- Blue accent
- Few words per slide
- Use diagrams and result tables
- Highlight key numbers
- Avoid long paragraphs

---

## 10. Demo Plan

The demo notebook should be short and safe.

### File

```text
notebooks/08_demo.ipynb
```

### Demo flow

```text
1. Load processed UCI Adult data
2. Show dataset shape and class distribution
3. Train or load baseline MLP result
4. Train or load DP-SGD noise 1.5 result
5. Print comparison table
6. Show privacy-utility graph
7. Show simple membership inference comparison
```

### If time is short during presentation

Do not retrain every model live. Load precomputed result CSV files and explain the pipeline.

### Demo messages to show

```text
Baseline MLP:
accuracy = 0.8501
f1 = 0.6567

DP-SGD balanced:
noise_multiplier = 1.5
epsilon = 1.0946
accuracy = 0.8464
f1 = 0.6544
accuracy_drop = 0.0036
```

Update these values if final reruns produce different results.

---

## 11. Defense Questions

### Q1. Why use UCI Adult instead of MNIST?

UCI Adult contains personal tabular attributes such as age, education, occupation, marital status, sex, hours per week, and income label. It better matches the Privacy in Development theme because it resembles personal records. MNIST is useful as a benchmark, but handwritten digits are less intuitive for discussing personal data privacy.

### Q2. What does epsilon mean?

Epsilon measures privacy loss. Smaller epsilon means stronger privacy because the model training output changes less when one individual record is added or removed from the training dataset.

### Q3. Why does increasing noise reduce epsilon?

DP-SGD adds Gaussian noise to clipped gradients. More noise makes it harder to infer the contribution of a single training record, so the privacy accountant reports a smaller epsilon.

### Q4. Why does accuracy not decrease much?

UCI Adult is a relatively simple tabular classification task. A small MLP can learn the main patterns without needing to memorize individual records. Therefore, moderate DP noise can provide privacy protection with only a small utility loss.

### Q5. Why choose noise multiplier 1.5?

Noise 0.5 gives high accuracy but weak privacy because epsilon is high. Noise 3.0 gives stronger privacy but slightly lower utility. Noise 1.5 provides a practical balance: epsilon around 1.09 with accuracy close to baseline.

### Q6. Does DP-SGD prevent all privacy attacks?

No. DP-SGD provides a formal Differential Privacy guarantee under its assumptions and parameter settings. This project also evaluates a simple membership inference attack, but it does not cover every possible attack such as full model inversion or advanced adaptive attacks.

### Q7. What is the role of Opacus?

Opacus provides DP-SGD tools for PyTorch, including per-example gradients, gradient clipping, Gaussian noise addition, and privacy accounting. It allows applying Differential Privacy to standard PyTorch models without writing the full DP training mechanism from scratch.

### Q8. What does membership inference attack AUC mean?

Attack AUC measures how well an attacker can distinguish training samples from non-training samples. AUC near 0.5 means random guessing. Higher AUC means higher membership leakage risk.

### Q9. Why run multiple seeds?

DP-SGD is stochastic because of random initialization, mini-batch sampling, and noise injection. Multiple seeds help verify that results are stable and not caused by a lucky run.

### Q10. What is the biggest limitation?

The project uses a simple confidence-based membership inference attack and a small MLP on one dataset. Stronger attacks, more datasets, and more architectures should be tested in future work.

---

## 12. Final Acceptance Criteria

### Code and data

- [ ] Raw UCI Adult data is stored in `data/raw/`.
- [ ] Processed arrays are stored in `data/processed/`.
- [ ] Preprocessing is reproducible.
- [ ] Baseline Logistic Regression runs.
- [ ] Baseline MLP runs.
- [ ] DP-SGD MLP runs with Opacus.
- [ ] All experiment notebooks run without manual edits.

### Experiments

- [ ] Noise multiplier sweep is complete.
- [ ] Max grad norm sweep is complete.
- [ ] Batch size sweep is complete.
- [ ] Multiple seed evaluation is complete.
- [ ] Membership inference attack is complete.

### Results

- [ ] All result CSV files are saved.
- [ ] All required figures are saved.
- [ ] Final summary table is prepared.
- [ ] Recommended configuration is clearly selected.

### Report

- [ ] Report includes research questions.
- [ ] Report includes threat model.
- [ ] Report includes theory section.
- [ ] Report includes methodology.
- [ ] Report includes all experiments.
- [ ] Report includes membership inference attack.
- [ ] Report includes limitations.
- [ ] Report includes conclusion.

### Slides

- [ ] Slides contain DP-SGD workflow.
- [ ] Slides contain dataset explanation.
- [ ] Slides contain main result table.
- [ ] Slides contain privacy-utility graph.
- [ ] Slides contain MIA result.
- [ ] Slides contain final recommendation.

### Presentation readiness

- [ ] Demo notebook is ready.
- [ ] Each member can explain their part.
- [ ] Defense questions are prepared.
- [ ] Numbers in report, slides, and CSV files are consistent.

---

## 13. Recommended Final Conclusion

Use or adapt this conclusion:

This project demonstrates that Differential Privacy can be integrated into a practical machine learning training pipeline using DP-SGD and Opacus. On the UCI Adult Dataset, increasing the noise multiplier substantially reduces the privacy budget epsilon while causing only a small decrease in classification performance. The balanced DP-SGD configuration with noise multiplier 1.5 achieves strong privacy improvement with accuracy close to the non-private MLP baseline. Additional hyperparameter analysis shows that clipping norm and batch size influence the privacy-utility tradeoff, while multiple-seed evaluation improves confidence in the results. Finally, a simple membership inference attack provides an intuitive privacy-risk evaluation by comparing model confidence on train and test samples. Overall, the project shows that DP-SGD is a feasible Privacy in Development technique, where privacy protection is embedded directly into model training rather than added after deployment.

---

## 14. Immediate Next Actions

Follow this exact order:

1. Create final folder structure.
2. Move current notebooks and report into the new structure.
3. Refactor preprocessing and model code into `src/`.
4. Rerun baseline models and save `baseline_results.csv`.
5. Rerun noise multiplier sweep and save `dp_sgd_noise_results.csv`.
6. Run max grad norm sweep.
7. Run batch size sweep.
8. Run multiple seed evaluation.
9. Run membership inference attack.
10. Generate all figures.
11. Write final report.
12. Create final slides.
13. Prepare demo notebook.
14. Cross-check all result numbers.
15. Practice defense answers.
