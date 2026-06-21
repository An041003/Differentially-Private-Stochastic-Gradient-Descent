# Privacy-Preserving Data with Differential Privacy using DP-SGD

## 1. Introduction

This project studies whether DP-SGD can be integrated into a normal PyTorch training pipeline to reduce privacy risk while preserving useful performance on the UCI Adult personal tabular dataset.

## 2. Threat Model

The assumed attacker can query or inspect a trained model and may try to infer whether a specific person's record was part of the training data. DP-SGD limits the influence of any individual record through per-example gradient clipping, Gaussian noise, and privacy accounting. The project also evaluates a simple confidence-based membership inference attack, but it does not claim protection against every possible privacy attack.

## 3. Dataset and Method

- Dataset: UCI Adult / Census Income.
- Task: binary classification, `>50K` vs `<=50K`.
- Split: original `adult.data` train file and `adult.test` test file.
- Preprocessing: drop rows with `?`, standardize numeric columns, one-hot encode categorical columns.
- Models: Logistic Regression baseline, non-DP MLP, and DP-SGD MLP with Opacus.
- Main DP parameters: `delta = 1e-5`, `epochs = 20`, `learning_rate = 0.05`, `momentum = 0.9`.

## 4. Baseline Results

| model_name | accuracy | precision | recall | f1_score | roc_auc | pr_auc | training_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.8480 | 0.7309 | 0.6035 | 0.6611 | 0.9033 | 0.7662 | 0.3174 |
| mlp_baseline | 0.8487 | 0.7297 | 0.6100 | 0.6645 | 0.9033 | 0.7717 | 20.8721 |

## 5. Noise Multiplier Sweep

| noise_multiplier | epsilon | accuracy | precision | recall | f1_score | roc_auc | pr_auc | training_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 16.8660 | 0.8525 | 0.7492 | 0.6008 | 0.6669 | 0.9078 | 0.7781 | 58.8097 |
| 0.8000 | 3.9063 | 0.8525 | 0.7529 | 0.5946 | 0.6645 | 0.9070 | 0.7771 | 69.1811 |
| 1.0000 | 2.3534 | 0.8518 | 0.7535 | 0.5897 | 0.6616 | 0.9064 | 0.7760 | 72.8704 |
| 1.5000 | 1.2143 | 0.8497 | 0.7474 | 0.5862 | 0.6571 | 0.9047 | 0.7728 | 66.6714 |
| 2.0000 | 0.8306 | 0.8485 | 0.7431 | 0.5862 | 0.6554 | 0.9029 | 0.7688 | 64.6728 |
| 3.0000 | 0.5124 | 0.8465 | 0.7352 | 0.5868 | 0.6526 | 0.8984 | 0.7588 | 66.0001 |

The balanced configuration remains `noise_multiplier = 1.5`: epsilon = 1.2143, accuracy = 0.8497, and F1-score = 0.6571. The strongest formal privacy setting in this sweep is `noise_multiplier = 3.0` with epsilon = 0.5124.


## 5.1 Strong Non-DP Baseline

After tuning the simple MLP baseline, a stronger classical baseline was tested to check whether the project had an under-powered non-DP reference. The strongest clean local result came from HistGradientBoosting trained on the full Adult training file, using a threshold selected from an earlier training-validation split.

| model | default accuracy | default F1 | tuned accuracy | tuned F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| hgb_full_train_lr0.08_leaf15_l20.01 | 0.8708 | 0.7154 | 0.8586 | 0.7261 | 0.9266 | 0.8288 |

This is now the strongest clean non-DP reference. It improves the baseline, but it still does not reach 0.90 accuracy or 0.90 F1 on the original Adult test split.


## 5.2 Boosted Feature Baseline

XGBoost, LightGBM, CatBoost, and HistGradientBoosting were tested with additional binning and interaction features. Thresholds were selected on a training-validation split, then the selected model was evaluated on the held-out Adult test file.

| model | default accuracy | default F1 | tuned accuracy | tuned F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| hgb_fe_lr0.03 | 0.8689 | 0.7128 | 0.8541 | 0.7273 | 0.9259 | 0.8278 |

The best boosted/feature-engineered baseline still does not reach 0.90 accuracy or 0.90 F1, so a 0.90 target should be treated as a stretch goal or leak-check trigger.


## 6. Max Grad Norm Sweep

| max_grad_norm | epsilon | accuracy | precision | recall | f1_score | roc_auc | pr_auc | training_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 1.2143 | 0.8491 | 0.7521 | 0.5757 | 0.6522 | 0.9057 | 0.7730 | 64.7887 |
| 1.0000 | 1.2143 | 0.8497 | 0.7474 | 0.5862 | 0.6571 | 0.9047 | 0.7728 | 67.7063 |
| 1.5000 | 1.2143 | 0.8491 | 0.7425 | 0.5908 | 0.6580 | 0.9032 | 0.7702 | 68.0225 |
| 2.0000 | 1.2143 | 0.8493 | 0.7372 | 0.6005 | 0.6619 | 0.9019 | 0.7681 | 66.4494 |

With fixed noise multiplier, sample rate, epochs, and delta, epsilon is unchanged across clipping norms. The best F1-score in this run is obtained at `max_grad_norm = 2.0`.

## 7. Batch Size Sweep

| batch_size | epsilon | accuracy | precision | recall | f1_score | roc_auc | pr_auc | training_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.5645 | 0.7934 | 0.7732 | 0.2249 | 0.3484 | 0.8571 | 0.6556 | 78.1864 |
| 128 | 0.8254 | 0.8503 | 0.7723 | 0.5538 | 0.6450 | 0.8990 | 0.7656 | 64.5831 |
| 256 | 1.2143 | 0.8497 | 0.7474 | 0.5862 | 0.6571 | 0.9047 | 0.7728 | 64.2887 |
| 512 | 1.7991 | 0.8514 | 0.7515 | 0.5903 | 0.6612 | 0.9064 | 0.7738 | 54.0743 |

Batch size changes privacy accounting through the sampling rate. In this run, batch size 64 gives the smallest epsilon but has much weaker F1-score, while batch size 512 has stronger utility but a larger epsilon.

## 8. Multi-Seed Robustness

| model_config | mean_accuracy | std_accuracy | mean_f1_score | std_f1_score | mean_roc_auc | std_roc_auc | mean_epsilon | std_epsilon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dp_noise_1.0 | 0.8517 | 0.0010 | 0.6655 | 0.0074 | 0.9061 | 0.0002 | 2.3534 | 0.0000 |
| dp_noise_1.5 | 0.8505 | 0.0008 | 0.6620 | 0.0101 | 0.9046 | 0.0006 | 1.2143 | 0.0000 |
| dp_noise_3.0 | 0.8458 | 0.0008 | 0.6513 | 0.0154 | 0.8969 | 0.0019 | 0.5124 | 0.0000 |
| mlp_baseline | 0.8491 | 0.0011 | 0.6694 | 0.0053 | 0.9044 | 0.0015 | N/A | N/A |

The standard deviations are small, which suggests that the main results are reasonably stable across the selected seeds.

## 9. Membership Inference Attack

| model_name | epsilon | attack_auc | mean_train_confidence | mean_test_confidence | confidence_gap |
| --- | --- | --- | --- | --- | --- |
| mlp_baseline | N/A | 0.5021 | 0.8668 | 0.8652 | 0.0016 |
| dp_sgd_noise_1.5 | 1.2143 | 0.5021 | 0.9639 | 0.9643 | -0.0004 |
| dp_sgd_noise_3.0 | 0.5124 | 0.5016 | 0.9625 | 0.9621 | 0.0004 |

All attack AUC values are close to 0.5. This means the simple confidence-based attack is close to random guessing in this setup. The result should be interpreted carefully: it is an empirical attack check, while epsilon remains the formal privacy metric.


## 9. GPU Tuning Result

The tuning run used the CUDA PyTorch environment on the NVIDIA T1200 Laptop GPU and focused on improving DP-SGD with `noise_multiplier = 1.5`. The best configuration found was:

| max_grad_norm | batch_size | epochs | learning_rate | schedule | epsilon | accuracy | F1-score |
|---:|---:|---:|---:|---|---:|---:|---:|
| 2.0000 | 512 | 20 | 0.0800 | cosine | 1.7991 | 0.8531 | 0.6680 |

The search stopped because later attempts changed F1 by less than the predefined meaningful threshold or required a larger epsilon without a clear utility gain. Detailed configuration notes are stored in `docs/tuning_notes.md`.


## 10. Noise x Clipping Stress Test

This additional GPU experiment was added after feedback that the original noise range may not be large enough to show clear utility collapse. It uses F1-score as the primary metric and evaluates a joint grid of `noise_multiplier x max_grad_norm`. The comparison reference is the tuned non-DP MLP baseline when available, so low-noise DP is not allowed to look better only because the original baseline was under-tuned.

| noise_multiplier | max_grad_norm | epsilon | precision | recall | f1_score |
| --- | --- | --- | --- | --- | --- |
| 0.5000 | 3.0000 | 16.8660 | 0.7543 | 0.5973 | 0.6667 |
| 1.0000 | 1.0000 | 2.3534 | 0.7594 | 0.5843 | 0.6605 |
| 2.0000 | 2.0000 | 0.8306 | 0.7485 | 0.5711 | 0.6479 |
| 3.0000 | 0.5000 | 0.5124 | 0.7568 | 0.5524 | 0.6387 |
| 5.0000 | 1.0000 | 0.2908 | 0.7072 | 0.5732 | 0.6332 |
| 8.0000 | 0.5000 | 0.1766 | 0.7155 | 0.5389 | 0.6148 |
| 10.0000 | 0.5000 | 0.1403 | 0.6876 | 0.5432 | 0.6070 |
| 15.0000 | 0.5000 | 0.0934 | 0.6154 | 0.5276 | 0.5681 |
| 20.0000 | 0.5000 | 0.0707 | 0.5632 | 0.4719 | 0.5135 |

Tuned non-DP baseline F1: 0.6951. The best DP result in the grid is below this tuned baseline. Best-per-noise F1 first drops by at least 0.05 below tuned baseline at noise `3.0`, and larger stress-test noise values such as 10.0, 15.0, and 20.0 show a much clearer utility drop. Detailed notes are stored in `docs/noise_clip_stress_notes.md`.


## 10.1 Model x Clipping x Noise Grid

The full DP grid evaluates four MLP variants across the same clipping and noise ranges, for 144 GPU runs.

| noise_multiplier | model_variant | max_grad_norm | epsilon | F1-score |
|---:|---|---:|---:|---:|
| 0.5 | mlp_64_32_d0p1 | 1.0 | 16.8660 | 0.6760 |
| 1.0 | mlp_64_32_d0p1 | 2.0 | 2.3534 | 0.6716 |
| 2.0 | mlp_64_32_d0p1 | 1.0 | 0.8306 | 0.6625 |
| 3.0 | mlp_64_32_d0p1 | 1.0 | 0.5124 | 0.6556 |
| 5.0 | mlp_128_64_d0p0 | 0.5 | 0.2908 | 0.6430 |
| 8.0 | mlp_128_64_d0p0 | 0.5 | 0.1766 | 0.6345 |
| 10.0 | mlp_64_32_d0p1 | 0.5 | 0.1403 | 0.6252 |
| 15.0 | mlp_64_32_d0p1 | 0.5 | 0.0934 | 0.5893 |
| 20.0 | mlp_256_128_d0p1 | 0.5 | 0.0707 | 0.5469 |

Best DP grid result: `mlp_64_32_d0p1`, noise `0.5`, clip `1.0`, F1 `0.6760`. Worst grid result: `mlp_64_32_d0p1`, noise `15.0`, clip `3.0`, F1 `0.2413`.


## 11. Comparison with Abadi et al.

| Criterion | Abadi et al. | This project |
|---|---|---|
| Dataset | MNIST, CIFAR-10 | UCI Adult |
| Data type | Images | Personal tabular data |
| Model | Neural network / CNN | Logistic Regression and MLP |
| Privacy mechanism | DP-SGD | DP-SGD through Opacus |
| Privacy accounting | Moments Accountant | Opacus PrivacyEngine |
| Analysis axis | Epsilon, delta, accuracy | Epsilon, utility metrics, training time, MIA AUC |

## 12. Limitations

- The membership inference attack is simple and confidence-based.
- The project uses one tabular dataset and a small MLP.
- The final stress-test and tuning runs use the local NVIDIA T1200 GPU, but the project is still limited by laptop hardware.
- Secure RNG was off in Opacus for faster experimentation; production-grade privacy training should enable secure mode.

## 13. Conclusion

The final pipeline shows that DP-SGD can be integrated into model training for a personal tabular dataset. Increasing noise reduces epsilon, and the new stress test shows that F1-score drops clearly once the noise multiplier is pushed into much larger values. The balanced setting `noise_multiplier = 1.5` remains a useful privacy-utility compromise, while extreme noise settings are useful for demonstrating utility collapse. The membership inference attack results show low confidence-based membership signal in this particular experiment.
