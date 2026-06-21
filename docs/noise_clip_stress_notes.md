# Noise x clipping stress-test notes

- Device: `cuda`.
- Grid: noise multiplier `[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]` x max grad norm `[0.5, 1.0, 2.0, 3.0]`.
- Main metric: F1-score. Accuracy is secondary because UCI Adult is imbalanced.
- Reference non-DP MLP F1-score: `0.6951`. This prefers `results/best_tuned_baseline.json` when available.

## Best and worst observed configurations

- Best F1: noise `0.5`, clip `3.0`, epsilon `16.8660`, F1 `0.6667`.
- Worst F1: noise `5.0`, clip `3.0`, epsilon `0.2908`, F1 `0.4325`.

## Interpretation

- The earlier noise range up to 3.0 tested the practical privacy-utility region.
- This stress test adds much larger noise values to check whether utility collapse appears only under extreme privacy settings.
- Best-per-noise F1 first drops by at least 0.05 below the tuned non-DP baseline at noise `3.0`.
- Use `figures/noise_clip_f1_heatmap.png`, `figures/noise_clip_f1_lines.png`, and `figures/extreme_noise_best_f1.png` for the final discussion.

## Best F1 per noise

| noise_multiplier | best_clip | epsilon | F1-score | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 3.0 | 16.8660 | 0.6667 | 0.7543 | 0.5973 |
| 1.0 | 1.0 | 2.3534 | 0.6605 | 0.7594 | 0.5843 |
| 2.0 | 2.0 | 0.8306 | 0.6479 | 0.7485 | 0.5711 |
| 3.0 | 0.5 | 0.5124 | 0.6387 | 0.7568 | 0.5524 |
| 5.0 | 1.0 | 0.2908 | 0.6332 | 0.7072 | 0.5732 |
| 8.0 | 0.5 | 0.1766 | 0.6148 | 0.7155 | 0.5389 |
| 10.0 | 0.5 | 0.1403 | 0.6070 | 0.6876 | 0.5432 |
| 15.0 | 0.5 | 0.0934 | 0.5681 | 0.6154 | 0.5276 |
| 20.0 | 0.5 | 0.0707 | 0.5135 | 0.5632 | 0.4719 |
