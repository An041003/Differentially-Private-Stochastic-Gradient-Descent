# Model x clipping x noise grid notes

- Device: `cuda`.
- Completed runs: `144`.
- Reference non-DP F1-score: `0.7273`.
- Grid: 4 MLP model variants x 4 clipping norms x 9 noise multipliers.

## Best and worst

- Best DP F1: model `mlp_64_32_d0p1`, noise `0.5`, clip `1.0`, epsilon `16.8660`, F1 `0.6760`.
- Worst DP F1: model `mlp_64_32_d0p1`, noise `15.0`, clip `3.0`, epsilon `0.0934`, F1 `0.2413`.

## Best per model

| model_variant | noise | clip | epsilon | F1-score | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| mlp_64_32_d0p1 | 0.5 | 1.0 | 16.8660 | 0.6760 | 0.7367 | 0.6246 |
| mlp_128_64_d0p0 | 0.5 | 3.0 | 16.8660 | 0.6718 | 0.7356 | 0.6181 |
| mlp_256_128_d0p1 | 0.5 | 1.0 | 16.8660 | 0.6716 | 0.7483 | 0.6092 |
| mlp_128_64_d0p1 | 0.5 | 3.0 | 16.8660 | 0.6667 | 0.7543 | 0.5973 |

## Best per noise

| noise | model_variant | clip | epsilon | F1-score |
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
