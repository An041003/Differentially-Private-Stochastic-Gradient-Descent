# Best variant 50-epoch noise x clipping notes

- Device: `cuda`.
- Model variant: `mlp_64_32_d0p1`.
- Hidden dims: `(64, 32)`.
- Dropout: `0.1`.
- Epochs: `50`.
- Early stopping: disabled; every run uses the full 50 epochs.
- Completed runs: `36`.
- Reference non-DP F1-score: `0.7273`.

## Best and worst

- Best F1: noise `0.5`, clip `3.0`, epsilon `26.7664`, F1 `0.6761`.
- Worst F1: noise `10.0`, clip `3.0`, epsilon `0.2245`, F1 `0.2501`.

## Best per noise

| noise | best_clip | epsilon | F1-score | precision | recall | accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 3.0 | 26.7664 | 0.6761 | 0.7362 | 0.6251 | 0.8529 |
| 1.0 | 1.0 | 3.8217 | 0.6643 | 0.7515 | 0.5951 | 0.8522 |
| 2.0 | 0.5 | 1.3568 | 0.6515 | 0.7577 | 0.5714 | 0.8498 |
| 3.0 | 0.5 | 0.8348 | 0.6424 | 0.7553 | 0.5589 | 0.8471 |
| 5.0 | 0.5 | 0.4713 | 0.6400 | 0.7554 | 0.5551 | 0.8465 |
| 8.0 | 0.5 | 0.2841 | 0.6301 | 0.7340 | 0.5519 | 0.8408 |
| 10.0 | 0.5 | 0.2245 | 0.6307 | 0.7051 | 0.5705 | 0.8359 |
| 15.0 | 0.5 | 0.1475 | 0.6086 | 0.5799 | 0.6403 | 0.7977 |
| 20.0 | 0.5 | 0.1103 | 0.5558 | 0.5274 | 0.5873 | 0.7693 |
