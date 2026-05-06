# Experiment Log

| Date | Experiment ID | Pipeline | Config | Dataset/Input | Metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- |

## FER2013 Visual Emotion Experiments

The following entries summarize the real FER2013 CNN runs found in local
training artifacts. No new training was run for this documentation update.

### Run Summary

| Field | Baseline CNN | Improved CNN |
| --- | --- | --- |
| Date | 2026-05-05 | 2026-05-06 |
| Run ID | `fer2013_baseline_cnn_2026-05-05_235630` | `fer2013_improved_cnn_2026-05-06_002358` |
| Dataset | FER2013 (`image_folders`) | FER2013 (`image_folders`) |
| Epochs | 30 | 30 |
| Batch size | 128 | 128 |
| Parameter count | 1199495 | 1469095 |
| Training time (seconds) | 366.51164689997677 | 1638.9081575999735 |
| Best validation accuracy | 0.5368005572324124 (epoch 9) | 0.6477826793591828 (epoch 29) |
| Final validation accuracy | 0.5191548641745994 | 0.6436034362665428 |

### Test Metrics

| Metric | Baseline CNN | Improved CNN |
| --- | --- | --- |
| Accuracy | 0.5295346893285038 | 0.6457230426302591 |
| Macro precision | 0.562241631858799 | 0.6456137436609625 |
| Macro recall | 0.47130186356489917 | 0.5788930363572949 |
| Macro F1 | 0.4895215502192262 | 0.5891470970491275 |

### Artifact Paths

Baseline CNN:

- Metrics: `outputs/metrics/fer2013_baseline_cnn_metrics.json`
- History: `outputs/metrics/fer2013_baseline_cnn_history.csv`
- Training curve: `outputs/figures/fer2013_baseline_cnn_training_curves.png`
- Confusion matrix: `outputs/figures/fer2013_baseline_cnn_confusion_matrix.png`
- Checkpoint: `models/checkpoints/fer2013_baseline_cnn.pt`
- Notes: real run; `is_smoke_test=false`; no data augmentation or early stopping.

Improved CNN:

- Metrics: `outputs/metrics/fer2013_improved_cnn_metrics.json`
- History: `outputs/metrics/fer2013_improved_cnn_history.csv`
- Training curve: `outputs/figures/fer2013_improved_cnn_training_curves.png`
- Confusion matrix: `outputs/figures/fer2013_improved_cnn_confusion_matrix.png`
- Checkpoint: `models/checkpoints/fer2013_improved_cnn.pt`
- Notes: real run; `is_smoke_test=false`; uses data augmentation, batch
  normalization, dropout, and early stopping logic.

Current selected visual model: **improved CNN**.

Validation macro F1 was not available in the saved artifacts, so the selection
rule falls back to test macro F1. The improved CNN is selected because its test
macro F1 is 0.5891470970491275, higher than the baseline CNN test macro F1 of
0.4895215502192262. It also has higher test accuracy and higher best validation
accuracy.

Generated artifacts such as datasets, processed arrays, checkpoints, model
files, output metrics, and output figures are local artifacts. They should not
be committed unless a specific artifact is intentionally selected later for the
final report or presentation.
