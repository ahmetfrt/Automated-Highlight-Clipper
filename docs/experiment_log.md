# Experiment Log

| Date | Experiment ID | Pipeline | Config | Dataset/Input | Metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- |

## FER2013 Visual Emotion Experiments

The following entries summarize the real FER2013 CNN runs found in local training artifacts. No new training was run for this documentation update.

| Date | Run ID | Model Name | Dataset | Epochs | Batch Size | Accuracy | Best Validation Accuracy | Macro Precision | Macro Recall | Macro F1 | Parameter Count | Training Time (seconds) | Output Metrics Path | Figure Paths | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-05 | fer2013_baseline_cnn_2026-05-05_235630 | baseline CNN | FER2013 (image_folders) | 30 | 128 | 0.5295346893285038 | 0.5368005572324124 (epoch 9) | 0.562241631858799 | 0.47130186356489917 | 0.4895215502192262 | 1199495 | 366.51164689997677 | `outputs/metrics/fer2013_baseline_cnn_metrics.json`; `outputs/metrics/fer2013_baseline_cnn_history.csv` | `outputs/figures/fer2013_baseline_cnn_training_curves.png`; `outputs/figures/fer2013_baseline_cnn_confusion_matrix.png` | Real run; `is_smoke_test=false`; no data augmentation or early stopping. Final validation accuracy was 0.5191548641745994. Checkpoint: `models/checkpoints/fer2013_baseline_cnn.pt`. |
| 2026-05-06 | fer2013_improved_cnn_2026-05-06_002358 | improved CNN | FER2013 (image_folders) | 30 | 128 | 0.6457230426302591 | 0.6477826793591828 (epoch 29) | 0.6456137436609625 | 0.5788930363572949 | 0.5891470970491275 | 1469095 | 1638.9081575999735 | `outputs/metrics/fer2013_improved_cnn_metrics.json`; `outputs/metrics/fer2013_improved_cnn_history.csv` | `outputs/figures/fer2013_improved_cnn_training_curves.png`; `outputs/figures/fer2013_improved_cnn_confusion_matrix.png` | Real run; `is_smoke_test=false`; uses data augmentation, batch normalization, dropout, and early stopping logic. Final validation accuracy was 0.6436034362665428. Checkpoint: `models/checkpoints/fer2013_improved_cnn.pt`. |

Current selected visual model: **improved CNN**. Validation macro F1 was not available in the saved artifacts, so the selection rule falls back to test macro F1. The improved CNN is selected because its test macro F1 is 0.5891470970491275, higher than the baseline CNN test macro F1 of 0.4895215502192262. It also has higher test accuracy (0.6457230426302591 vs. 0.5295346893285038) and higher best validation accuracy (0.6477826793591828 vs. 0.5368005572324124).

Generated artifacts such as datasets, processed arrays, checkpoints, model files, output metrics, and output figures are local artifacts. They should not be committed unless a specific artifact is intentionally selected later for the final report or presentation.
