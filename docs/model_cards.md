# Model Cards

Use this document to summarize trained models used in the project.

## Selected FER2013 Visual Emotion Model

### FER2013 Improved CNN

- **Version:** `fer2013_improved_cnn_2026-05-06_002358`
- **Training dataset:** FER2013
- **Dataset format used in this run:** `image_folders`
- **Task:** Facial emotion recognition for sampled face crops.
- **Input format:** 48x48 grayscale face crop.
- **Output classes:** angry, disgust, fear, happy, sad, surprise, neutral.
- **Architecture summary:** Three convolutional feature blocks with batch normalization, ReLU activations, max pooling, and dropout, followed by a dense classifier with batch normalization and dropout. The model has 1469095 trainable parameters.
- **Selected checkpoint/model path:** `models/checkpoints/fer2013_improved_cnn.pt`
- **Metric files:** `outputs/metrics/fer2013_improved_cnn_metrics.json`; `outputs/metrics/fer2013_improved_cnn_history.csv`
- **Figure files:** `outputs/figures/fer2013_improved_cnn_training_curves.png`; `outputs/figures/fer2013_improved_cnn_confusion_matrix.png`
- **Best validation accuracy available:** 0.6477826793591828 at epoch 29.
- **Final validation accuracy available:** 0.6436034362665428.
- **Test accuracy:** 0.6457230426302591.
- **Macro precision:** 0.6456137436609625.
- **Macro recall:** 0.5788930363572949.
- **Macro F1:** 0.5891470970491275.
- **Known weaknesses:** FER2013 is a small, class-imbalanced facial-expression dataset, so minority or visually ambiguous classes may be weaker. In the saved test classification report, disgust has recall 0.2702702702702703 and fear has recall 0.3203125, indicating weaker performance on those classes compared with happy and surprise.
- **Intended use in the project:** Use as the current visual emotion model for the next visual-pipeline phase, where sampled video face crops will be converted into frame-level emotion predictions and later visual excitement features.
- **Why selected over the other FER model:** Validation macro F1 was not available, so the project selection rule falls back to test macro F1. The improved CNN achieved test macro F1 0.5891470970491275, compared with 0.4895215502192262 for the baseline CNN. The improved CNN also had higher test accuracy and best validation accuracy.
- **Artifact commit note:** Generated datasets, processed arrays, checkpoints, trained model files, output metrics, and figures are local artifacts and should not be committed unless intentionally selected later for the final report or presentation.

## Template

### Model Name

- **Version:** TBD
- **Training data:** TBD
- **Task:** TBD
- **Inputs:** TBD
- **Outputs:** TBD
- **Metrics:** TBD
- **Limitations:** TBD
- **Intended use:** CS 466 project experiments only.
