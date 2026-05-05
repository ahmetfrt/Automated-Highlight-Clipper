# Models Directory

Store trained model checkpoints, exported models, and model metadata here.

Large model files are ignored by git. Add lightweight notes or model cards to `docs/model_cards.md` when a model becomes part of an experiment.

## Current FER2013 Selection

The current selected visual emotion model is the improved FER2013 CNN:

```text
models/checkpoints/fer2013_improved_cnn.pt
```

This checkpoint is a local artifact and should not be committed by default. The model card and selection rationale are tracked in `docs/model_cards.md`, and the experiment summary is tracked in `docs/experiment_log.md`.
