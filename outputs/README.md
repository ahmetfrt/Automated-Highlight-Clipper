# Outputs Directory

This directory stores generated project artifacts.

- `predictions/`: Ranked highlight JSON files and modality scores.
- `metrics/`: Evaluation metrics.
- `figures/`: Generated plots.
- `tables/`: Ablation tables and summary CSV files.
- `user_study/`: Human annotation forms or aggregated study results.

Generated outputs are ignored by git except for placeholder files.

## FER2013 Outputs

The FER2013 training runs produced local metrics and figures under:

```text
outputs/metrics/
outputs/figures/
```

The selected visual model is documented in `docs/model_cards.md`. Generated metrics, figures, processed datasets, and checkpoints are local artifacts and should not be committed unless intentionally selected later for the final report or presentation.
