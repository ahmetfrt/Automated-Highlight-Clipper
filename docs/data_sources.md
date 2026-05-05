# Data Sources

Track datasets, videos, chat logs, annotation files, and external resources used in the project.

| Source | Type | Location | License/Terms | Notes |
| --- | --- | --- | --- | --- |
| FER2013 | Facial emotion dataset | `data/raw/fer2013/` (local, ignored by git) | TBD | Used to train the baseline and improved CNN visual emotion models. The selected model is documented in `docs/model_cards.md`. |
| Project VODs | Video data | TBD | TBD | Long videos used for highlight extraction experiments. |
| Human annotations | Labels | TBD | TBD | Ground-truth highlight windows for evaluation. |

FER2013 raw files, processed arrays, and any derived local training artifacts should remain uncommitted unless a small, intentional artifact is later selected for reporting.
