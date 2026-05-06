# Data Sources

Track datasets, videos, chat logs, annotation files, and external resources used
in the project.

| Source | Type | Location |
| --- | --- | --- |
| FER2013 | Facial emotion dataset | `data/raw/fer2013/` |
| Project VODs | Video metadata | `data/processed/annotations/video_registry.csv` |
| Human annotations | Highlight labels | `data/processed/annotations/human_highlights.csv` |

## Source Notes

FER2013:

- Raw files are local and ignored by git.
- The dataset was used to train the baseline and improved CNN visual emotion
  models.
- The selected model is documented in `docs/model_cards.md`.

Project VODs:

- Record selected long videos in `video_registry.csv`.
- Start from `data/processed/annotations/video_registry_template.csv`.
- Respect source-platform terms when using any VOD URL or metadata.

Human annotations:

- Record ground-truth highlight windows in `human_highlights.csv`.
- Start from `data/processed/annotations/human_highlights_template.csv`.
- Treat annotations as course/project artifacts unless another license is
  explicitly documented.

## Selected VOD Registry Format

Each selected VOD should have one row in `video_registry.csv` with these
required columns:

```text
video_id,title,source_url,platform,duration_seconds,genre,has_chat,has_visible_face,notes
```

`video_id` is the stable key used to connect video metadata, human annotations,
future modality features, and ranked highlight outputs.

`duration_seconds` should be the full VOD duration in seconds. `has_chat` and
`has_visible_face` should be boolean-like values such as `true` or `false`.

## Human Highlight Annotation Format

Each human-labeled highlight should have one row in `human_highlights.csv` with
these required columns:

```text
video_id,start_time,end_time,start_seconds,end_seconds,annotator,reason,confidence
```

Timestamps should use `HH:MM:SS`, and the seconds columns should match those
timestamps. `confidence` should be a value from 0 to 1.

The annotation loader can optionally validate that all `video_id` values exist
in the VOD registry and that annotations do not exceed the recorded video
duration.

Real VOD URLs, annotation rows, generated datasets, processed arrays,
checkpoints, output metrics, and output figures are local artifacts. They should
not be committed unless intentionally selected later for the final report or
presentation.
