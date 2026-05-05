# Data Directory

This directory stores local data for the project.

- `raw/`: Original videos, chat logs, FER2013 files, and annotation files.
- `interim/`: Intermediate extracted frames, audio chunks, transcripts, and temporary features.
- `processed/`: Cleaned datasets, annotation metadata, and synchronized modality features.
- `external/`: Third-party datasets or metadata that should remain separate from generated project data.

Large datasets, videos, and audio files should not be committed to git.

## Annotation Templates

Tiny CSV templates for selected VOD metadata and human highlight annotations live in:

```text
data/processed/annotations/video_registry_template.csv
data/processed/annotations/human_highlights_template.csv
```

Copy these locally to `video_registry.csv` and `human_highlights.csv` when recording real project VODs and annotations. The real CSVs are local artifacts and should not be committed by default.
