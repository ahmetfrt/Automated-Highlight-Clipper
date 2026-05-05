# Project Plan

## Milestone 1: Repository and Baselines

- Status: completed.
- Created a stable repository structure.
- Defined configuration files and data paths.
- Added placeholder scripts and modules for each pipeline stage.
- Implement simple keyword and metadata baselines before adding heavy models.

## Milestone 2: Visual Pipeline

- Status: in progress.
- FER2013 preparation and CNN training are complete for the baseline and improved CNN runs.
- The current selected FER2013 visual emotion model is the improved CNN checkpoint at `models/checkpoints/fer2013_improved_cnn.pt`.
- Selection was based on saved test macro F1 because validation macro F1 was not available: improved CNN `0.5891470970491275` vs. baseline CNN `0.4895215502192262`.
- VOD metadata and human highlight annotation utilities are now in place. Selected VODs should be recorded in `data/processed/annotations/video_registry.csv`, and human highlight labels should be recorded in `data/processed/annotations/human_highlights.csv`.
- Export visual emotion predictions for sampled video frames.
- Convert frame-level predictions into visual excitement features.

## Milestone 3: Audio Pipeline

- Extract audio tracks from source videos.
- Split audio into 30-second chunks.
- Transcribe chunks with Whisper.
- Build audio and transcript excitement features.

## Milestone 4: Text and Chat Pipeline

- Parse chat logs, OCR text, or transcript windows.
- Implement keyword-based text excitement scoring.
- Optionally compare against an LLM-based scoring approach.

## Milestone 5: Fusion and Evaluation

- Synchronize all modality outputs into 60-second windows.
- Run visual-only, audio-only, text-only, and full late-fusion systems.
- Generate ranked highlight JSON outputs.
- Evaluate against human annotations.
- Produce ablation tables, plots, and report figures.

## Milestone 6: Final Deliverables

- Complete final report.
- Prepare presentation materials.
- Document AI tool usage.
- Archive reproducible experiment settings and results.

## Artifact Policy

Generated datasets, processed arrays, checkpoints, model files, output metrics, and output figures are local artifacts and should not be committed unless a specific artifact is intentionally selected later for the final report or presentation.
