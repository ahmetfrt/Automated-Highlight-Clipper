# Project Plan

## Milestone 1: Repository and Baselines

- Create a stable repository structure.
- Define configuration files and data paths.
- Add placeholder scripts and modules for each pipeline stage.
- Implement simple keyword and metadata baselines before adding heavy models.

## Milestone 2: Visual Pipeline

- Explore FER2013.
- Train a custom CNN for facial emotion recognition.
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
