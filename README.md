# The Automated Highlight Clipper

The Automated Highlight Clipper is a CS 466 Introduction to Deep Learning term project for building a multimodal highlight-detection pipeline. Given a long VOD or video, the planned system will score candidate windows and output ranked 60-second highlight segments.

The repository now includes the FER2013 facial emotion preparation and CNN training pipeline. The selected visual emotion model is documented for later use in the video visual pipeline, while video, audio, text, fusion, and highlight evaluation remain future milestones.

## Planned Pipeline

1. **Visual pipeline**
   - Train a custom CNN on FER2013 for facial emotion recognition.
   - Sample frames from VODs.
   - Convert facial emotion predictions into visual excitement scores.

2. **Audio pipeline**
   - Extract audio from videos.
   - Split audio into 30-second chunks.
   - Transcribe chunks with Whisper.
   - Build excitement features from audio and transcripts.

3. **Text/chat pipeline**
   - Process chat logs, OCR text, and transcript windows.
   - Score text excitement using a keyword baseline.
   - Optionally compare against an LLM-based scorer.

4. **Fusion and evaluation**
   - Synchronize modality outputs into 60-second windows.
   - Run visual-only, audio-only, text-only, and multimodal late-fusion scoring.
   - Output ranked highlight JSON files.
   - Evaluate against human annotations with ablation tables and plots.

## Repository Layout

```text
config/        Project configuration files.
data/          Local datasets and generated data artifacts.
docs/          Planning notes, data-source notes, and experiment records.
models/        Trained checkpoints and model metadata.
notebooks/     Exploration, training, demo, and evaluation notebooks.
outputs/       Predictions, metrics, tables, figures, and user-study outputs.
report/        Final report draft, references, and report figures.
presentation/  Presentation assets.
scripts/       Command-line entry points for each project stage.
src/           Python package containing placeholder modules.
```

## Setup

Create an environment with either `pip`:

```bash
pip install -r requirements.txt
```

or Conda:

```bash
conda env create -f environment.yml
conda activate automated-highlight-clipper
```

Copy `.env.example` to `.env` before adding local paths or optional API keys. Do not commit `.env`, datasets, videos, audio files, model checkpoints, or generated outputs.

## FER2013 Training

Place FER2013 under `data/raw/fer2013/`. The preparation script supports the common Kaggle `fer2013.csv` file with `emotion`, `pixels`, and optional `Usage` columns, or image folders such as `train/<class_name>/...` and `test/<class_name>/...`. Real datasets stay local and are ignored by git.

Run a quick synthetic smoke test without downloading FER2013:

```bash
python scripts/02_train_fer_model.py --smoke-test --model all
```

Smoke-test metrics are only for checking that the pipeline runs; they are not FER2013 results.

Prepare normalized 48x48 grayscale splits:

```bash
python scripts/01_prepare_fer2013.py --dataset-path data/raw/fer2013
```

Train the baseline CNN:

```bash
python scripts/02_train_fer_model.py --model baseline --epochs 30 --batch-size 128
```

Train the improved CNN with augmentation, batch normalization, dropout, and early stopping:

```bash
python scripts/02_train_fer_model.py --model improved --epochs 30 --batch-size 128
```

The training script saves checkpoints to `models/checkpoints/`, metrics and comparison CSV files to `outputs/metrics/`, and training curves plus confusion matrices to `outputs/figures/`. Use `--output-dir` to redirect metrics and figures, or `--models-dir` to redirect checkpoints.

## Current FER2013 Selection

The current selected visual emotion model is the **improved CNN** trained on FER2013. Validation macro F1 was not available in the saved artifacts, so the selection rule used test macro F1. The improved CNN test macro F1 was `0.5891470970491275`, compared with the baseline CNN test macro F1 of `0.4895215502192262`.

Selected local checkpoint:

```text
models/checkpoints/fer2013_improved_cnn.pt
```

Relevant local artifacts:

```text
outputs/metrics/fer2013_improved_cnn_metrics.json
outputs/metrics/fer2013_improved_cnn_history.csv
outputs/figures/fer2013_improved_cnn_training_curves.png
outputs/figures/fer2013_improved_cnn_confusion_matrix.png
```

These generated artifacts are local and ignored by git. Do not commit datasets, processed arrays, checkpoints, model files, metrics, or figures unless a specific artifact is intentionally selected later for the final report or presentation.

## VOD Metadata and Human Annotations

Selected VODs and human highlight labels are tracked with small CSV files under:

```text
data/processed/annotations/
```

Use these templates to create local project files:

```text
data/processed/annotations/video_registry_template.csv
data/processed/annotations/human_highlights_template.csv
```

The actual local files expected by the loaders are:

```text
data/processed/annotations/video_registry.csv
data/processed/annotations/human_highlights.csv
```

`video_registry.csv` records selected videos with `video_id`, title, source URL, platform, duration, genre, and flags for chat/visible face availability. `human_highlights.csv` records human-labeled highlight intervals with timestamps, seconds, annotator, reason, and confidence. Real VOD metadata and annotation files remain local artifacts and should not be committed by default.

## Current Status

The repository currently implements and documents the FER2013 preparation, training, evaluation-summary, visual-model selection, VOD registry, and human annotation utilities. The next implementation phase is to use the selected FER2013 model in the video visual pipeline. The audio, text, fusion, and highlight-evaluation pipelines remain placeholders for later project milestones.
