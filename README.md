# Audio Classification

This project helps you work with audio classification in a simple way.

It can:

1. Load a music dataset.
2. Convert audio into log-mel spectrograms.
3. Train and compare an AST-based SoTA baseline.
4. Classify new audio files in a Streamlit app.

## Setup

Install dependencies:

```bash
uv sync
```

If you want to run the SoTA (AST) baseline, also install the transformer dependencies:

```bash
uv sync --all-extras
```

Run the app:

```bash
uv run streamlit run gui.py
```

## What each file does

- `core/data/io.py`
  - Scans labeled audio files.
  - Loads waveform and normalizes length.
  - Converts waveform to log-mel spectrogram.
- `core/models/sota.py`
  - Uses Hugging Face AST transformer as feature extractor.
  - Trains Logistic Regression on AST embeddings for comparison.
- `core/models/cnn.py`
  - Trains and evaluates CNN baseline on log-mel features.
- `core/models/transformer.py`
  - Runs the saved Mini AudioTransformer baseline.
- `gui.py`
  - Pick dataset source (GTZAN or local folder).
  - Prepare train/validation split.
  - Run the AST baseline.
  - Upload and classify a new audio file.

## Dataset notes

Default dataset in `pyproject.toml` is GTZAN via KaggleHub:

- `andradaolteanu/gtzan-dataset-music-genre-classification`

You can add more datasets later in the dataset config section and expose them in the GUI selector.

## SoTA download

The AST model is downloaded automatically the first time you run the baseline.

If you want to check it manually, run:

```bash
uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')"
```
