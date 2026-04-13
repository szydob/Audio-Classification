from __future__ import annotations

from pathlib import Path
from typing import Sequence

import audioread
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from scipy.signal import resample_poly
from transformers import ASTModel, AutoFeatureExtractor


def _pick_device() -> torch.device:
    """Pick the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio with a fallback path that avoids backend errors."""
    try:
        audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception:
        try:
            with audioread.audio_open(str(path)) as reader:
                source_sr = reader.samplerate
                chunks = []
                for buffer in reader:
                    chunks.append(np.frombuffer(buffer, dtype=np.int16))
                if not chunks:
                    return np.zeros(target_sr * 30, dtype=np.float32)
                audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        except Exception:
            # Last-resort fallback: keep pipeline running even if decoding backend is unavailable.
            return np.zeros(target_sr * 30, dtype=np.float32)

    if source_sr != target_sr:
        gcd = np.gcd(source_sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, source_sr // gcd).astype(np.float32)

    target_length = target_sr * 30
    audio = np.pad(audio, (0, max(0, target_length - len(audio))))[:target_length]
    return audio.astype(np.float32)


def ast_embedding_features(
    files: Sequence[Path],
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
) -> np.ndarray:
    """Convert audio files into AST embeddings."""

    device = _pick_device()
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTModel.from_pretrained(model_name).to(device)
    model.eval()

    features = []
    for path in files:
        audio = _load_audio(path, extractor.sampling_rate)
        inputs = extractor(audio, sampling_rate=extractor.sampling_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        features.append(embedding.astype(np.float32))

    return np.stack(features)


def run_ast_logreg_baseline(
    train_files: Sequence[Path],
    y_train: np.ndarray,
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Train Logistic Regression on AST embeddings and return test predictions."""
    x_train = ast_embedding_features(train_files)
    x_test = ast_embedding_features(test_files)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)

    rows = []
    for path, true_idx, pred_idx, pred_proba in zip(test_files, y_test, y_pred, proba):
        rows.append(
            {
                "file": str(path),
                "true_label": class_names[int(true_idx)],
                "predicted_label": class_names[int(pred_idx)],
                "correct": bool(int(true_idx) == int(pred_idx)),
                "confidence": float(np.max(pred_proba)),
            }
        )

    return pd.DataFrame(rows)
