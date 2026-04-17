from __future__ import annotations

from pathlib import Path
from typing import Sequence

from audio_classification.utils import pick_device, safe_load_audio
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import ASTModel, AutoFeatureExtractor


def ast_embedding_features(
    files: Sequence[Path],
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
) -> np.ndarray:
    """Convert audio files into AST embeddings."""

    device = pick_device()
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTModel.from_pretrained(model_name).to(device)
    model.eval()

    features = []
    for path in files:
        audio = safe_load_audio(path, extractor.sampling_rate)
        inputs = extractor(
            audio, sampling_rate=extractor.sampling_rate, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            embedding = (
                model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            )
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
