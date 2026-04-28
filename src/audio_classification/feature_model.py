from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.audio_classification.features import extract_features_batch


def _prepare_X(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    cols = [c for c in df.columns if c != "file"]
    X = df[cols].fillna(0.0).astype(float)
    return X.values, cols


def run_feature_baseline(
    train_files: Sequence[Path],
    y_train: np.ndarray,
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Train a simple feature-based classifier and return test predictions.

    Uses MFCC/spectral/chroma features from `features.extract_features_batch`.
    """
    df_train = extract_features_batch(train_files)
    df_test = extract_features_batch(test_files)

    X_train, _ = _prepare_X(df_train)
    X_test, _ = _prepare_X(df_test)

    # Simple standardized logistic regression baseline
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

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
