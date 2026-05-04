from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.models.features import extract_features_batch


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


def train_and_save_feature_baseline(
    train_files: Sequence[Path],
    y_train: np.ndarray,
    val_files: Sequence[Path],
    y_val: np.ndarray,
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str],
    save_dir: Path | str = "artifacts/feature_baseline",
) -> Tuple[pd.DataFrame, dict]:
    """Train feature-based classifier, save weights, and return test predictions + metrics.
    
    Saves the model to save_dir with meta.json and model.pkl.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract features for all splits
    df_train = extract_features_batch(train_files)
    df_val = extract_features_batch(val_files)
    df_test = extract_features_batch(test_files)

    X_train, train_cols = _prepare_X(df_train)
    X_val, _ = _prepare_X(df_val)
    X_test, _ = _prepare_X(df_test)

    # Train model
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    val_acc = float(clf.score(X_val, y_val))

    # Test predictions
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

    test_results = pd.DataFrame(rows)
    test_acc = float(test_results["correct"].mean())

    # Save model and metadata
    with open(save_dir / "model.pkl", "wb") as f:
        pickle.dump(clf, f)

    meta = {
        "class_names": list(class_names),
        "feature_columns": train_cols,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    history = {
        "val_acc": [val_acc],
        "test_acc": [test_acc],
    }

    return test_results, history


def load_and_eval_feature_baseline(
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str] | None = None,
    load_dir: Path | str = "artifacts/feature_baseline",
) -> pd.DataFrame:
    """Load saved feature-based model and evaluate on test set."""
    load_dir = Path(load_dir)

    # Load model and metadata
    with open(load_dir / "model.pkl", "rb") as f:
        clf = pickle.load(f)

    with open(load_dir / "meta.json", "r") as f:
        meta = json.load(f)

    saved_class_names = meta.get("class_names", [])
    active_class_names = list(class_names) if class_names is not None else saved_class_names

    # Extract features and predict
    df_test = extract_features_batch(test_files)
    X_test, _ = _prepare_X(df_test)

    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    rows = []
    for path, true_idx, pred_idx, pred_proba in zip(test_files, y_test, y_pred, proba):
        rows.append(
            {
                "file": str(path),
                "true_label": active_class_names[int(true_idx)],
                "predicted_label": active_class_names[int(pred_idx)],
                "correct": bool(int(true_idx) == int(pred_idx)),
                "confidence": float(np.max(pred_proba)),
            }
        )

    return pd.DataFrame(rows)
