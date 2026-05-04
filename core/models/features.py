from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import librosa

from core.utils.audio import safe_load_audio
from core.data.io import SAMPLE_RATE, HOP_LENGTH


def _agg_stats(x: np.ndarray) -> Dict[str, float]:
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def extract_features(path: Path | str, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """Extract ~30 most important audio features for music genre classification.

    Focused on timbral, spectral, and rhythmic features that are most discriminative
    for genre classification. Excludes redundant deltas and less important features.

    Returns a flat dictionary of scalar features suitable for DataFrame rows.
    """
    path = Path(path)
    y = safe_load_audio(path, target_sr=sr)
    duration = len(y) / sr

    features: Dict[str, float] = {"file": str(path), "duration": float(duration)}

    # Rhythmic features (4)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    features["tempo_bpm"] = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    if len(beat_times) > 1:
        ibi = np.diff(beat_times)
        features["beat_interval_variance_s"] = float(np.var(ibi))
    else:
        features["beat_interval_variance_s"] = 0.0

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    features["onset_density_per_s"] = float(len(onset_times) / max(1.0, duration))

    # MFCC features - keep only first 6 coefficients with mean and first 4 with std (10)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

    # MFCC means for first 6 coefficients
    for i in range(6):
        stats = _agg_stats(mfcc[i])
        features[f"mfcc{i+1}_mean"] = stats["mean"]

    # MFCC std for first 4 coefficients
    for i in range(4):
        stats = _agg_stats(mfcc[i])
        features[f"mfcc{i+1}_std"] = stats["std"]

    # Spectral features - keep only means of most important ones (4)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]

    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_bandwidth_mean"] = float(np.mean(sbw))
    features["rolloff_mean"] = float(np.mean(rolloff))
    features["zcr_mean"] = float(np.mean(zcr))

    # Chroma features - keep all 12 chroma bins' means (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    for i in range(chroma.shape[0]):
        stats = _agg_stats(chroma[i])
        features[f"chroma_{i+1}_mean"] = stats["mean"]

    # RMS energy - just mean (1)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_stats = _agg_stats(rms)
    features["rms_mean"] = rms_stats["mean"]

    return features


def extract_features_batch(files: Sequence[Path | str], sr: int = SAMPLE_RATE) -> pd.DataFrame:
    """Extract features for a sequence of files and return a DataFrame."""
    rows: List[Dict[str, float]] = []
    for p in files:
        try:
            rows.append(extract_features(p, sr=sr))
        except Exception:
            # Keep pipeline robust: if a file fails, return a minimal row with file path
            rows.append({"file": str(p)})

    return pd.DataFrame(rows)