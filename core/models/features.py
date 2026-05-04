from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
#import librosa

from core.utils.audio import safe_load_audio
from core.data.io import SAMPLE_RATE, HOP_LENGTH


def _agg_stats(x: np.ndarray) -> Dict[str, float]:
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def extract_features(path: Path | str, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """Extract a broad set of audio features from a single file.

    Returns a flat dictionary of scalar features suitable for DataFrame rows.
    """
    path = Path(path)
    y = safe_load_audio(path, target_sr=sr)
    duration = len(y) / sr

    features: Dict[str, float] = {"file": str(path), "duration": float(duration)}

    # Onset envelope and tempo / beats
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    features["tempo_bpm"] = float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    if len(beat_times) > 1:
        ibi = np.diff(beat_times)
        features["beat_interval_variance_s"] = float(np.var(ibi))
    else:
        features["beat_interval_variance_s"] = 0.0

    # Onset density (onsets per second)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    features["onset_density_per_s"] = float(len(onset_times) / max(1.0, duration))

    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(mfcc.shape[0]):
        stats = _agg_stats(mfcc[i])
        features[f"mfcc{i+1}_mean"] = stats["mean"]
        features[f"mfcc{i+1}_std"] = stats["std"]

        dstats = _agg_stats(delta[i])
        features[f"mfcc{i+1}_delta_mean"] = dstats["mean"]
        features[f"mfcc{i+1}_delta_std"] = dstats["std"]

        d2stats = _agg_stats(delta2[i])
        features[f"mfcc{i+1}_delta2_mean"] = d2stats["mean"]
        features[f"mfcc{i+1}_delta2_std"] = d2stats["std"]

    # Spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]

    for k, stat in _agg_stats(sc).items():
        features[f"spectral_centroid_{k}"] = stat
    for k, stat in _agg_stats(sbw).items():
        features[f"spectral_bandwidth_{k}"] = stat
    for i in range(contrast.shape[0]):
        stats = _agg_stats(contrast[i])
        features[f"spectral_contrast_{i+1}_mean"] = stats["mean"]
        features[f"spectral_contrast_{i+1}_std"] = stats["std"]
    for k, stat in _agg_stats(rolloff).items():
        features[f"rolloff_{k}"] = stat
    for k, stat in _agg_stats(zcr).items():
        features[f"zcr_{k}"] = stat

    # Chroma and tonnetz
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    for i in range(chroma.shape[0]):
        stats = _agg_stats(chroma[i])
        features[f"chroma_{i+1}_mean"] = stats["mean"]
        features[f"chroma_{i+1}_std"] = stats["std"]

    ton = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(ton.shape[0]):
        stats = _agg_stats(ton[i])
        features[f"tonnetz_{i+1}_mean"] = stats["mean"]
        features[f"tonnetz_{i+1}_std"] = stats["std"]

    # RMS energy and dynamic range
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_stats = _agg_stats(rms)
    features["rms_mean"] = rms_stats["mean"]
    features["rms_std"] = rms_stats["std"]
    # dynamic range in dB (safe epsilon)
    eps = 1e-8
    dyn_db = 20.0 * float(np.log10((rms_stats["max"] + eps) / (rms_stats["min"] + eps)))
    features["dynamic_range_db"] = dyn_db

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
