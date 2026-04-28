from __future__ import annotations

from pathlib import Path

import audioread
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def pick_device() -> torch.device:
    """Pick the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_load_audio(path: Path, target_sr: int) -> np.ndarray:
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
        audio = resample_poly(audio, target_sr // gcd, source_sr // gcd).astype(
            np.float32
        )

    target_length = target_sr * 30
    audio = np.pad(audio, (0, max(0, target_length - len(audio))))[:target_length]
    return audio.astype(np.float32)
