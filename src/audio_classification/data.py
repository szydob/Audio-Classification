from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import audioread
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SAMPLE_RATE = 22_050
TARGET_SECONDS = 30
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def find_audio_files(dataset_root: str | Path) -> List[Path]:
    """Return all supported audio files found recursively in dataset_root."""
    root = Path(dataset_root)
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def infer_label(audio_path: Path, dataset_root: Path) -> str:
    """Infer class label from the audio file parent folder name."""
    _ = dataset_root
    return audio_path.parent.name if audio_path.parent.name else "unknown"


def build_label_index(labels: Sequence[str]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Map string labels to integer ids and return encoded array plus mappings."""
    class_names = sorted(set(labels))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.array([class_to_idx[label] for label in labels], dtype=np.int64)
    return y, class_names, class_to_idx


def scan_labeled_audio(
    dataset_root: str | Path,
    max_files: int | None = None,
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """Scan dataset, infer labels from folders, and return files + encoded labels."""
    root = Path(dataset_root)
    files = find_audio_files(root)
    if max_files is not None:
        files = files[:max_files]

    labels = [infer_label(path, root) for path in files]
    y, class_names, _ = build_label_index(labels)
    return files, y, class_names


def load_audio(path: str | Path, sr: int = SAMPLE_RATE, seconds: int = TARGET_SECONDS) -> np.ndarray:
    """Load mono audio and pad/crop it to a fixed duration.

    Uses a robust fallback chain to handle environments where one backend
    cannot decode specific files: soundfile -> audioread -> zeros.
    """
    source_sr: int | None = None
    target_len = sr * seconds

    try:
        y, source_sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        try:
            with audioread.audio_open(str(path)) as reader:
                source_sr = reader.samplerate
                chunks = []
                for buffer in reader:
                    chunks.append(np.frombuffer(buffer, dtype=np.int16))

            if not chunks:
                return np.zeros(target_len, dtype=np.float32)

            y = np.concatenate(chunks).astype(np.float32) / 32768.0
        except Exception:
            return np.zeros(target_len, dtype=np.float32)

    if source_sr is None:
        return np.zeros(target_len, dtype=np.float32)

    if source_sr != sr:
        gcd = np.gcd(source_sr, sr)
        y = resample_poly(y, sr // gcd, source_sr // gcd).astype(np.float32)

    y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
    return y.astype(np.float32)


def audio_to_logmel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert waveform to normalized log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)
    return logmel.astype(np.float32)


def split_files_train_val_test(
    files: Sequence[Path],
    y: np.ndarray,
    used_fraction: float = 0.8,
    val_fraction_of_used: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], np.ndarray, np.ndarray, np.ndarray]:
    """Split full set into train/val from used_fraction and test as the remainder."""
    if not 0 < used_fraction < 1:
        raise ValueError("used_fraction must be between 0 and 1")
    if not 0 < val_fraction_of_used < 1:
        raise ValueError("val_fraction_of_used must be between 0 and 1")

    file_idx = np.arange(len(files))
    used_idx, test_idx = train_test_split(
        file_idx,
        train_size=used_fraction,
        random_state=random_state,
        stratify=y,
    )

    y_used = y[used_idx]
    train_idx, val_idx = train_test_split(
        used_idx,
        test_size=val_fraction_of_used,
        random_state=random_state,
        stratify=y_used,
    )

    files_arr = np.array(files)
    return (
        files_arr[train_idx].tolist(),
        files_arr[val_idx].tolist(),
        files_arr[test_idx].tolist(),
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )
