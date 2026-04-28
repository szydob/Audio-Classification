from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from core.utils.audio import pick_device, safe_load_audio
import torch
from torch import nn

from core.data.io import N_MELS, SAMPLE_RATE, TARGET_SECONDS, audio_to_logmel, load_audio


def _pick_device() -> torch.device:
    return pick_device()


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MiniAudioTransformer(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_mels: int = N_MELS,
        patch_mels: int = 16,
        patch_time: int = 16,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.15,
    ):
        super().__init__()
        _ = n_mels
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(patch_mels, patch_time),
            stride=(patch_mels, patch_time),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def _resolve_artifact_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    candidates = [base / "0p59", base]
    for candidate in candidates:
        if (candidate / "meta.pt").exists() and (candidate / "mini_audio_transformer_best.pt").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find model artifacts in {base}. Expected meta.pt and mini_audio_transformer_best.pt"
    )


def _load_model_from_artifacts(base_dir: str | Path, device: torch.device) -> tuple[MiniAudioTransformer, list[str]]:
    artifact_dir = _resolve_artifact_dir(base_dir)
    meta = torch.load(artifact_dir / "meta.pt", map_location="cpu")
    params = dict(meta.get("params", {}))
    class_names = list(meta.get("class_names", []))

    if not class_names:
        raise ValueError(f"No class_names found in {artifact_dir / 'meta.pt'}")

    model = MiniAudioTransformer(
        n_classes=len(class_names),
        n_mels=int(params.get("N_MELS", N_MELS)),
        patch_mels=int(params.get("PATCH_MELS", 16)),
        patch_time=int(params.get("PATCH_TIME", 16)),
        embed_dim=int(params.get("EMBED_DIM", 128)),
        depth=int(params.get("DEPTH", 6)),
        num_heads=int(params.get("NUM_HEADS", 8)),
        mlp_ratio=float(params.get("MLP_RATIO", 4.0)),
        dropout=float(params.get("DROPOUT", 0.15)),
    ).to(device)

    state = torch.load(artifact_dir / "mini_audio_transformer_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, class_names


def _predict_multicrop(
    model: MiniAudioTransformer,
    path: Path,
    device: torch.device,
    crop_seconds: int = 10,
    n_crops: int = 5,
) -> tuple[int, float]:
    wav = load_audio(path, sr=SAMPLE_RATE, seconds=TARGET_SECONDS)
    crop_len = int(crop_seconds * SAMPLE_RATE)
    total_len = len(wav)

    if crop_len >= total_len:
        starts = [0]
    else:
        starts = np.linspace(0, total_len - crop_len, num=max(1, n_crops), dtype=int).tolist()

    probs = []
    with torch.no_grad():
        for start in starts:
            crop = wav[start : start + crop_len]
            mel = audio_to_logmel(crop, sr=SAMPLE_RATE).astype(np.float32)
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            probs.append(p)

    mean_proba = np.mean(np.stack(probs, axis=0), axis=0)
    pred_idx = int(np.argmax(mean_proba))
    confidence = float(np.max(mean_proba))
    return pred_idx, confidence


def run_saved_transformer_baseline(
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str] | None = None,
    artifacts_dir: str | Path = "artifacts/mini_audio_transformer",
    crop_seconds: int = 10,
    n_crops: int = 5,
) -> pd.DataFrame:
    """Run predictions on test files using the saved MiniAudioTransformer artifacts.

    Returns a DataFrame with the same columns as GUI test tables.
    """
    device = _pick_device()
    model, trained_class_names = _load_model_from_artifacts(artifacts_dir, device)

    active_class_names = list(class_names) if class_names is not None else trained_class_names
    if len(active_class_names) != len(trained_class_names):
        active_class_names = trained_class_names

    rows = []
    for path, true_idx in zip(test_files, y_test):
        pred_idx, confidence = _predict_multicrop(
            model=model,
            path=Path(path),
            device=device,
            crop_seconds=crop_seconds,
            n_crops=n_crops,
        )
        rows.append(
            {
                "file": str(path),
                "true_label": active_class_names[int(true_idx)],
                "predicted_label": active_class_names[pred_idx],
                "correct": bool(int(true_idx) == int(pred_idx)),
                "confidence": confidence,
            }
        )

    return pd.DataFrame(rows)
