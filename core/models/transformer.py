from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from core.utils.audio import pick_device, safe_load_audio
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

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


def _files_to_transformer_inputs(files: Sequence[Path]) -> np.ndarray:
    """Convert audio files to transformer-ready log-mel spectrograms: (N, 1, H, W)."""
    x = []
    for path in files:
        y = load_audio(path, sr=SAMPLE_RATE, seconds=TARGET_SECONDS)
        logmel = audio_to_logmel(y, sr=SAMPLE_RATE)
        x.append(logmel[None, ...])
    return np.stack(x).astype(np.float32)


def train_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    n_mels: int = N_MELS,
    patch_mels: int = 16,
    patch_time: int = 16,
    embed_dim: int = 128,
    depth: int = 6,
    num_heads: int = 8,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[MiniAudioTransformer, Dict[str, List[float]], float]:
    """Train MiniAudioTransformer on log-mels."""
    device = pick_device()
    model = MiniAudioTransformer(
        n_classes=n_classes,
        n_mels=n_mels,
        patch_mels=patch_mels,
        patch_time=patch_time,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=3, factor=0.5
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {"train_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))

        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.append(pred)
                val_targets.append(yb.numpy())

        val_pred_all = np.concatenate(val_preds)
        val_target_all = np.concatenate(val_targets)
        val_acc = float(accuracy_score(val_target_all, val_pred_all))

        scheduler.step(val_acc)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return model, history, best_val_acc


def train_and_save_transformer_baseline(
    train_files: Sequence[Path],
    y_train: np.ndarray,
    val_files: Sequence[Path],
    y_val: np.ndarray,
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_dir: Path | str = "artifacts/mini_audio_transformer_custom",
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """Train MiniAudioTransformer, save weights, and return test predictions + history."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert files to log-mel inputs
    x_train = _files_to_transformer_inputs(train_files)
    x_val = _files_to_transformer_inputs(val_files)
    x_test = _files_to_transformer_inputs(test_files)

    # Train model
    model, history, _ = train_transformer(
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_val=x_val,
        y_val=y_val.astype(np.int64),
        n_classes=len(class_names),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Test predictions
    device = pick_device()
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.int64))
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    y_pred_all = []
    y_proba_all = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probas = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred_all.append(preds)
            y_proba_all.append(probas)

    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all)

    rows = []
    for path, true_idx, pred_idx, pred_proba in zip(test_files, y_test, y_pred, y_proba):
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

    # Save model and metadata
    meta = {
        "class_names": list(class_names),
        "params": {
            "N_MELS": N_MELS,
            "PATCH_MELS": 16,
            "PATCH_TIME": 16,
            "EMBED_DIM": 128,
            "DEPTH": 6,
            "NUM_HEADS": 8,
            "MLP_RATIO": 4.0,
            "DROPOUT": 0.15,
        },
    }
    torch.save(meta, save_dir / "meta.pt")
    torch.save(model.state_dict(), save_dir / "mini_audio_transformer_best.pt")

    return test_results, history


def run_transformer_baseline(
    train_files: Sequence[Path],
    y_train: np.ndarray,
    val_files: Sequence[Path],
    y_val: np.ndarray,
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """Train MiniAudioTransformer with fixed params and return test predictions + history."""
    # Fixed parameters for quick eval
    epochs = 5
    batch_size = 32
    lr = 1e-3

    # Convert files to log-mel inputs
    x_train = _files_to_transformer_inputs(train_files)
    x_val = _files_to_transformer_inputs(val_files)
    x_test = _files_to_transformer_inputs(test_files)

    # Train model
    model, history, _ = train_transformer(
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_val=x_val,
        y_val=y_val.astype(np.int64),
        n_classes=len(class_names),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Test predictions
    device = pick_device()
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.int64))
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    y_pred_all = []
    y_proba_all = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probas = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred_all.append(preds)
            y_proba_all.append(probas)

    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all)

    rows = []
    for path, true_idx, pred_idx, pred_proba in zip(test_files, y_test, y_pred, y_proba):
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

    return test_results, history
