from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import json

import numpy as np
import pandas as pd
from core.utils.audio import pick_device, safe_load_audio
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from core.data.io import SAMPLE_RATE, audio_to_logmel


class ResBlock(nn.Module):
    """A small 'brain' piece that remembers the original signal."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # The 'Skip Connection'
        return F.relu(out)


class CNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)  # Prevents memorization
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[CNN, Dict[str, List[float]], float]:
    device = pick_device()
    model = CNN(n_classes=n_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {"train_loss": [], "val_acc": []}

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
        val_acc = evaluate_cnn(model, val_loader, device)
        scheduler.step(val_acc)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

    return model, history, history["val_acc"][-1]


def evaluate_cnn(model: CNN, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    pred_all = np.concatenate(preds)
    target_all = np.concatenate(targets)
    return float(accuracy_score(target_all, pred_all))


def predict_single(model: CNN, x_single: np.ndarray) -> int:
    device = pick_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_single[None, ...]).to(device)
        logits = model(x)
        return int(torch.argmax(logits, dim=1).item())


def _files_to_cnn_inputs(files: Sequence[Path]) -> np.ndarray:
    """Convert files to CNN-ready normalized log-mel tensors: (N, 1, H, W)."""

    x = []
    for path in files:
        y = safe_load_audio(path, target_sr=SAMPLE_RATE)
        logmel = audio_to_logmel(y)
        x.append(logmel[None, ...])
    return np.stack(x).astype(np.float32)


def run_cnn_baseline(
    train_files: Sequence[Path],
    y_train: np.ndarray,
    val_files: Sequence[Path],
    y_val: np.ndarray,
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str],
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[pd.DataFrame, Dict[str, List[float]]]:
    """Train CNN on log-mels and return test predictions + training history."""
    x_train = _files_to_cnn_inputs(train_files)
    x_val = _files_to_cnn_inputs(val_files)
    x_test = _files_to_cnn_inputs(test_files)

    model, history, _ = train_cnn(
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_val=x_val,
        y_val=y_val.astype(np.int64),
        n_classes=len(class_names),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

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

    y_pred: list[int] = []
    y_conf: list[float] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy().tolist()
            conf = torch.max(probs, dim=1).values.cpu().numpy().tolist()
            y_pred.extend(int(p) for p in pred)
            y_conf.extend(float(c) for c in conf)

    rows = []
    for path, true_idx, pred_idx, conf in zip(test_files, y_test, y_pred, y_conf):
        rows.append(
            {
                "file": str(path),
                "true_label": class_names[int(true_idx)],
                "predicted_label": class_names[int(pred_idx)],
                "correct": bool(int(true_idx) == int(pred_idx)),
                "confidence": float(conf),
            }
        )

    return pd.DataFrame(rows), history


def train_and_save_cnn_baseline(
    train_files: Sequence[Path],
    y_train: np.ndarray,
    val_files: Sequence[Path],
    y_val: np.ndarray,
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str],
    epochs: int,
    batch_size: int,
    lr: float,
    save_dir: Path | str = "artifacts/cnn_baseline",
) -> tuple[pd.DataFrame, Dict[str, List[float]]]:
    """Train CNN, save weights, and return test predictions + training history."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x_train = _files_to_cnn_inputs(train_files)
    x_val = _files_to_cnn_inputs(val_files)
    x_test = _files_to_cnn_inputs(test_files)

    model, history, _ = train_cnn(
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_val=x_val,
        y_val=y_val.astype(np.int64),
        n_classes=len(class_names),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

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

    y_pred: list[int] = []
    y_conf: list[float] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy().tolist()
            conf = torch.max(probs, dim=1).values.cpu().numpy().tolist()
            y_pred.extend(int(p) for p in pred)
            y_conf.extend(float(c) for c in conf)

    rows = []
    for path, true_idx, pred_idx, conf in zip(test_files, y_test, y_pred, y_conf):
        rows.append(
            {
                "file": str(path),
                "true_label": class_names[int(true_idx)],
                "predicted_label": class_names[int(pred_idx)],
                "correct": bool(int(true_idx) == int(pred_idx)),
                "confidence": float(conf),
            }
        )

    test_results = pd.DataFrame(rows)

    # Save model and metadata
    meta = {
        "class_names": list(class_names),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }
    torch.save(meta, save_dir / "meta.pt")
    torch.save(model.state_dict(), save_dir / "cnn_best.pt")

    return test_results, history


def load_and_eval_cnn_baseline(
    test_files: Sequence[Path],
    y_test: np.ndarray,
    class_names: Sequence[str] | None = None,
    load_dir: Path | str = "artifacts/cnn_baseline",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Load saved CNN model and evaluate on test set."""
    load_dir = Path(load_dir)

    # Load model and metadata
    meta = torch.load(load_dir / "meta.pt", map_location="cpu")
    saved_class_names = meta.get("class_names", [])
    active_class_names = list(class_names) if class_names is not None else saved_class_names

    # Recreate model
    model = CNN(n_classes=len(active_class_names))
    state = torch.load(load_dir / "cnn_best.pt", map_location="cpu")
    model.load_state_dict(state)

    device = pick_device()
    model = model.to(device)
    model.eval()

    # Prepare test data
    x_test = _files_to_cnn_inputs(test_files)
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.int64))
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    y_pred: list[int] = []
    y_conf: list[float] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy().tolist()
            conf = torch.max(probs, dim=1).values.cpu().numpy().tolist()
            y_pred.extend(int(p) for p in pred)
            y_conf.extend(float(c) for c in conf)

    rows = []
    for path, true_idx, pred_idx, conf in zip(test_files, y_test, y_pred, y_conf):
        rows.append(
            {
                "file": str(path),
                "true_label": active_class_names[int(true_idx)],
                "predicted_label": active_class_names[int(pred_idx)],
                "correct": bool(int(true_idx) == int(pred_idx)),
                "confidence": float(conf),
            }
        )

    return pd.DataFrame(rows)
