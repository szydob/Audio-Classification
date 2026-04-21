from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[SimpleCNN, Dict[str, List[float]], float]:
    device = pick_device()
    model = SimpleCNN(n_classes=n_classes).to(device)

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
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

    return model, history, history["val_acc"][-1]


def evaluate_cnn(model: SimpleCNN, val_loader: DataLoader, device: torch.device) -> float:
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


def predict_single(model: SimpleCNN, x_single: np.ndarray) -> int:
    device = pick_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_single[None, ...]).to(device)
        logits = model(x)
        return int(torch.argmax(logits, dim=1).item())
