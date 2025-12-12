#!/usr/bin/env python3
"""
3-layer one-hot CNN baseline for promoter classification (700bp sequences).

Inputs:
  train/dev/test CSV with columns: sequence,label

Outputs:
  - best_model.pt
  - metrics.json
  - pred_dev.csv, pred_test.csv

Requirements:
  torch, numpy, pandas, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: sequence,label")
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.upper()
    df["label"] = df["label"].astype(int)
    return df

_COMP = str.maketrans({"A":"T","C":"G","G":"C","T":"A","N":"N"})
def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]

def onehot_encode(seq: str, L: int) -> np.ndarray:
    """
    Returns array shape (4, L) for A,C,G,T.
    Unknown (e.g. N) -> all zeros.
    """
    seq = seq.upper()
    if len(seq) != L:
        raise ValueError(f"Expected length {L} but got {len(seq)}")
    x = np.zeros((4, L), dtype=np.float32)
    mapping = {"A":0, "C":1, "G":2, "T":3}
    for i, ch in enumerate(seq):
        j = mapping.get(ch, None)
        if j is not None:
            x[j, i] = 1.0
    return x

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, rc_aug: bool, split: str, seed: int):
        self.seqs = df["sequence"].tolist()
        self.y = df["label"].to_numpy(dtype=np.float32)
        self.seq_len = seq_len
        self.rc_aug = rc_aug
        self.split = split
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.seqs[idx]
        if self.rc_aug and self.split == "train":
            if self.rng.rand() < 0.5:
                s = revcomp(s)
        x = onehot_encode(s, self.seq_len)
        return torch.from_numpy(x), torch.tensor(self.y[idx])

class ThreeLayerCNN(nn.Module):
    """
    A Basset-style 3-conv + 2-FC network (adapted for 700bp, binary output).
    """
    def __init__(self, seq_len: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 320, kernel_size=19, padding=9)
        self.conv2 = nn.Conv1d(320, 480, kernel_size=11, padding=5)
        self.conv3 = nn.Conv1d(480, 960, kernel_size=7, padding=3)

        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.drop = nn.Dropout(dropout)

        self.fc1 = nn.Linear(960, 925)
        self.fc2 = nn.Linear(925, 1)

        # optional init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,4,L)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        # global max pool over length
        x = torch.max(x, dim=2).values  # (B,960)
        x = self.drop(self.relu(self.fc1(self.drop(x))))
        logits = self.fc2(x).squeeze(1)  # (B,)
        return logits

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> dict:
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
    }

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probs)

def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total += float(loss.detach().cpu()) * len(y)
        n += len(y)
    return total / max(1, n)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, type=Path)
    ap.add_argument("--dev_csv", required=True, type=Path)
    ap.add_argument("--test_csv", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seq_len", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=6, help="Early stopping patience on dev AUROC.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rc_aug", action="store_true", help="Reverse-complement augmentation during training only.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)

    train_df = read_split(args.train_csv)
    dev_df = read_split(args.dev_csv)
    test_df = read_split(args.test_csv)

    # Sanity check lengths
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        lens = df["sequence"].astype(str).str.len()
        if lens.min() != args.seq_len or lens.max() != args.seq_len:
            raise ValueError(f"{name} has sequence lengths outside {args.seq_len}. Found min={lens.min()} max={lens.max()}")

    train_ds = SeqDataset(train_df, args.seq_len, args.rc_aug, "train", args.seed)
    dev_ds   = SeqDataset(dev_df,   args.seq_len, False,      "dev",   args.seed)
    test_ds  = SeqDataset(test_df,  args.seq_len, False,      "test",  args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    dev_loader   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    model = ThreeLayerCNN(seq_len=args.seq_len, dropout=args.dropout).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_path = args.outdir / "best_model.pt"
    bad = 0

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, loss_fn, device)
        y_dev, p_dev = predict(model, dev_loader, device)
        dev_metrics = compute_metrics(y_dev, p_dev)

        history.append({"epoch": epoch, "train_loss": tr_loss, **{f"dev_{k}": v for k, v in dev_metrics.items()}})
        print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | dev_auroc={dev_metrics['auroc']:.4f} | dev_auprc={dev_metrics['auprc']:.4f} | dev_mcc={dev_metrics['mcc']:.4f}")

        if dev_metrics["auroc"] > best_auc + 1e-4:
            best_auc = dev_metrics["auroc"]
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch} (no dev AUROC improvement for {args.patience} epochs).")
                break

    # Load best and evaluate
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    y_dev, p_dev = predict(model, dev_loader, device)
    y_test, p_test = predict(model, test_loader, device)
    dev_metrics = compute_metrics(y_dev, p_dev)
    test_metrics = compute_metrics(y_test, p_test)

    with (args.outdir / "metrics.json").open("w") as f:
        json.dump({"dev": dev_metrics, "test": test_metrics, "history": history}, f, indent=2)

    def save_preds(df: pd.DataFrame, prob: np.ndarray, split: str) -> None:
        out = df.copy()
        out["prob"] = prob
        out["pred"] = (prob >= 0.5).astype(int)
        out.to_csv(args.outdir / f"pred_{split}.csv", index=False)

    save_preds(dev_df, p_dev, "dev")
    save_preds(test_df, p_test, "test")

    print("Done.")
    print("Dev metrics:", dev_metrics)
    print("Test metrics:", test_metrics)
    print(f"Saved to: {args.outdir}")

if __name__ == "__main__":
    main()