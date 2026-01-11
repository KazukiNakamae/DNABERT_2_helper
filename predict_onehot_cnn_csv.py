#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict CSV with a one-hot 3-layer CNN baseline (compatible with onehot_cnn_baseline.py).

Input CSV:
  - must contain: sequence
  - may contain: label (optional)

Output CSV (DNABERT2-like):
  sequence,label,sample_id,pred_label,prob_0,prob_1

Checkpoint:
  - best_model.pt produced by onehot_cnn_baseline.py
    typically torch.save({"model_state": model.state_dict(), "args": vars(args)}, path)
  - also supports raw state_dict checkpoints
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# One-hot encoding utilities
# -------------------------
_MAP = np.full(256, -1, dtype=np.int16)
_MAP[ord("A")] = 0
_MAP[ord("C")] = 1
_MAP[ord("G")] = 2
_MAP[ord("T")] = 3

def onehot_encode(seq: str, L: int) -> np.ndarray:
    """
    Returns array shape (4, L) for A,C,G,T.
    Unknown (e.g. N) -> all zeros.
    """
    s = seq.strip().upper()
    if len(s) != L:
        raise ValueError(f"Sequence length mismatch: expected {L}, got {len(s)} (seq={s})")

    x = np.zeros((4, L), dtype=np.float32)
    b = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
    if b.shape[0] != L:
        # fallback for any unexpected encoding issues
        for i, ch in enumerate(s):
            j = {"A": 0, "C": 1, "G": 2, "T": 3}.get(ch)
            if j is not None:
                x[j, i] = 1.0
        return x

    idx = _MAP[b]
    pos = np.where(idx >= 0)[0]
    x[idx[pos], pos] = 1.0
    return x


# -------------------------
# Model definition (same as onehot_cnn_baseline.py)
# -------------------------
class ThreeLayerCNN(nn.Module):
    """
    A Basset-style 3-conv network (binary output).
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


# -------------------------
# Dataset
# -------------------------
class SeqOnlyDataset(Dataset):
    def __init__(self, seqs: np.ndarray, seq_len: int):
        self.seqs = seqs
        self.seq_len = seq_len

    def __len__(self) -> int:
        return int(self.seqs.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = onehot_encode(str(self.seqs[idx]), self.seq_len)
        return torch.from_numpy(x)


# -------------------------
# Checkpoint loader
# -------------------------
def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # handles DataParallel "module." prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")

    # Case 1: training script saved {"model_state": ..., "args": ...}
    if isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
        args = obj.get("args", {})
        return _strip_module_prefix(state), (args if isinstance(args, dict) else {})

    # Case 2: some scripts save {"state_dict": ...}
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        args = obj.get("args", {})
        return _strip_module_prefix(state), (args if isinstance(args, dict) else {})

    # Case 3: raw state_dict
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return _strip_module_prefix(obj), {}

    raise ValueError(f"Unsupported checkpoint format: {path}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best_model.pt")
    ap.add_argument("--input", required=True, help="Input CSV (must have 'sequence')")
    ap.add_argument("--output", required=True, help="Output CSV (DNABERT2-like columns)")
    ap.add_argument("--sequence_column", default="sequence")
    ap.add_argument("--label_column", default="label")
    ap.add_argument("--seq_len", type=int, default=None, help="Override sequence length; default=infer/from checkpoint args")
    ap.add_argument("--dropout", type=float, default=None, help="Override dropout; default=from checkpoint args or 0.5")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.sequence_column not in df.columns:
        raise ValueError(f"Missing sequence column '{args.sequence_column}' in {args.input}")

    # label is optional; if absent, fill with -1
    if args.label_column not in df.columns:
        df[args.label_column] = -1

    seqs = df[args.sequence_column].astype(str).values
    inferred_len = len(str(seqs[0]).strip())
    if not np.all([len(str(s).strip()) == inferred_len for s in seqs]):
        raise ValueError("Not all sequences have the same length. Please fix input or specify --seq_len accordingly.")

    state, ckpt_args = load_checkpoint(args.model)

    seq_len = args.seq_len
    if seq_len is None:
        # try checkpoint args
        if "seq_len" in ckpt_args and isinstance(ckpt_args["seq_len"], int):
            seq_len = int(ckpt_args["seq_len"])
        else:
            seq_len = inferred_len

    dropout = args.dropout
    if dropout is None:
        if "dropout" in ckpt_args:
            try:
                dropout = float(ckpt_args["dropout"])
            except Exception:
                dropout = 0.5
        else:
            dropout = 0.5

    if seq_len < 16:
        raise ValueError(f"seq_len={seq_len} is too short for this CNN (pooling may collapse).")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = ThreeLayerCNN(seq_len=seq_len, dropout=dropout)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    ds = SeqOnlyDataset(seqs=seqs, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    prob1_list = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True).float()
            logits = model(xb)
            p1 = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
            prob1_list.append(p1)

    prob_1 = np.concatenate(prob1_list, axis=0)
    prob_0 = 1.0 - prob_1
    pred_label = (prob_1 >= 0.5).astype(int)

    out = pd.DataFrame({
        "sequence": df[args.sequence_column].astype(str).values,
        "label": df[args.label_column].astype(int).values,
        "sample_id": np.arange(df.shape[0], dtype=int),
        "pred_label": pred_label.astype(int),
        "prob_0": prob_0,
        "prob_1": prob_1,
    })
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
