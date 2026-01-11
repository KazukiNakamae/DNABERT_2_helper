#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Motif (IUPAC) pattern-match baseline.

Input CSV:
  - must contain: sequence
  - may contain: label (optional)

Output CSV (DNABERT2-like):
  sequence,label,sample_id,pred_label,prob_0,prob_1

Default behavior:
  - Check subsequence sequence[start : start+len(pattern)] (0-based start)
  - If matches IUPAC pattern -> pred_label=1 else 0
"""

import argparse
import numpy as np
import pandas as pd

IUPAC_CODES = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}

def matches_pattern(seq: str, pattern: str) -> bool:
    seq = seq.upper()
    pattern = pattern.upper()
    if len(seq) != len(pattern):
        return False
    for s, p in zip(seq, pattern):
        allowed = IUPAC_CODES.get(p)
        if allowed is None:
            raise ValueError(f"Unsupported IUPAC code in pattern: '{p}'")
        if s not in allowed:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV")
    ap.add_argument("--output", required=True, help="Output CSV (DNABERT2-like columns)")
    ap.add_argument("--sequence_column", default="sequence")
    ap.add_argument("--label_column", default="label")
    ap.add_argument("--pattern", required=True, help="IUPAC motif pattern (e.g., 'TGG', 'NNN', 'RYS')")
    ap.add_argument("--start", type=int, default=19,
                    help="0-based start position to check (default=19 means positions 20-.. in 1-based)")
    ap.add_argument("--pos_prob", type=float, default=1.0, help="prob_1 when motif matches")
    ap.add_argument("--neg_prob", type=float, default=0.0, help="prob_1 when motif does NOT match")
    args = ap.parse_args()

    if not (0.0 <= args.pos_prob <= 1.0 and 0.0 <= args.neg_prob <= 1.0):
        raise ValueError("--pos_prob/--neg_prob must be in [0,1]")

    df = pd.read_csv(args.input)
    if args.sequence_column not in df.columns:
        raise ValueError(f"Missing sequence column '{args.sequence_column}' in {args.input}")

    if args.label_column not in df.columns:
        df[args.label_column] = -1

    seqs = df[args.sequence_column].astype(str).values
    pat = args.pattern.strip().upper()
    w = len(pat)

    pred = np.zeros(df.shape[0], dtype=int)
    prob_1 = np.full(df.shape[0], args.neg_prob, dtype=float)

    for i, s in enumerate(seqs):
        ss = str(s).strip().upper()
        if args.start < 0 or args.start + w > len(ss):
            # out-of-range -> treat as non-match
            continue
        sub = ss[args.start: args.start + w]
        if matches_pattern(sub, pat):
            pred[i] = 1
            prob_1[i] = args.pos_prob

    prob_0 = 1.0 - prob_1

    out = pd.DataFrame({
        "sequence": df[args.sequence_column].astype(str).values,
        "label": df[args.label_column].astype(int).values,
        "sample_id": np.arange(df.shape[0], dtype=int),
        "pred_label": pred.astype(int),
        "prob_0": prob_0,
        "prob_1": prob_1,
    })
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
