#!/usr/bin/env python3
"""
Evaluate predictions from ANY model (DNABERT-2, CNN baseline, regex rules, etc.).

Inputs:
  --gold_file : CSV/TSV that contains ground-truth labels (e.g., columns: sample_id, label)
  --pred_file : CSV/TSV that contains predicted labels (e.g., columns: sample_id, pred_label)

Matching policy:
  - If --gold_file and --pred_file refer to the SAME file (common in operations), we DO NOT merge.
    We simply read once and evaluate using columns within that file.
  - Otherwise:
      - if --id_column exists in BOTH files and --no_merge is NOT set -> merge on id_column
      - else -> align by row order (must have same length)

Outputs:
  - <out_prefix>_metrics.csv
  - <out_prefix>_joined.csv
  - <out_prefix>_confusion_matrix.csv
"""

import argparse
import os
from typing import Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def read_table(path: str) -> Tuple[pd.DataFrame, str]:
    """
    Read CSV/TSV with automatic delimiter detection (csv/tsv).
    Returns (df, sep).
    """
    lower = path.lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        sep = "\t"
    else:
        sep = ","
    df = pd.read_csv(path, sep=sep)
    return df, sep


def write_table(df: pd.DataFrame, path: str) -> None:
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df.to_csv(path, index=False, sep=sep)


def _is_same_file(a: str, b: str) -> bool:
    """
    Robust same-file check (handles relative/absolute paths and symlinks when possible).
    """
    try:
        return os.path.samefile(a, b)
    except Exception:
        return os.path.abspath(a) == os.path.abspath(b)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted labels against gold labels.")
    parser.add_argument("--gold_file", required=True, type=str)
    parser.add_argument("--pred_file", required=True, type=str)
    parser.add_argument("--out_prefix", required=True, type=str)

    parser.add_argument("--gold_label_column", default="label", type=str)
    parser.add_argument("--pred_label_column", default="pred_label", type=str)

    parser.add_argument("--id_column", default="sample_id", type=str)
    parser.add_argument("--average", default="binary", type=str)  # binary / macro / weighted / micro
    parser.add_argument("--pos_label", default=1, type=str)       # kept as str; ignored unless average='binary'
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="Do not merge by id_column even if present; align by row order (or use single file if gold==pred).",
    )

    args = parser.parse_args()

    gold_df, _ = read_table(args.gold_file)
    same_file = _is_same_file(args.gold_file, args.pred_file)

    # Paths for outputs
    metrics_path = f"{args.out_prefix}_metrics.csv"
    joined_path = f"{args.out_prefix}_joined.csv"
    cm_path = f"{args.out_prefix}_confusion_matrix.csv"

    # Validate columns that must exist in gold
    if args.gold_label_column not in gold_df.columns:
        raise ValueError(f"Missing gold label column '{args.gold_label_column}' in {args.gold_file}")

    # Case A: operational mode (gold_file == pred_file) -> no merge, no re-reading required
    if same_file:
        joined = gold_df.copy()
        if args.pred_label_column not in joined.columns:
            raise ValueError(
                f"Missing pred label column '{args.pred_label_column}' in {args.gold_file}. "
                f"If your prediction column has a different name, pass --pred_label_column <name>."
            )

    else:
        pred_df, _ = read_table(args.pred_file)
        if args.pred_label_column not in pred_df.columns:
            raise ValueError(f"Missing pred label column '{args.pred_label_column}' in {args.pred_file}")

        can_merge_on_id = (
            (not args.no_merge)
            and (args.id_column in gold_df.columns)
            and (args.id_column in pred_df.columns)
        )

        if can_merge_on_id:
            # Avoid column collision if gold_df already contains the prediction column name
            if args.pred_label_column in gold_df.columns:
                gold_df = gold_df.drop(columns=[args.pred_label_column])

            joined = pd.merge(
                gold_df,
                pred_df[[args.id_column, args.pred_label_column]],
                on=args.id_column,
                how="inner",
                validate="one_to_one",
            )
            if len(joined) != len(gold_df):
                print(f"[WARN] Joined rows ({len(joined)}) != gold rows ({len(gold_df)}). Check id alignment.")
        else:
            if len(gold_df) != len(pred_df):
                raise ValueError(
                    "No merge performed, and row counts differ. "
                    "Either provide a shared --id_column (recommended) or ensure both files have identical row counts."
                )
            joined = gold_df.copy()
            joined[args.pred_label_column] = pred_df[args.pred_label_column].values

    # Compute metrics
    y_true = joined[args.gold_label_column].tolist()
    y_pred = joined[args.pred_label_column].tolist()
    
    if args.average=="binary":
        args.pos_label = int(args.pos_label)
    
    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=args.average, pos_label=args.pos_label),
        "matthews_correlation": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=args.average, pos_label=args.pos_label),
        "recall": recall_score(y_true, y_pred, average=args.average, pos_label=args.pos_label),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(cm_path, index=True)

    # Joined output (for inspection/debug)
    write_table(joined, joined_path)

    print("Metrics:", metrics)
    print("Wrote:", metrics_path)
    print("Wrote:", joined_path)
    print("Wrote:", cm_path)


if __name__ == "__main__":
    main()
