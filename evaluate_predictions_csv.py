#!/usr/bin/env python3
"""
Evaluate predictions from ANY model (DNABERT-2, CNN baseline, regex rules, etc.).

Inputs:
  --gold_file : CSV/TSV with ground-truth labels (e.g., columns: sample_id, sequence, label)
  --pred_file : CSV/TSV with predicted labels (e.g., columns: sample_id, pred_label) or any column name

Default matching:
  - if --id_column exists in BOTH files -> merge on it
  - else -> align by row order (must have same length)

Outputs:
  - <out_prefix>_metrics.csv
  - <out_prefix>_joined.csv  (gold + pred columns joined for inspection)
  - <out_prefix>_confusion_matrix.csv (optional but generated here)
"""

import argparse
from typing import Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)


def read_table(path: str) -> Tuple[pd.DataFrame, str]:
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    return df, sep


def write_table(df: pd.DataFrame, path: str) -> None:
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df.to_csv(path, index=False, sep=sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted labels against gold labels.")
    parser.add_argument("--gold_file", required=True, type=str)
    parser.add_argument("--pred_file", required=True, type=str)
    parser.add_argument("--out_prefix", required=True, type=str)

    parser.add_argument("--gold_label_column", default="label", type=str)
    parser.add_argument("--pred_label_column", default="pred_label", type=str)

    parser.add_argument("--id_column", default="sample_id", type=str)
    parser.add_argument("--average", default="binary", type=str)  # binary / macro / weighted / micro
    parser.add_argument("--pos_label", default=1, type=str)       # kept as str; sklearn accepts str/int

    args = parser.parse_args()

    gold_df, _ = read_table(args.gold_file)
    pred_df, _ = read_table(args.pred_file)

    if args.gold_label_column not in gold_df.columns:
        raise ValueError(f"Missing gold label column '{args.gold_label_column}' in {args.gold_file}")
    if args.pred_label_column not in pred_df.columns:
        raise ValueError(f"Missing pred label column '{args.pred_label_column}' in {args.pred_file}")

    # Join strategy
    can_merge_on_id = (args.id_column in gold_df.columns) and (args.id_column in pred_df.columns)
    if can_merge_on_id:
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
                "No common id_column found, and row counts differ. "
                "Provide a shared id_column (recommended: sample_id)."
            )
        joined = gold_df.copy()
        joined[args.pred_label_column] = pred_df[args.pred_label_column].values

    y_true = joined[args.gold_label_column].tolist()
    y_pred = joined[args.pred_label_column].tolist()

    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=args.average, pos_label=args.pos_label),
        "matthews_correlation": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=args.average, pos_label=args.pos_label),
        "recall": recall_score(y_true, y_pred, average=args.average, pos_label=args.pos_label),
    }

    metrics_path = f"{args.out_prefix}_metrics.csv"
    joined_path = f"{args.out_prefix}_joined.csv"
    cm_path = f"{args.out_prefix}_confusion_matrix.csv"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # Confusion matrix (labels sorted by appearance)
    labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(cm_path, index=True)

    write_table(joined, joined_path)

    print("Metrics:", metrics)
    print("Wrote:", metrics_path)
    print("Wrote:", joined_path)
    print("Wrote:", cm_path)


if __name__ == "__main__":
    main()
