#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter rows where pred_label == 1 and output only sequence,label columns (pandas).
Also print counts (kept/dropped and label distribution within kept rows) to STDOUT.

Example:
  python filter_evalres_seq_label.py \
    --input fd1_rnaofftarget_test_dnabert2_test_pred.csv \
    --output pred1_sequence_label.csv
"""

import argparse
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract rows with pred_label==1 and keep only sequence,label columns (pandas)."
    )
    p.add_argument("--input", required=True, help="Input CSV file path")
    p.add_argument("--output", required=True, help="Output CSV file path")
    p.add_argument("--pred-value", type=int, default=1, help="Target value for pred_label (default: 1)")
    p.add_argument("--encoding", default="utf-8", help="Encoding (default: utf-8). Try utf-8-sig if needed.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        df = pd.read_csv(args.input, encoding=args.encoding)
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    required = {"sequence", "label", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"Error: required columns missing: {missing}. Found: {list(df.columns)}", file=sys.stderr)
        return 2

    # pred_label / label の型ゆれ対策（"1", 1, 1.0 など）
    pred = pd.to_numeric(df["pred_label"], errors="coerce")
    lab = pd.to_numeric(df["label"], errors="coerce")

    mask = pred == args.pred_value
    total = int(len(df))
    kept = int(mask.sum())
    dropped = total - kept

    # 抽出行の label 分布
    kept_lab = lab[mask]
    kept_label_1 = int((kept_lab == 1).sum())
    kept_label_0 = int((kept_lab == 0).sum())
    kept_label_other = int(kept - kept_label_1 - kept_label_0)  # NaN や 0/1以外

    out_df = df.loc[mask, ["sequence", "label"]]
    out_df.to_csv(args.output, index=False, encoding=args.encoding)

    # STDOUT に件数を出力（タブ区切り）
    print(f"total_rows\t{total}")
    print(f"kept_rows\t{kept}")
    print(f"dropped_rows\t{dropped}")
    print(f"kept_label_1\t{kept_label_1}")
    print(f"kept_label_0\t{kept_label_0}")
    print(f"kept_label_other\t{kept_label_other}")
    print(f"output_file\t{args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
