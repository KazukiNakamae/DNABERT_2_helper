#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_METRICS = ["accuracy", "f1", "matthews_correlation", "precision", "recall"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge per-(model,dataset[,rep]) metrics CSVs into one long-format CSV with model/dataset/rep/n columns."
    )
    p.add_argument("--input_glob", default="**/*_metrics.csv",
                   help="Glob for metrics CSV files. Default: '**/*_metrics.csv'")
    p.add_argument("--recursive", action="store_true",
                   help="Enable recursive glob (**) expansion.")
    p.add_argument("--out_csv", default="merged_metrics.csv",
                   help="Output merged CSV path.")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                   help="Comma-separated metric columns to extract.")
    p.add_argument("--n_col", default="n",
                   help="Column name for sample size in the input metrics CSV. Default: 'n'")

    # filename parsing (fixed by your convention)
    p.add_argument("--filename_regex", default=None,
                   help=(
                       "Regex to parse filename basename into model/dataset. "
                       "Must contain named groups (?P<model>...) and (?P<dataset>...). "
                       "Default matches: <model>_eval_<dataset>_test_dnabert2_test_pred_metrics.csv"
                   ))

    # rep extraction
    p.add_argument("--rep_regex", default=None,
                   help=(
                       "Regex to extract rep from FULL PATH (not just basename). "
                       "Must contain named group (?P<rep>\\d+). "
                       "If not matched and no rep column exists, rep=1."
                   ))
    p.add_argument("--rep_col", default=None,
                   help="If the input CSV already has this column name (e.g. 'rep'), use it as rep.")
    p.add_argument("--row_agg", choices=["mean", "median", "first"], default="first",
                   help="If a metrics CSV has multiple rows, how to summarize into one row.")
    p.add_argument("--strict", action="store_true",
                   help="If set, missing metric columns cause an error; otherwise set them to NaN.")
    return p.parse_args()


def list_files(pattern: str, recursive: bool) -> List[str]:
    return sorted(glob.glob(pattern, recursive=recursive))


def default_filename_regex() -> re.Pattern:
    # <model>_eval_<dataset>_test_dnabert2_test_pred_metrics.csv
    return re.compile(r"^(?P<model>.+?)_eval_(?P<dataset>.+)_test_dnabert2_test_pred_metrics\.csv$")


def parse_model_dataset(path: str, rx: re.Pattern) -> Dict[str, str]:
    base = os.path.basename(path)
    m = rx.match(base)
    if not m:
        raise ValueError(f"Filename did not match regex: {base}")
    return {"model": m.group("model"), "dataset": m.group("dataset")}


def extract_rep(path: str, rep_col: Optional[str], rep_rx: Optional[re.Pattern], df: pd.DataFrame) -> int:
    if rep_col and rep_col in df.columns:
        try:
            return int(df[rep_col].iloc[0])
        except Exception:
            pass

    if rep_rx:
        m = rep_rx.search(path)
        if m:
            return int(m.group("rep"))

    return 1


def summarize_one_value(series: pd.Series, row_agg: str) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return float("nan")
    if row_agg == "mean":
        return float(s.mean())
    if row_agg == "median":
        return float(s.median())
    return float(s.iloc[0])


def summarize_metrics(df: pd.DataFrame, metrics: List[str], row_agg: str, strict: bool) -> Dict[str, float]:
    out = {}
    for m in metrics:
        if m not in df.columns:
            if strict:
                raise ValueError(f"Missing metric column: {m}. Available: {list(df.columns)}")
            out[m] = float("nan")
            continue
        out[m] = summarize_one_value(df[m], row_agg=row_agg)
    return out


def summarize_n(df: pd.DataFrame, n_col: str, row_agg: str) -> float:
    if n_col not in df.columns:
        return float("nan")
    # n は通常一定なので first で十分だが、複数行あり得るため集約可能にしておく
    return summarize_one_value(df[n_col], row_agg=row_agg)


def main():
    args = parse_args()
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]

    files = list_files(args.input_glob, args.recursive)
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    fn_rx = re.compile(args.filename_regex) if args.filename_regex else default_filename_regex()
    rep_rx = re.compile(args.rep_regex) if args.rep_regex else None

    rows = []
    for f in files:
        df = pd.read_csv(f)

        md = parse_model_dataset(f, fn_rx)
        rep = extract_rep(f, args.rep_col, rep_rx, df)
        n_val = summarize_n(df, n_col=args.n_col, row_agg="first")  # nは基本first推奨
        met = summarize_metrics(df, metrics, args.row_agg, args.strict)

        row = {
            "model": md["model"],
            "dataset": md["dataset"],
            "rep": rep,
            "n": n_val,
            "source_file": f,
        }
        row.update(met)
        rows.append(row)

    merged = pd.DataFrame(rows).sort_values(["model", "dataset", "rep"]).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"[OK] merged rows: {len(merged)}")
    print(f"[OK] out: {out_path}")
    print(f"[OK] metrics: {metrics}")
    print(f"[OK] n_col: {args.n_col}")


if __name__ == "__main__":
    main()
