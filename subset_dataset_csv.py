#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd


def make_subset_train(
    input_path: str,
    output_path: str,
    subset_ratio: float,
    seed: int,
    label_col: str | None,
) -> None:
    """train.csv からサブセットを作成して output_path に保存する。"""
    df = pd.read_csv(input_path)
    n_total = len(df)

    if n_total == 0:
        raise ValueError(f"{input_path} has 0 rows.")

    n_subset = max(1, int(n_total * subset_ratio))

    if label_col is not None and label_col in df.columns:
        # ラベルごとに層別サンプリング
        df_subset = (
            df.groupby(label_col, group_keys=False)
            .apply(lambda x: x.sample(frac=subset_ratio, random_state=seed))
            .reset_index(drop=True)
        )
        # 端数の影響で n_subset とズレることがあるので、必要なら再度調整
        if len(df_subset) > n_subset:
            df_subset = df_subset.sample(n=n_subset, random_state=seed)
    else:
        # ラベル列が無い / 指定されていない場合は単純ランダムサンプリング
        df_subset = df.sample(frac=subset_ratio, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_subset.to_csv(output_path, index=False)

    print(f"train.csv: {n_total} -> {len(df_subset)} rows (subset_ratio={subset_ratio})")


def copy_if_exists(src: str, dst: str) -> None:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
    else:
        print(f"WARNING: {src} not found, skipping.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a subset of train.csv (and copy dev/test as-is)."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input dir containing train.csv, dev.csv, test.csv",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output dir for subset dataset",
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.2,
        help="Fraction of TRAIN examples to keep (e.g., 0.2 for 20%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of label column for stratified sampling. "
             "If not present, simple random sampling is used.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) train.csv サブセット
    train_in = os.path.join(args.input_dir, "train.csv")
    train_out = os.path.join(args.output_dir, "train.csv")
    if not os.path.exists(train_in):
        raise FileNotFoundError(f"{train_in} not found.")
    make_subset_train(
        input_path=train_in,
        output_path=train_out,
        subset_ratio=args.subset_ratio,
        seed=args.seed,
        label_col=args.label_column,
    )

    # 2) dev/test はそのままコピー
    for name in ["dev.csv", "test.csv"]:
        src = os.path.join(args.input_dir, name)
        dst = os.path.join(args.output_dir, name)
        copy_if_exists(src, dst)


if __name__ == "__main__":
    main()
