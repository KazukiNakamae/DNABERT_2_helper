#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = ["accuracy", "f1", "matthews_correlation", "precision", "recall"]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot per-metric heatmaps from merged metrics CSV; "
            "show dataset sample size as '(n=...)' under dataset labels; "
            "move baseline model(s) to the top and mark them."
        )
    )
    p.add_argument("--input_csv", required=True, help="Merged metrics CSV (from merge_metrics_csvs.py).")
    p.add_argument("--outdir", default="heatmaps_out", help="Output directory.")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                   help="Comma-separated metrics to plot.")
    p.add_argument("--agg", choices=["mean", "median"], default="mean",
                   help="Aggregation across reps for each (model,dataset).")
    p.add_argument("--dpi", type=int, default=500, help="DPI for raster outputs (PNG/TIFF).")
    p.add_argument("--formats", default="png,tiff,pdf",
                   help="Comma-separated output formats. e.g. 'png,tiff,pdf'")
    p.add_argument("--annotate", action="store_true", help="Annotate each cell with numeric value.")
    p.add_argument("--annot_fmt", default="{:.3f}", help="Annotation format string.")
    p.add_argument("--cmap", default=None, help="Matplotlib colormap name (optional).")

    p.add_argument("--n_col", default="n", help="Sample size column name in merged CSV. Default: 'n'")
    p.add_argument("--dataset_n_agg", choices=["max", "median", "min", "first"], default="max",
                   help="How to choose dataset-level n if inconsistent. Default: max")

    # Baseline controls
    p.add_argument(
        "--baseline_models",
        default="",
        help="Comma-separated list of baseline model names to put at the top, in the given order."
    )
    p.add_argument(
        "--baseline_model",
        action="append",
        default=[],
        help="Specify a baseline model name. Can be repeated. Takes precedence order as provided."
    )
    p.add_argument(
        "--baseline_suffix",
        default=" [baseline]",
        help="Suffix appended to y-axis labels for baseline models."
    )
    p.add_argument(
        "--draw_baseline_separator",
        action="store_true",
        help="Draw a horizontal separator line under the baseline block."
    )

    # scale settings
    p.add_argument("--vmin", type=float, default=None,
                   help="Color scale min for metrics except MCC unless overridden.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Color scale max for metrics except MCC unless overridden.")
    p.add_argument("--mcc_vmin", type=float, default=-1.0, help="MCC color scale min.")
    p.add_argument("--mcc_vmax", type=float, default=1.0, help="MCC color scale max.")

    p.add_argument("--title_prefix", default="", help="Optional figure title prefix.")
    return p.parse_args()


def parse_baselines(args) -> List[str]:
    baselines: List[str] = []
    # --baseline_model (repeatable) in the given order
    for b in args.baseline_model:
        if b and b not in baselines:
            baselines.append(b)

    # --baseline_models comma-separated
    if args.baseline_models:
        for b in [x.strip() for x in args.baseline_models.split(",") if x.strip()]:
            if b not in baselines:
                baselines.append(b)

    return baselines


def wrap_label(s: str, width: int = 22) -> str:
    if len(s) <= width:
        return s
    toks = s.split("_")
    lines = []
    cur = ""
    for t in toks:
        if not cur:
            cur = t
        elif len(cur) + 1 + len(t) <= width:
            cur = f"{cur}_{t}"
        else:
            lines.append(cur)
            cur = t
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def get_scale(metric: str, args) -> Tuple[float, float]:
    if "matthews" in metric.lower() or metric.lower() in ("mcc", "matthews_correlation"):
        return args.mcc_vmin, args.mcc_vmax

    vmin = 0.0 if args.vmin is None else args.vmin
    vmax = 1.0 if args.vmax is None else args.vmax
    return vmin, vmax


def compute_dataset_n(df: pd.DataFrame, dataset_col: str, n_col: str, how: str) -> pd.Series:
    d = df[[dataset_col, n_col]].copy()
    d[n_col] = pd.to_numeric(d[n_col], errors="coerce")

    var_check = (
        d.dropna()
         .groupby(dataset_col)[n_col]
         .apply(lambda x: sorted(set([int(v) for v in x.values if np.isfinite(v)])))
    )

    for ds, uniq in var_check.items():
        if len(uniq) > 1:
            print(f"[WARN] dataset '{ds}' has multiple n values: {uniq} -> using {how}")

    if how == "min":
        return d.groupby(dataset_col)[n_col].min()
    if how == "median":
        return d.groupby(dataset_col)[n_col].median()
    if how == "first":
        return d.groupby(dataset_col)[n_col].first()
    return d.groupby(dataset_col)[n_col].max()


def reorder_models(index_models: List[str], baseline_models: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (new_order, missing_baselines).
    Baselines first in user-specified order; remaining models keep original order.
    """
    present = set(index_models)
    missing = [b for b in baseline_models if b not in present]

    baseline_present = [b for b in baseline_models if b in present]
    rest = [m for m in index_models if m not in baseline_present]
    return baseline_present + rest, missing


def plot_heatmap(matrix: pd.DataFrame, metric: str, outdir: str, args, dataset_n: pd.Series, baseline_models: List[str]):
    data = matrix.to_numpy(dtype=float)
    nrows, ncols = data.shape

    fig_w = max(6.5, 0.62 * ncols)
    fig_h = max(4.5, 0.55 * nrows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin, vmax = get_scale(metric, args)

    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", vmin=vmin, vmax=vmax, cmap=args.cmap)

    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))

    # x labels: dataset + (n=...)
    xticklabels = []
    for ds in matrix.columns:
        n_val = dataset_n.get(ds, np.nan)
        if np.isfinite(n_val):
            label = f"{wrap_label(str(ds), 26)}\n(n={int(n_val)})"
        else:
            label = f"{wrap_label(str(ds), 26)}\n(n=NA)"
        xticklabels.append(label)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    # y labels: baseline marked
    ylabels = []
    baseline_set = set(baseline_models)
    for m in matrix.index:
        if m in baseline_set:
            ylabels.append(f"{m}{args.baseline_suffix}")
        else:
            ylabels.append(str(m))
    ax.set_yticklabels(ylabels)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model")

    title = metric
    if args.title_prefix:
        title = f"{args.title_prefix} {title}".strip()
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)

    # baseline separator line
    if args.draw_baseline_separator:
        # baseline rows are the first k rows, if baseline_models present in matrix order
        k = 0
        for m in matrix.index:
            if m in baseline_set:
                k += 1
            else:
                break
        if k > 0 and k < nrows:
            # draw between row k-1 and k => y = k-0.5
            ax.axhline(y=k - 0.5, linewidth=1.0)

    if args.annotate:
        for i in range(nrows):
            for j in range(ncols):
                val = data[i, j]
                if np.isfinite(val):
                    ax.text(j, i, args.annot_fmt.format(val), ha="center", va="center", fontsize=8)

    fig.tight_layout()

    Path(outdir).mkdir(parents=True, exist_ok=True)
    base = os.path.join(outdir, f"heatmap_{metric}")
    for fmt in [f.strip().lower() for f in args.formats.split(",") if f.strip()]:
        if fmt in ("png", "tif", "tiff"):
            fig.savefig(f"{base}.{fmt}", dpi=args.dpi, bbox_inches="tight")
        else:
            fig.savefig(f"{base}.{fmt}", bbox_inches="tight")

    plt.close(fig)


def main():
    args = parse_args()
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    baseline_models = parse_baselines(args)

    df = pd.read_csv(args.input_csv)

    required = {"model", "dataset", "rep", args.n_col}
    missing_req = required - set(df.columns)
    if missing_req:
        raise SystemExit(f"Missing required columns in merged CSV: {sorted(missing_req)}")

    # dataset-level n (for x tick labels)
    dataset_n = compute_dataset_n(df, dataset_col="dataset", n_col=args.n_col, how=args.dataset_n_agg)

    # aggregate across reps per (model,dataset)
    agg_func = "mean" if args.agg == "mean" else "median"
    g = df.groupby(["model", "dataset"], as_index=False)[metrics].agg(agg_func)

    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)
    g.to_csv(os.path.join(outdir, "aggregated_model_dataset_metrics.csv"), index=False)

    # for each metric: pivot -> reorder models -> save matrix -> plot
    for metric in metrics:
        if metric not in g.columns:
            raise SystemExit(f"Metric column not found after aggregation: {metric}")

        pivot = g.pivot(index="model", columns="dataset", values=metric)

        # reorder models so baselines come first
        new_order, missing_baselines = reorder_models(list(pivot.index), baseline_models)
        if missing_baselines:
            print(f"[WARN] baseline model(s) not found in data for metric '{metric}': {missing_baselines}")

        pivot = pivot.reindex(new_order)

        # store matrix (reordered)
        pivot.to_csv(os.path.join(outdir, f"matrix_{metric}.csv"))

        plot_heatmap(pivot, metric, outdir, args, dataset_n=dataset_n, baseline_models=baseline_models)

    print(f"[OK] input: {args.input_csv}")
    print(f"[OK] outdir: {outdir}")
    print(f"[OK] metrics: {metrics}")
    print(f"[OK] aggregation across reps: {args.agg}")
    if baseline_models:
        print(f"[OK] baseline models (top, marked): {baseline_models}")


if __name__ == "__main__":
    main()
