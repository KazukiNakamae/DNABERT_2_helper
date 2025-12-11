#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd


def load_metric_from_eval_results(
    eval_results_path: Path,
    metric_name: str,
) -> Optional[float]:
    """
    Load eval_results.json and return the value of `metric_name`.
    If the key does not exist or is not numeric, return None.
    """
    with eval_results_path.open("r") as f:
        data = json.load(f)

    if metric_name not in data:
        return None

    val = data[metric_name]
    if isinstance(val, (int, float)):
        return float(val)
    return None


def parse_run_name(run_name: str) -> Dict[str, Any]:
    """
    Parse run_name of the form:

      {prefix}_bs{BS}_gas{GAS}_lr{LR}_ep{EPOCHS}_wr{WR}_wd{WD}_g{WORLD_SIZE}

    Example:
      mammal_promoter_bs8_gas8_lr1e-5_ep2_wr0.05_wd0.01_g1

    Returns:
      {
        "bs": 8,
        "gas": 8,
        "lr": 1e-5,
        "epochs": 2,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "world_size": 1,
      }
    """
    patterns = {
        "bs": r"_bs(\d+)",
        "gas": r"_gas(\d+)",
        "lr": r"_lr([^_]+)",
        "epochs": r"_ep(\d+)",
        "warmup_ratio": r"_wr([^_]+)",
        "weight_decay": r"_wd([^_]+)",
        "world_size": r"_g(\d+)",
    }

    result: Dict[str, Any] = {}

    for key, pat in patterns.items():
        m = re.search(pat, run_name)
        if not m:
            continue
        val_str = m.group(1)
        if key in ("bs", "gas", "epochs", "world_size"):
            result[key] = int(val_str)
        else:
            try:
                result[key] = float(val_str)
            except ValueError:
                # Fallback: keep as string if parsing fails (should not happen for typical grids)
                result[key] = val_str

    return result


def compute_warmup_steps(
    train_csv_path: Path,
    bs: int,
    gas: int,
    world_size: int,
    num_train_epochs: int,
    warmup_ratio: float,
) -> Tuple[int, int, int]:
    """
    Compute steps_per_epoch, total_steps, and warmup_steps for full training.

    - N_train = number of rows in train.csv
    - eff_batch = bs * gas * world_size
    - steps_per_epoch = ceil(N_train / eff_batch)
    - total_steps = steps_per_epoch * num_train_epochs
    - warmup_steps = max(1, int(total_steps * warmup_ratio))
    """
    df = pd.read_csv(train_csv_path)
    n_train = len(df)
    if n_train == 0:
        raise ValueError(f"{train_csv_path} has 0 rows.")

    eff_batch = bs * gas * world_size
    if eff_batch <= 0:
        raise ValueError(f"Invalid effective batch size: {eff_batch}")

    steps_per_epoch = math.ceil(n_train / eff_batch)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    return steps_per_epoch, total_steps, warmup_steps


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan grid search runs under output_root, pick the best run "
            "by a given metric from eval_results.json, and print recommended "
            "train.py arguments for full training (with fixed num_train_epochs "
            "and suggested warmup_steps)."
        )
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory containing each grid search run output (subdirectories).",
    )
    parser.add_argument(
        "--metric_name",
        required=True,
        help="Metric key to select the best run "
             "(e.g., 'eval_matthews_correlation', 'eval_f1', 'eval_loss').",
    )
    parser.add_argument(
        "--metric_mode",
        choices=["max", "min"],
        default="max",
        help="Whether higher ('max') or lower ('min') is better for the metric.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=8,
        help="Fixed epoch count to be used in the suggested train.py arguments.",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="train.py",
        help="Path to train.py (used only for the printed command template).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="MODEL_NAME_OR_PATH",
        help="Model name/path placeholder for the printed command template.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to full dataset directory containing train.csv (used to compute warmup_steps).",
    )
    parser.add_argument(
        "--print_full_command",
        action="store_true",
        help="If set, print a template torchrun command using the best hyperparameters.",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"{output_root} does not exist.")

    data_path = Path(args.data_path)
    train_csv_path = data_path / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"{train_csv_path} does not exist.")

    best_run_name: Optional[str] = None
    best_metric_val: Optional[float] = None
    best_hparams: Optional[Dict[str, Any]] = None

    # Walk over each run directory under output_root
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name

        # Expected eval_results.json path:
        # run_dir / "results" / run_name / "eval_results.json"
        eval_results_path = run_dir / "results" / run_name / "eval_results.json"
        if not eval_results_path.exists():
            # Skip if eval_results is not present
            continue

        metric_val = load_metric_from_eval_results(eval_results_path, args.metric_name)
        if metric_val is None:
            continue

        if best_metric_val is None:
            best_metric_val = metric_val
            best_run_name = run_name
            best_hparams = parse_run_name(run_name)
        else:
            if args.metric_mode == "max":
                if metric_val > best_metric_val:
                    best_metric_val = metric_val
                    best_run_name = run_name
                    best_hparams = parse_run_name(run_name)
            else:
                if metric_val < best_metric_val:
                    best_metric_val = metric_val
                    best_run_name = run_name
                    best_hparams = parse_run_name(run_name)

    if best_run_name is None or best_hparams is None or best_metric_val is None:
        print("No valid runs found (no eval_results.json with the given metric).")
        return

    print("=== Best run detected ===")
    print(f"Run directory name   : {best_run_name}")
    print(f"Best {args.metric_name} ({args.metric_mode}) : {best_metric_val:.6f}")
    print()

    print("=== Parsed hyperparameters from run name ===")
    for k, v in best_hparams.items():
        print(f"{k}: {v}")
    print()

    bs = best_hparams.get("bs")
    gas = best_hparams.get("gas")
    lr = best_hparams.get("lr")
    warmup_ratio = best_hparams.get("warmup_ratio")
    wd = best_hparams.get("weight_decay")
    world_size = best_hparams.get("world_size", 1)

    # Compute warmup_steps for full training if warmup_ratio is available
    steps_per_epoch = None
    total_steps = None
    warmup_steps = None
    if warmup_ratio is not None:
        steps_per_epoch, total_steps, warmup_steps = compute_warmup_steps(
            train_csv_path=train_csv_path,
            bs=bs,
            gas=gas,
            world_size=world_size,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=float(warmup_ratio),
        )

    print("=== Suggested train.py arguments (epochs fixed, warmup_steps computed) ===")
    print(f"--per_device_train_batch_size {bs}")
    print(f"--gradient_accumulation_steps {gas}")
    print(f"--learning_rate {lr}")
    print(f"--num_train_epochs {args.num_train_epochs}")
    if wd is not None:
        print(f"--weight_decay {wd}")
    if warmup_ratio is not None and warmup_steps is not None:
        print(f"--warmup_steps {warmup_steps}  # warmup_ratio={warmup_ratio}")
        print(f"# steps_per_epoch (full data) : {steps_per_epoch}")
        print(f"# total_steps (full data)     : {total_steps}")
    else:
        print("# WARNING: warmup_ratio was not found in run name; "
              "warmup_steps is not computed.")
    print(f"# world_size (GPUs): {world_size}")
    print()

    if args.print_full_command:
        print("=== Example torchrun command template ===")
        print(
            "torchrun --nproc_per_node="
            f"{world_size} {args.train_script} \\"
        )
        print(f"  --model_name_or_path {args.model_name_or_path} \\")
        print(f"  --data_path {args.data_path} \\")
        print(f"  --per_device_train_batch_size {bs} \\")
        print(f"  --gradient_accumulation_steps {gas} \\")
        print(f"  --learning_rate {lr} \\")
        print(f"  --num_train_epochs {args.num_train_epochs} \\")
        if wd is not None:
            print(f"  --weight_decay {wd} \\")
        if warmup_steps is not None:
            print(f"  --warmup_steps {warmup_steps} \\")
        print("  # TODO: add other task-specific arguments "
              "(e.g., --kmer, --model_max_length, --fp16, etc.)")


if __name__ == "__main__":
    main()
