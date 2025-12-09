#!/usr/bin/env python
"""
Evaluate a fine-tuned DNABERT-2 (or any HF classifier) on a labeled test set,
and run prediction on an unlabeled CSV.

Modes:
  - eval:    requires a CSV with columns [sequence, label]
  - predict: requires a CSV with column [sequence]
"""

import argparse
import os
import math

import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def softmax_logits(logits):
    """Convert logits (tensor) to probabilities."""
    # logits: [batch_size, num_labels]
    probs = F.softmax(logits, dim=-1)
    return probs


def batched_predict(model, tokenizer, texts, device, batch_size=32, max_length=512):
    """
    Run batched prediction on a list of DNA sequences.
    Returns:
      - all_probs: [N, num_labels] tensor (on CPU)
      - all_preds: [N] tensor of predicted class indices (on CPU)
    """
    all_probs = []
    all_preds = []

    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start:start + batch_size])
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits  # [B, num_labels]
            probs = softmax_logits(logits)  # [B, num_labels]
            preds = torch.argmax(probs, dim=-1)  # [B]

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    return all_probs, all_preds


def run_eval(model_dir, test_file, output_prefix, text_column, label_column,
             batch_size=32, max_length=512, average_f1="binary"):
    """
    Evaluate on a labeled test set and save metrics + per-sample predictions.

    test_file: CSV/TSV with columns [text_column, label_column]
    output_prefix: path prefix for outputs (without extension)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    # Detect delimiter by extension
    ext = os.path.splitext(test_file)[-1].lower()
    if ext == ".tsv":
        df = pd.read_csv(test_file, sep="\t")
    else:
        df = pd.read_csv(test_file)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Columns '{text_column}' and/or '{label_column}' not found in {test_file}. "
            f"Available columns: {df.columns.tolist()}"
        )

    texts = df[text_column].astype(str).tolist()
    true_labels = df[label_column].tolist()

    probs, preds = batched_predict(
        model,
        tokenizer,
        texts,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Convert to Python types
    pred_labels = preds.tolist()
    num_labels = probs.shape[1]

    # Metrics
    # For binary classification, average="binary" for f1, precision, recall.
    # For multi-class, use "macro" or "weighted".
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels, average=average_f1),
        "matthews_correlation": matthews_corrcoef(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average=average_f1),
        "recall": recall_score(true_labels, pred_labels, average=average_f1),
    }

    print("Metrics:", metrics)

    # Save metrics as a single-row CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = f"{output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save per-sample predictions
    # Add prediction columns to df
    df["true_label"] = true_labels
    df["pred_label"] = pred_labels

    # Add probability columns: prob_0, prob_1, ...
    for cls_idx in range(num_labels):
        df[f"prob_{cls_idx}"] = probs[:, cls_idx].numpy()

    preds_path = f"{output_prefix}_predictions.csv"
    df.to_csv(preds_path, index=False)

    return metrics, metrics_path, preds_path


def run_predict(model_dir, input_file, output_file, text_column,
                batch_size=32, max_length=512):
    """
    Prediction-only mode on unlabeled CSV/TSV.
    input_file: CSV/TSV with [text_column] column.
    Saves a CSV with [text_column, pred_label, prob_0, prob_1, ...].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    # Detect delimiter
    ext = os.path.splitext(input_file)[-1].lower()
    if ext == ".tsv":
        df = pd.read_csv(input_file, sep="\t")
    else:
        df = pd.read_csv(input_file)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {input_file}. "
            f"Available columns: {df.columns.tolist()}"
        )

    texts = df[text_column].astype(str).tolist()

    probs, preds = batched_predict(
        model,
        tokenizer,
        texts,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    pred_labels = preds.tolist()
    num_labels = probs.shape[1]

    out_df = df.copy()
    out_df["pred_label"] = pred_labels
    for cls_idx in range(num_labels):
        out_df[f"prob_{cls_idx}"] = probs[:, cls_idx].numpy()

    out_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or predict with a fine-tuned DNABERT-2/Transformers classifier."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned model directory (compatible with AutoModelForSequenceClassification).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["eval", "predict"],
        help="'eval' for labeled test set, 'predict' for unlabeled CSV.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="CSV/TSV file for evaluation or prediction.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="results",
        help="Output prefix for 'eval' mode (metrics & predictions). "
             "For 'predict' mode, acts as full output filename if it ends with .csv/.tsv.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="sequence",
        help="Column name containing DNA sequences.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Column name containing labels (eval mode only).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenizer.",
    )
    parser.add_argument(
        "--average_f1",
        type=str,
        default="binary",
        help="Averaging method for f1/precision/recall (e.g., 'binary', 'macro', 'weighted').",
    )
    args = parser.parse_args()

    if args.mode == "eval":
        # output_prefix is used as is
        metrics, metrics_path, preds_path = run_eval(
            model_dir=args.model_dir,
            test_file=args.input_file,
            output_prefix=args.output_prefix,
            text_column=args.text_column,
            label_column=args.label_column,
            batch_size=args.batch_size,
            max_length=args.max_length,
            average_f1=args.average_f1,
        )
        print("Finished eval. Metrics written to:", metrics_path)
        print("Per-sample predictions written to:", preds_path)

    elif args.mode == "predict":
        # Determine output file path
        if args.output_prefix.endswith(".csv") or args.output_prefix.endswith(".tsv"):
            out_file = args.output_prefix
        else:
            out_file = args.output_prefix + "_predictions.csv"

        run_predict(
            model_dir=args.model_dir,
            input_file=args.input_file,
            output_file=out_file,
            text_column=args.text_column,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
