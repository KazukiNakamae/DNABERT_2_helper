#!/usr/bin/env python3
"""
Prediction-only script for a fine-tuned Hugging Face classifier (e.g., DNABERT-2).

Input : CSV/TSV with at least [sequence] column (labels may exist but are ignored)
Output: CSV/TSV with added columns:
  - sample_id (if not present)
  - pred_label
  - prob_0, prob_1, ... (num_labels dependent)
"""

import argparse
import os
from typing import Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def read_table(path: str) -> Tuple[pd.DataFrame, str]:
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    return df, sep


def write_table(df: pd.DataFrame, path: str, sep: str) -> None:
    df.to_csv(path, index=False, sep=sep)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_and_tokenizer(model_dir: str, device: torch.device, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def batched_predict(model, tokenizer, texts, device: torch.device, batch_size: int, max_length: int):
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

        outputs = model(**enc)
        logits = outputs.logits  # [B, num_labels]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    return all_probs, all_preds


def main():
    parser = argparse.ArgumentParser(description="Prediction-only for HF sequence classifier.")
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)

    parser.add_argument("--text_column", default="sequence", type=str)
    parser.add_argument("--id_column", default="sample_id", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=512, type=int)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], type=str)
    parser.add_argument("--trust_remote_code", action="store_true")

    args = parser.parse_args()

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device, trust_remote_code=args.trust_remote_code)

    df, sep_in = read_table(args.input_file)
    sep_out = "\t" if args.output_file.lower().endswith(".tsv") else ","

    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column '{args.text_column}'. Available: {df.columns.tolist()}")

    out_df = df.copy()

    # Ensure stable alignment key for later evaluation
    if args.id_column not in out_df.columns:
        out_df[args.id_column] = list(range(len(out_df)))

    texts = out_df[args.text_column].astype(str).tolist()
    probs, preds = batched_predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    out_df["pred_label"] = preds.numpy().tolist()

    num_labels = probs.shape[1]
    for i in range(num_labels):
        out_df[f"prob_{i}"] = probs[:, i].numpy()

    write_table(out_df, args.output_file, sep_out)
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
