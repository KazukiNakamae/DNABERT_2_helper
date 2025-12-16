#!/usr/bin/env python3
# resume_dnabert2_seqcls.py

import os
import csv
import json
import pathlib
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, Any, Union, List

import torch
import numpy as np
import sklearn.metrics
import transformers
from torch.utils.data import Dataset


# ---- Minimal dataset / collator (close to your original pipeline) ----

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        with open(data_path, "r") as f:
            rows = list(csv.reader(f))[1:]  # skip header

        # Expect: sequence,label
        texts = [r[0] for r in rows]
        labels = [int(r[1]) for r in rows]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        self.input_ids = enc["input_ids"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[i], "labels": torch.tensor(self.labels[i]).long()}


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in instances]
        labels = torch.stack([x["labels"] for x in instances], dim=0)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---- Metrics (optional but helpful) ----

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "f1": sklearn.metrics.f1_score(labels, preds, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, preds),
        "precision": sklearn.metrics.precision_score(labels, preds, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="e.g. /path/to/output/checkpoint-1800")
    ap.add_argument("--data_path", required=True, help="Dir containing train.csv/dev.csv/test.csv")
    ap.add_argument("--output_dir", default=None, help="Where to write new checkpoints/final model")
    ap.add_argument("--add_epochs", type=int, default=1, help="Additional epochs AFTER resuming")
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--do_test_eval", action="store_true", help="Evaluate on test.csv at the end")

    args = ap.parse_args()

    ckpt = os.path.abspath(args.checkpoint_dir)
    parent = os.path.dirname(ckpt)
    outdir = os.path.abspath(args.output_dir) if args.output_dir else parent

    p = pathlib.Path(os.path.join(ckpt,"trainer_state.json"))
    st = json.loads(p.read_text())
    st["best_model_checkpoint"] = None
    st["best_metric"] = None
    p.write_text(json.dumps(st, indent=2))
    print("patched:", p)

    # 1) Load tokenizer/model from checkpoint (most reliable for head/labels consistency)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt, use_fast=True, trust_remote_code=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(ckpt, trust_remote_code=True)

    # 2) Build datasets (same CSV layout)
    train_csv = os.path.join(args.data_path, "train.csv")
    dev_csv   = os.path.join(args.data_path, "dev.csv")
    test_csv  = os.path.join(args.data_path, "test.csv")

    train_ds = SupervisedDataset(train_csv, tokenizer)
    dev_ds   = SupervisedDataset(dev_csv, tokenizer)
    test_ds  = SupervisedDataset(test_csv, tokenizer)

    collator = DataCollatorForSupervisedDataset(tokenizer)

    # 3) TrainingArguments
    #    NOTE: For strict resume, keep settings consistent with the original run.
    #    This script keeps them simple; if you need exact matching, pass the same values as before.
    training_args = transformers.TrainingArguments(
        optim="adamw_torch",
        output_dir=outdir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=100,
        num_train_epochs=args.add_epochs,  # "additional" epochs (not total); Trainer will resume state from ckpt
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    # 4) Resume
    trainer.train(resume_from_checkpoint=ckpt)  # official supported pattern :contentReference[oaicite:4]{index=4}

    # 5) Save final (explicit)
    trainer.save_model(outdir)
    tokenizer.save_pretrained(outdir)

    if args.do_test_eval:
        results = trainer.evaluate(eval_dataset=test_ds)
        os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
        with open(os.path.join(outdir, "results", "eval_results_test.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
