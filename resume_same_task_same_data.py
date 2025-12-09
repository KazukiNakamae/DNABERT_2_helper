#!/usr/bin/env python
# resume_same_task_same_data.py
"""
Resume DNABERT-2 fine-tuning on the same task and same dataset
from an existing checkpoint directory.
Supports fp16 and gradient_accumulation_steps.
"""

import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def tokenize_function(examples, tokenizer, text_column):
    return tokenizer(
        examples[text_column],
        padding="max_length",
        truncation=True,
        max_length=512,  # Adjust if you use shorter sequences
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resume DNABERT-2 training from a checkpoint on the same task and data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Base DNABERT-2 model or initial checkpoint path/name."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory to resume from (e.g., checkpoint-5000)."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training data file (CSV or TSV)."
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        required=True,
        help="Validation data file (CSV or TSV)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save final model and checkpoints."
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        required=True,
        help="Number of labels for classification."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (AFTER resuming)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device batch size."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for the learning rate scheduler."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="sequence",
        help="Column name containing DNA sequences."
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Column name containing labels."
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision (fp16) training."
    )
    args = parser.parse_args()

    # 1. Load dataset
    data_files = {
        "train": args.train_file,
        "validation": args.valid_file,
    }
    extension = os.path.splitext(args.train_file)[-1].lstrip(".")
    if extension == "tsv":
        extension = "csv"
        sep = "\t"
    else:
        sep = ","

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        delimiter=sep,
    )

    # 2. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
    )

    # 3. Tokenize
    def _tokenize(examples):
        result = tokenize_function(examples, tokenizer, args.text_column)
        result["labels"] = examples[args.label_column]
        return result

    tokenized_datasets = raw_datasets.map(
        _tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # 4. Training arguments (fp16 & grad_accum_steps supported)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=5,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=args.seed,
        gradient_accumulation_steps=args.grad_accum_steps,
        fp16=args.fp16,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 6. Resume from checkpoint
    trainer.train(resume_from_checkpoint=args.checkpoint_dir)

    # 7. Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
