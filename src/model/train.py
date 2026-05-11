"""
Fine-tune CodeBERT for bug detection (5-class classification).
Run this on Google Colab with T4 GPU for best results.
On local GTX 1650 reduce batch_size to 8.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import wandb

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME   = "microsoft/codebert-base"
MAX_LENGTH   = 512
NUM_LABELS   = 5
OUTPUT_DIR   = "models/checkpoints"
DATA_PATH    = "data/processed/dataset.csv"
BATCH_SIZE   = 8        # use 16 on Colab T4
NUM_EPOCHS   = 5
LR           = 2e-5
SEED         = 42

LABEL_NAMES  = ["no_bug", "null_dereference", "off_by_one",
                "resource_leak", "logic_error"]

# ── Dataset class ────────────────────────────────────────────────────────────

class CodeDataset(TorchDataset):
    def __init__(self, codes, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            codes, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_per_class = f1_score(labels, preds, average=None, labels=list(range(NUM_LABELS)))
    metrics = {"f1_macro": f1_macro}
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"f1_{name}"] = f1_per_class[i]
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def train():
    # Init W&B
    wandb.init(project="ai-code-reviewer", name="codebert-finetune")

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH).dropna(subset=["code", "label"])
    df["label"] = df["label"].astype(int)

    train_df, temp_df = train_test_split(df, test_size=0.25, random_state=SEED,
                                          stratify=df["label"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.4, random_state=SEED,
                                          stratify=temp_df["label"])

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Build torch datasets
    train_ds = CodeDataset(train_df["code"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH)
    val_ds   = CodeDataset(val_df["code"].tolist(),   val_df["label"].tolist(),   tokenizer, MAX_LENGTH)
    test_ds  = CodeDataset(test_df["code"].tolist(),  test_df["label"].tolist(),  tokenizer, MAX_LENGTH)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )

    # Training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="wandb",
        run_name="codebert-finetune",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_ds)
    print(results)

    # Detailed report
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    print("\n" + classification_report(test_df["label"].tolist(), preds,
                                        target_names=LABEL_NAMES))

    # Save
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
    print(f"\nModel saved to {OUTPUT_DIR}/best_model")

    wandb.finish()


if __name__ == "__main__":
    train()
