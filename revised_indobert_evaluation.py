# -*- coding: utf-8 -*-
"""
Fine-tune and evaluate IndoBERT on the revised 3-brand sentiment dataset.

Default model:
    indobenchmark/indobert-base-p1

Outputs are written to ./revised_outputs/:
- revised_indobert_holdout_metrics.csv
- revised_indobert_classification_report.csv
- revised_indobert_confusion_matrix.csv
- revised_indobert_training_history.csv
- revised_indobert_run_config.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


RANDOM_STATE = 42
OUTPUT_DIR = Path("revised_outputs")
LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class ReviewDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run revised_absa_pipeline.py before IndoBERT."
        )
    reviews = pd.read_csv(path)
    reviews = reviews.dropna(subset=["text", "sentiment"]).copy()
    reviews = reviews[reviews["sentiment"].isin(LABEL_TO_ID)].copy()
    reviews["text"] = reviews["text"].astype(str)
    reviews["label"] = reviews["sentiment"].map(LABEL_TO_ID)
    return reviews


def maybe_sample_for_smoke_test(reviews: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(reviews):
        return reviews
    parts = []
    for _, group in reviews.groupby("sentiment"):
        n = max(2, round(max_samples * len(group) / len(reviews)))
        parts.append(group.sample(min(n, len(group)), random_state=RANDOM_STATE))
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def encode_dataset(tokenizer, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


def class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    counts = labels.value_counts().sort_index()
    total = counts.sum()
    weights = [total / (len(counts) * counts.get(idx, 1)) for idx in range(len(counts))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, data_loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = [ID_TO_LABEL[idx] for idx in all_labels]
    y_pred = [ID_TO_LABEL[idx] for idx in all_preds]
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": all_probs,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train(args) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_seed(args.seed)

    reviews = load_reviews(Path(args.data))
    reviews = maybe_sample_for_smoke_test(reviews, args.max_samples)

    train_df, test_df = train_test_split(
        reviews,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=reviews["sentiment"],
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    train_encodings = encode_dataset(tokenizer, train_df["text"].tolist(), args.max_length)
    test_encodings = encode_dataset(tokenizer, test_df["text"].tolist(), args.max_length)
    train_dataset = ReviewDataset(train_encodings, train_df["label"].tolist())
    test_dataset = ReviewDataset(test_encodings, test_df["label"].tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_weights = class_weights(train_df["label"], device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

            if step % args.log_every == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        eval_result = evaluate(model, test_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "eval_accuracy": eval_result["accuracy"],
                "eval_f1_weighted": eval_result["f1_weighted"],
                "eval_f1_macro": eval_result["f1_macro"],
            }
        )
        print(
            f"epoch {epoch} done: train_loss={avg_loss:.4f}, "
            f"eval_acc={eval_result['accuracy']:.4f}, "
            f"eval_f1_weighted={eval_result['f1_weighted']:.4f}"
        )

    final_eval = evaluate(model, test_loader, device)
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "revised_indobert_training_history.csv", index=False
    )

    metrics = pd.DataFrame(
        [
            {
                "Model": args.model_name,
                "Accuracy": final_eval["accuracy"],
                "Precision_Weighted": final_eval["precision_weighted"],
                "Recall_Weighted": final_eval["recall_weighted"],
                "F1_Weighted": final_eval["f1_weighted"],
                "Precision_Macro": final_eval["precision_macro"],
                "Recall_Macro": final_eval["recall_macro"],
                "F1_Macro": final_eval["f1_macro"],
                "Train_Size": len(train_df),
                "Test_Size": len(test_df),
                "Epochs": args.epochs,
                "Max_Length": args.max_length,
                "Batch_Size": args.batch_size,
                "Device": str(device),
            }
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "revised_indobert_holdout_metrics.csv", index=False)

    report = classification_report(
        final_eval["y_true"],
        final_eval["y_pred"],
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            report_rows.append(
                {
                    "Class": label,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1-Score": values.get("f1-score"),
                    "Support": values.get("support"),
                }
            )
    pd.DataFrame(report_rows).to_csv(
        OUTPUT_DIR / "revised_indobert_classification_report.csv", index=False
    )

    labels = ["negative", "positive"]
    cm = confusion_matrix(final_eval["y_true"], final_eval["y_pred"], labels=labels)
    cm_rows = []
    for actual_idx, actual in enumerate(labels):
        for predicted_idx, predicted in enumerate(labels):
            cm_rows.append(
                {
                    "Actual": actual,
                    "Predicted": predicted,
                    "Count": int(cm[actual_idx, predicted_idx]),
                }
            )
    pd.DataFrame(cm_rows).to_csv(
        OUTPUT_DIR / "revised_indobert_confusion_matrix.csv", index=False
    )

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "label_mapping": LABEL_TO_ID,
        }
    )
    (OUTPUT_DIR / "revised_indobert_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    if not args.no_save_model:
        model_dir = OUTPUT_DIR / "indobert_finetuned_model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    print("\nIndoBERT evaluation complete.")
    print(metrics.round(4).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT for sentiment classification.")
    parser.add_argument("--data", default=str(OUTPUT_DIR / "revised_clean_reviews.csv"))
    parser.add_argument("--model-name", default="indobenchmark/indobert-base-p1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
