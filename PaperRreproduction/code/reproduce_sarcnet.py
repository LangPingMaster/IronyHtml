"""
reproduce_sarcnet.py
Yue et al.（2024）SarcNet 簡化重現入口。

資料來源：
    ACL: https://aclanthology.org/2024.lrec-main.1248/
    Hugging Face: https://huggingface.co/datasets/alita9/sarcnet

本程式提供 text-only XLM-R baseline 的可執行框架。
若要完整重現 DT4MID，請接入 DT4MID / MMID 原模型架構。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from common_utils import ensure_dir


def compute_metrics(eval_pred):
    """Hugging Face Trainer 使用的評估函式。"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def tokenize_dataset(dataset, tokenizer, label_column: str):
    """將 SarcNet 文字資料 tokenization，並指定 label 欄位。

    label_column 可為：
        text_label：文字單模態
        multi_label：多模態整體標籤；若只用文字，這是文字模型對整體標籤的弱 baseline
    """
    def preprocess(example):
        encoded = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
        encoded["labels"] = example[label_column]
        return encoded
    return dataset.map(preprocess, batched=False)


def run_text_baseline(language: str, label_column: str, output: str):
    """執行 SarcNet text-only baseline。"""
    dataset = load_dataset("alita9/sarcnet", language)
    model_name = "FacebookAI/xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, label_column)

    columns_to_remove = [c for c in tokenized["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]]
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir="results/sarcnet_xlmr_ckpt",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(tokenized["test"])

    ensure_dir(str(Path(output).parent))
    pd.DataFrame([metrics]).to_csv(output, index=False, encoding="utf-8-sig")
    print(metrics)
    print(f"結果已輸出：{output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="選擇 SarcNet 語言 subset")
    parser.add_argument("--label_column", default="text_label", choices=["text_label", "multi_label"], help="文字模型要使用哪個標籤")
    parser.add_argument("--output", default="results/sarcnet_results.csv", help="結果 CSV")
    args = parser.parse_args()
    run_text_baseline(args.language, args.label_column, args.output)


if __name__ == "__main__":
    main()
