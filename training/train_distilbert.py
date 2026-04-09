"""
Fine-tune DistilBERT for emotion detection.

Trains the model, evaluates on test set, generates all plots
(training curves, confusion matrix, per-class metrics, confidence analysis),
and saves the model to models/emotion_model/.

Usage:
    python training/train_distilbert.py
    python training/train_distilbert.py --epochs 5 --batch-size 32 --lr 3e-5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


def plot_training_curves(trainer, output_dir):
    log_history = trainer.state.log_history

    train_steps = [x["step"] for x in log_history if "loss" in x and "eval_loss" not in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]

    eval_entries = [x for x in log_history if "eval_loss" in x]
    eval_epochs = [x["epoch"] for x in eval_entries]
    eval_losses = [x["eval_loss"] for x in eval_entries]
    eval_accuracies = [x["eval_accuracy"] for x in eval_entries]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

    axes[0].plot(train_steps, train_losses, color="#4C72B0", linewidth=1.5, alpha=0.8)
    axes[0].set_title("Training Loss vs Steps")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_epochs, eval_losses, "o-", color="#C44E52", linewidth=2, markersize=8)
    axes[1].set_title("Validation Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(eval_epochs)
    axes[1].grid(True, alpha=0.3)
    for e, l in zip(eval_epochs, eval_losses):
        axes[1].annotate(f"{l:.4f}", (e, l), textcoords="offset points", xytext=(0, 12), ha="center")

    axes[2].plot(eval_epochs, eval_accuracies, "o-", color="#55A868", linewidth=2, markersize=8)
    axes[2].set_title("Validation Accuracy per Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xticks(eval_epochs)
    axes[2].set_ylim(0.85, 1.0)
    axes[2].grid(True, alpha=0.3)
    for e, a in zip(eval_epochs, eval_accuracies):
        axes[2].annotate(f"{a:.4f}", (e, a), textcoords="offset points", xytext=(0, 12), ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training_curves.png")


def plot_confusion_matrix(y_true, y_pred, label_names, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix - Test Set", fontsize=16, fontweight="bold")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=axes[1])
    axes[1].set_title("Normalized (Recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion_matrix.png")


def plot_per_class_metrics(y_true, y_pred, label_names, output_dir):
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    num_labels = len(label_names)

    metrics_df = pd.DataFrame({
        "Emotion": label_names,
        "Precision": [report[l]["precision"] for l in label_names],
        "Recall": [report[l]["recall"] for l in label_names],
        "F1-Score": [report[l]["f1-score"] for l in label_names],
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(num_labels)
    width = 0.25

    bars1 = ax.bar(x - width, metrics_df["Precision"], width, label="Precision", color="#4C72B0")
    bars2 = ax.bar(x, metrics_df["Recall"], width, label="Recall", color="#55A868")
    bars3 = ax.bar(x + width, metrics_df["F1-Score"], width, label="F1-Score", color="#C44E52")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1-Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=30)
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per_class_metrics.png")
    print(f"\n{metrics_df.to_string(index=False)}")


def plot_confidence(y_true, y_pred, y_probs, label_names, output_dir):
    confidences = y_probs.max(axis=1)
    correct_mask = y_pred == y_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prediction Confidence Analysis", fontsize=16, fontweight="bold")

    axes[0].hist(confidences[correct_mask], bins=30, alpha=0.7, label="Correct", color="#55A868", density=True)
    axes[0].hist(confidences[~correct_mask], bins=30, alpha=0.7, label="Incorrect", color="#C44E52", density=True)
    axes[0].set_title("Confidence: Correct vs Incorrect")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    class_confs = [confidences[y_true == i].mean() for i in range(len(label_names))]
    colors = ["#55A868" if c > 0.9 else "#F0E442" if c > 0.8 else "#C44E52" for c in class_confs]
    axes[1].barh(label_names, class_confs, color=colors)
    axes[1].set_title("Average Confidence per Class")
    axes[1].set_xlabel("Confidence")
    axes[1].set_xlim(0.7, 1.0)
    axes[1].grid(axis="x", alpha=0.3)
    for i, v in enumerate(class_confs):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confidence_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for emotion detection")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--model-dir", type=str, default="models/emotion_model")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (requires GPU)")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("dair-ai/emotion")
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    print(f"Labels ({num_labels}): {label_names}")

    # Tokenize
    print("Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, batch_size=512)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    # Load model
    print("Loading DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)},
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Training
    use_fp16 = args.fp16 or torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr}, fp16={use_fp16})...")
    train_result = trainer.train()

    runtime = train_result.metrics["train_runtime"]
    print(f"\nTraining complete in {runtime:.1f}s ({runtime/60:.1f} min)")
    print(f"Final loss: {train_result.metrics['train_loss']:.4f}")

    # Plot training curves
    plot_training_curves(trainer, args.output_dir)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"])
    print(f"Test Loss:     {test_results['eval_loss']:.4f}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")

    # Predictions + plots
    preds_output = trainer.predict(tokenized["test"])
    y_pred = preds_output.predictions.argmax(axis=1)
    y_true = preds_output.label_ids
    y_probs = torch.softmax(torch.tensor(preds_output.predictions), dim=1).numpy()

    print(f"\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    plot_confusion_matrix(y_true, y_pred, label_names, args.output_dir)
    plot_per_class_metrics(y_true, y_pred, label_names, args.output_dir)
    plot_confidence(y_true, y_pred, y_probs, label_names, args.output_dir)

    # Save model
    print(f"\nSaving model to {args.model_dir}/...")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    for f in sorted(os.listdir(args.model_dir)):
        size = os.path.getsize(os.path.join(args.model_dir, f)) / 1e6
        print(f"  {f} ({size:.2f} MB)")

    print(f"\nDone! Model saved to {args.model_dir}/")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
