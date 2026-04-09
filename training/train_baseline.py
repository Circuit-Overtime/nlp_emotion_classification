"""
Train a TF-IDF + Logistic Regression baseline model for emotion detection.

Usage:
    python training/train_baseline.py
    python training/train_baseline.py --max-features 15000 --ngram-max 3
"""

import argparse
import os

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def main():
    parser = argparse.ArgumentParser(description="Train baseline emotion model")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")
    label_names = dataset["train"].features["label"].names

    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]
    X_val = dataset["validation"]["text"]
    y_val = dataset["validation"]["label"]

    # TF-IDF vectorization
    print(f"Vectorizing (max_features={args.max_features}, ngram_range=(1, {args.ngram_max}))...")
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        stop_words="english",
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # Train
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n{classification_report(y_val, y_pred, target_names=label_names)}")


if __name__ == "__main__":
    main()
