"""
Train an LSTM model for emotion detection using TensorFlow/Keras.

Usage:
    python training/train_lstm.py
    python training/train_lstm.py --epochs 20 --batch-size 32
"""

import argparse
import os

import numpy as np
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Train LSTM emotion model")
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Deferred import so --help is fast
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")
    label_names = dataset["train"].features["label"].names

    X_train = dataset["train"]["text"]
    y_train = np.array(dataset["train"]["label"])
    X_val = dataset["validation"]["text"]
    y_val = np.array(dataset["validation"]["label"])

    # Tokenize
    print(f"Tokenizing (vocab_size={args.vocab_size})...")
    tokenizer = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    X_train_pad = pad_sequences(X_train_seq, maxlen=args.max_len, padding="post")
    X_val_pad = pad_sequences(X_val_seq, maxlen=args.max_len, padding="post")

    # Build model
    print("Building LSTM model...")
    model = Sequential([
        Embedding(input_dim=args.vocab_size, output_dim=args.embedding_dim),
        LSTM(args.lstm_units),
        Dropout(args.dropout),
        Dense(len(label_names), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(
        X_train_pad, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val_pad, y_val),
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val_pad, y_val)
    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Loss:     {val_loss:.4f}")
    print(f"Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")


if __name__ == "__main__":
    main()
