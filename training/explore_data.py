"""
Data exploration and visualization for the emotion dataset.

Generates plots showing label distributions, class balance,
and text length statistics.

Usage:
    python training/explore_data.py
    python training/explore_data.py --output-dir outputs/
"""

import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Explore the emotion dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save plots (default: outputs/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)

    # Print basic info
    print(f"\nLabels ({num_labels}): {label_names}")
    for split in dataset:
        print(f"  {split}: {len(dataset[split]):,} samples")

    print(f"\nSample data:")
    for i in range(5):
        text = dataset["train"][i]["text"]
        label = label_names[dataset["train"][i]["label"]]
        print(f"  [{label:>8s}] {text[:80]}")

    # Class distribution
    counts = Counter(dataset["train"]["label"])
    print(f"\nClass distribution (train):")
    for label_id, count in sorted(counts.items()):
        print(f"  {label_names[label_id]:>10s}: {count:,}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Dataset Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Label distribution across splits
    split_counts = {}
    for split in ["train", "validation", "test"]:
        labels = dataset[split]["label"]
        split_counts[split] = [labels.count(i) for i in range(num_labels)]

    x = np.arange(num_labels)
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (split, cnts) in enumerate(split_counts.items()):
        axes[0].bar(x + i * width, cnts, width, label=split, color=colors[i])
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(label_names, rotation=30)
    axes[0].set_title("Label Distribution by Split")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Plot 2: Pie chart
    train_counts = split_counts["train"]
    palette = sns.color_palette("Set2", num_labels)
    axes[1].pie(
        train_counts, labels=label_names, autopct="%1.1f%%",
        colors=palette, startangle=90,
    )
    axes[1].set_title("Training Set Class Balance")

    # Plot 3: Text length distribution
    train_texts = dataset["train"]["text"]
    lengths = [len(t.split()) for t in train_texts]
    train_labels = dataset["train"]["label"]

    for i, name in enumerate(label_names):
        class_lengths = [l for l, lab in zip(lengths, train_labels) if lab == i]
        axes[2].hist(class_lengths, bins=30, alpha=0.5, label=name, density=True)
    axes[2].set_title("Text Length Distribution (words)")
    axes[2].set_xlabel("Number of words")
    axes[2].set_ylabel("Density")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "data_exploration.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to {save_path}")

    print(
        f"\nText length stats (words): "
        f"min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}"
    )


if __name__ == "__main__":
    main()
