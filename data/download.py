"""
Download the dair-ai/emotion dataset from HuggingFace.

Saves the dataset locally to data/emotion/ as Arrow files,
and optionally exports to CSV.

Usage:
    python data/download.py
    python data/download.py --csv
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download the emotion dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "emotion"),
        help="Directory to save the dataset (default: data/emotion/)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export the dataset as CSV files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dair-ai/emotion dataset from HuggingFace...")
    dataset = load_dataset("dair-ai/emotion")

    # Save as Arrow format (native, fast)
    dataset.save_to_disk(str(output_dir / "arrow"))
    print(f"Saved Arrow dataset to {output_dir / 'arrow'}")

    # Optionally export as CSV
    if args.csv:
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        for split in dataset:
            path = csv_dir / f"{split}.csv"
            dataset[split].to_csv(str(path))
            print(f"Saved {path} ({len(dataset[split])} rows)")

    # Print stats
    label_names = dataset["train"].features["label"].names
    print(f"\nDataset downloaded successfully!")
    print(f"Labels ({len(label_names)}): {label_names}")
    print(f"\nSplit sizes:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split]):,} samples")

    from collections import Counter
    counts = Counter(dataset["train"]["label"])
    print(f"\nTraining set class distribution:")
    for label_id, count in sorted(counts.items()):
        print(f"  {label_names[label_id]:>10s}: {count:,}")


if __name__ == "__main__":
    main()
