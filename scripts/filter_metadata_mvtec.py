#!/usr/bin/env python3
"""
Filter metadata.csv to keep only MVTec samples.

Creates outputs/metadata_mvtec.csv with the same schema but only MVTec rows.
Splits are preserved: train_normal, val_mix, test_mix all contain only MVTec data.

Usage:
    python scripts/filter_metadata_mvtec.py
    python scripts/filter_metadata_mvtec.py --input outputs/metadata.csv --output outputs/metadata_mvtec.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter metadata to MVTec-only samples",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/metadata.csv",
        help="Path to full metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/metadata_mvtec.csv",
        help="Path to save filtered metadata CSV",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="MVTec",
        help="Domain/dataset_type to keep",
    )
    args = parser.parse_args()

    # Load full metadata
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Filter by domain
    filtered = df[df["dataset_type"] == args.domain].copy()
    print(f"Filtered to {len(filtered)} rows for domain '{args.domain}'")

    if len(filtered) == 0:
        print(f"ERROR: No samples found for domain '{args.domain}'")
        print(f"Available domains: {sorted(df['dataset_type'].unique())}")
        return

    # Print split summary
    print("\nSplit summary:")
    for split in sorted(filtered["split"].unique()):
        split_df = filtered[filtered["split"] == split]
        normal = (split_df["label"] == "normal").sum()
        anomaly = (split_df["label"] == "anomaly").sum()
        print(f"  {split:15s}: {len(split_df):5d} total (normal: {normal:4d}, anomaly: {anomaly:4d})")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
