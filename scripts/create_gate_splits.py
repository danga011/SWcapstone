#!/usr/bin/env python3
"""
Gate 모델 학습을 위한 train_mix / val_gate 분할 생성.

기존 val_mix(정상+이상 혼합)를 80:20으로 나눠서:
- train_mix: Gate 학습용 (정상 + 이상 혼합)
- val_gate: Gate 검증용 (정상 + 이상 혼합)

기존 split(train_normal, test_mix, neu_test)은 그대로 유지.

Usage:
    python scripts/create_gate_splits.py
    python scripts/create_gate_splits.py --input outputs/metadata.csv --output outputs/metadata.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Create train_mix/val_gate splits for Gate model training",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/metadata.csv",
        help="Path to save updated metadata CSV (overwrites input by default)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of val_mix to use for train_mix (default: 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load metadata
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Check if splits already exist
    if "train_mix" in df["split"].values:
        print("train_mix split already exists. Skipping.")
        return

    # Split val_mix into train_mix + val_gate
    val_mix = df[df["split"] == "val_mix"].copy()
    print(f"\nOriginal val_mix: {len(val_mix)} rows")
    print(f"  normal: {(val_mix['label'] == 'normal').sum()}")
    print(f"  anomaly: {(val_mix['label'] == 'anomaly').sum()}")

    train_idx, val_idx = train_test_split(
        val_mix.index,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=val_mix["label"],  # 레이블 비율 유지
    )

    # Update splits
    df.loc[train_idx, "split"] = "train_mix"
    df.loc[val_idx, "split"] = "val_gate"

    # Print results
    train_mix = df[df["split"] == "train_mix"]
    val_gate = df[df["split"] == "val_gate"]

    print(f"\nCreated splits:")
    print(f"  train_mix: {len(train_mix)} rows "
          f"(normal: {(train_mix['label'] == 'normal').sum()}, "
          f"anomaly: {(train_mix['label'] == 'anomaly').sum()})")
    print(f"  val_gate:  {len(val_gate)} rows "
          f"(normal: {(val_gate['label'] == 'normal').sum()}, "
          f"anomaly: {(val_gate['label'] == 'anomaly').sum()})")

    # Print all splits summary
    print(f"\nAll splits:")
    for split in sorted(df["split"].unique()):
        sdf = df[df["split"] == split]
        normal = (sdf["label"] == "normal").sum()
        anomaly = (sdf["label"] == "anomaly").sum()
        print(f"  {split:15s}: {len(sdf):5d} total (normal: {normal:4d}, anomaly: {anomaly:4d})")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
