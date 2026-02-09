#!/usr/bin/env python3
"""
Create 3-round splits with a fixed test_mix and round-specific train/val sets.

Round policy (per user requirements):
  - test_mix: 20% of non-NEU data, stratified by label, fixed across rounds
  - neu_test: all NEU anomaly-only samples
  - Round 1: Kolektor only (train/val), MVTec excluded
  - Round 2: Kolektor + 50% of remaining MVTec (train/val)
  - Round 3: Kolektor + all remaining MVTec (train/val)
  - stratify 기준: label 단독
  - split 후 defect_type 분포 로그 출력

Outputs:
  - outputs/metadata_round1.csv
  - outputs/metadata_round2.csv
  - outputs/metadata_round3.csv
  - outputs/splits/round1.json
  - outputs/splits/round2.json
  - outputs/splits/round3.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.index_dataset import DatasetIndexer


def _split_test_mix(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split non-NEU indices into remaining and test_mix, stratified by label."""
    indices = df.index.tolist()
    if len(indices) <= 1:
        return indices, []

    labels = df["label"].values
    train_idx, test_idx = train_test_split(
        indices,
        test_size=min(test_ratio, 0.99),
        random_state=seed,
        stratify=labels,
    )
    return train_idx, test_idx


def _select_mvtec_subset(
    df: pd.DataFrame,
    mvtec_ratio: float,
    seed: int,
) -> List[int]:
    """Select a deterministic subset of MVTec rows, stratified by label."""
    if mvtec_ratio >= 1.0:
        return df.index.tolist()
    if mvtec_ratio <= 0.0 or len(df.index) == 0:
        return []

    indices = df.index.tolist()
    labels = df["label"].values
    train_idx, _ = train_test_split(
        indices,
        train_size=mvtec_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_idx


def _assign_round_splits(
    df_all: pd.DataFrame,
    test_ratio: float,
    val_ratio: float,
    mvtec_ratio: float,
    seed: int,
    round_name: str,
    neu_dataset_name: str = "NEU",
) -> pd.DataFrame:
    df = df_all.copy()
    df["split"] = None

    # NEU anomaly-only
    neu_mask = df["dataset_type"].str.lower() == neu_dataset_name.lower()
    df.loc[neu_mask, "split"] = "neu_test"

    # Non-NEU pool
    non_neu = df[~neu_mask].copy()

    # Fixed test_mix split
    remain_idx, test_idx = _split_test_mix(non_neu, test_ratio, seed)
    df.loc[test_idx, "split"] = "test_mix"

    # Round-specific eligible data (from remaining, non-NEU)
    remain_df = df.loc[remain_idx].copy()
    kolektor_mask = remain_df["dataset_type"].str.lower() == "kolektor"
    mvtec_mask = remain_df["dataset_type"].str.lower() == "mvtec"

    mvtec_df = remain_df[mvtec_mask]
    mvtec_keep_idx = _select_mvtec_subset(mvtec_df, mvtec_ratio, seed)

    eligible_idx = remain_df[kolektor_mask].index.tolist() + mvtec_keep_idx

    eligible_df = df.loc[eligible_idx].copy()
    normal_mask = eligible_df["label"] == "normal"
    anomaly_mask = eligible_df["label"] == "anomaly"

    normal_idx = eligible_df[normal_mask].index.tolist()
    anomaly_idx = eligible_df[anomaly_mask].index.tolist()

    # Split normals into train_normal and val_mix
    if len(normal_idx) > 1 and val_ratio > 0:
        train_normal_idx, val_normal_idx = train_test_split(
            normal_idx,
            test_size=min(val_ratio, 0.99),
            random_state=seed,
        )
    else:
        train_normal_idx = normal_idx
        val_normal_idx = []

    df.loc[train_normal_idx, "split"] = "train_normal"
    df.loc[val_normal_idx, "split"] = "val_mix"
    df.loc[anomaly_idx, "split"] = "val_mix"

    # Any remaining non-NEU samples not used in this round
    unused_mask = df["split"].isna()
    if unused_mask.any():
        df.loc[unused_mask, "split"] = f"unused_{round_name}"

    return df


def _defect_type_distribution(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split]
        dist = split_df.groupby("defect_type").size().to_dict()
        out[split] = {k: int(v) for k, v in dist.items()}
    return out


def _print_defect_type_distribution(df: pd.DataFrame, round_name: str) -> None:
    print(f"\nDefect type distribution ({round_name}):")
    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split]
        dist = split_df.groupby("defect_type").size().sort_values(ascending=False)
        print(f"  {split}:")
        for defect, count in dist.items():
            print(f"    {defect}: {count}")


def _save_json(
    df: pd.DataFrame,
    output_path: Path,
    meta: Dict[str, float],
    round_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": round_name,
        "meta": meta,
        "splits": {
            split: df[df["split"] == split]["path"].tolist()
            for split in sorted(df["split"].unique())
        },
        "defect_type_distribution": _defect_type_distribution(df),
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 3-round dataset splits with fixed test_mix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./final_dataset",
        help="Dataset root path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="test_mix ratio from non-NEU data",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="val_mix ratio from eligible normal data",
    )
    parser.add_argument(
        "--mvtec-round2-ratio",
        type=float,
        default=0.50,
        help="MVTec inclusion ratio for round2",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for metadata and split JSON",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    indexer = DatasetIndexer(
        dataset_root=args.dataset_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    df_all = indexer.scan_dataset()

    output_dir = Path(args.output_dir)
    split_dir = output_dir / "splits"

    rounds = [
        ("round1", 0.0),
        ("round2", args.mvtec_round2_ratio),
        ("round3", 1.0),
    ]

    for round_name, mvtec_ratio in rounds:
        df_round = _assign_round_splits(
            df_all,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            mvtec_ratio=mvtec_ratio,
            seed=args.seed,
            round_name=round_name,
        )

        # Save metadata CSV
        metadata_path = output_dir / f"metadata_{round_name}.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df_round.to_csv(metadata_path, index=False)
        print(f"\nSaved: {metadata_path}")

        # Log defect_type distribution
        _print_defect_type_distribution(df_round, round_name)

        # Save JSON split mapping
        meta = {
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "val_ratio": args.val_ratio,
            "mvtec_round2_ratio": args.mvtec_round2_ratio,
        }
        json_path = split_dir / f"{round_name}.json"
        _save_json(df_round, json_path, meta, round_name)
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
