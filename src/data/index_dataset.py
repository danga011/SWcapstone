#!/usr/bin/env python3
"""
Dataset Indexer for Cascade Anomaly Detection Pipeline.

Recursively scans a dataset root path following the structure:
    dataset_root/dataset_type/<DATASET_NAME>/defect_type/<DEFECT_TYPE>/<LABEL>/

Produces metadata.csv with columns:
    path, dataset_type, defect_type, label, split, height, width

Split policy:
    - train_normal: normal samples only from Kolektor+MVTec
    - val_mix: normal+anomaly samples for validation
    - test_mix: normal+anomaly samples for testing
    - neu_test: NEU anomaly-only samples (if present)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DatasetIndexer:
    """Index datasets and create reproducible train/val/test splits."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    VALID_LABELS = {"normal", "anomaly"}

    def __init__(
        self,
        dataset_root: str,
        seed: int = 42,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        train_sources: Optional[List[str]] = None,
        neu_dataset_name: str = "NEU",
    ):
        """
        Initialize the dataset indexer.

        Args:
            dataset_root: Root directory containing all datasets
            seed: Random seed for reproducibility
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for test
            train_sources: Dataset types to use for training (normal samples)
            neu_dataset_name: Name of the NEU dataset (anomaly-only)
        """
        self.dataset_root = Path(dataset_root).resolve()
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_sources = train_sources or ["Kolektor", "MVTec"]
        self.neu_dataset_name = neu_dataset_name

    def scan_dataset(self) -> pd.DataFrame:
        """
        Recursively scan the dataset root and extract metadata.

        Handles structure: dataset_root/dataset_type/<NAME>/defect_type/<DEFECT>/<LABEL>/

        Returns:
            DataFrame with columns: path, dataset_type, defect_type, label, height, width
        """
        records = []

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        # Handle nested structure: dataset_root/dataset_type/<NAME>/defect_type/<DEFECT>/<LABEL>/
        dataset_type_dir = self.dataset_root / "dataset_type"

        if dataset_type_dir.exists():
            # New structure with dataset_type subdirectory
            base_dir = dataset_type_dir
        else:
            # Fallback: direct structure dataset_root/<NAME>/<DEFECT>/<LABEL>/
            base_dir = self.dataset_root

        # Walk through dataset name directories (MVTec, Kolektor, NEU)
        for dataset_name_dir in sorted(base_dir.iterdir()):
            if not dataset_name_dir.is_dir():
                continue

            dataset_type = dataset_name_dir.name

            # Check for defect_type subdirectory
            defect_type_parent = dataset_name_dir / "defect_type"
            if defect_type_parent.exists():
                defect_base = defect_type_parent
            else:
                defect_base = dataset_name_dir

            # Walk through defect_type directories
            for defect_type_dir in sorted(defect_base.iterdir()):
                if not defect_type_dir.is_dir():
                    continue

                defect_type = defect_type_dir.name

                # Walk through label directories (normal/anomaly)
                for label_dir in sorted(defect_type_dir.iterdir()):
                    if not label_dir.is_dir():
                        continue

                    label = label_dir.name.lower()
                    if label not in self.VALID_LABELS:
                        # Check if this is a subdirectory of anomaly types
                        continue

                    # Scan for images (including subdirectories for anomaly types)
                    for img_path in sorted(label_dir.rglob("*")):
                        if not img_path.is_file():
                            continue
                        if img_path.suffix.lower() not in self.VALID_EXTENSIONS:
                            continue

                        # Get image dimensions
                        try:
                            with Image.open(img_path) as img:
                                width, height = img.size
                        except Exception as e:
                            print(f"Warning: Could not read image {img_path}: {e}")
                            continue

                        records.append({
                            "path": str(img_path),
                            "dataset_type": dataset_type,
                            "defect_type": defect_type,
                            "label": label,
                            "height": height,
                            "width": width,
                        })

        if not records:
            raise ValueError(f"No valid images found in {self.dataset_root}")

        df = pd.DataFrame(records)
        print(f"Scanned {len(df)} images across {df['dataset_type'].nunique()} datasets")
        return df

    def create_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create train/val/test splits following the defined policy.

        Split policy:
            - train_normal: normal samples only from Kolektor+MVTec
            - val_mix: normal+anomaly samples for validation
            - test_mix: normal+anomaly samples for testing
            - neu_test: NEU anomaly-only samples (if present)

        Args:
            df: DataFrame with scanned metadata

        Returns:
            DataFrame with added 'split' column
        """
        df = df.copy()
        df["split"] = None

        # Separate NEU dataset (anomaly-only, goes to neu_test)
        neu_mask = df["dataset_type"].str.lower() == self.neu_dataset_name.lower()
        df.loc[neu_mask, "split"] = "neu_test"

        # Process non-NEU datasets
        non_neu_df = df[~neu_mask].copy()

        if len(non_neu_df) == 0:
            print("Warning: Only NEU dataset found, no train/val/test splits created")
            return df

        # Separate by label
        normal_mask = non_neu_df["label"] == "normal"
        anomaly_mask = non_neu_df["label"] == "anomaly"

        normal_indices = non_neu_df[normal_mask].index.tolist()
        anomaly_indices = non_neu_df[anomaly_mask].index.tolist()

        # Split normal samples
        if len(normal_indices) > 0:
            # First split: train vs (val+test)
            val_test_ratio = self.val_ratio + self.test_ratio
            if val_test_ratio > 0 and len(normal_indices) > 1:
                train_normal_idx, val_test_normal_idx = train_test_split(
                    normal_indices,
                    test_size=min(val_test_ratio, 0.99),
                    random_state=self.seed,
                )

                # Second split: val vs test
                if len(val_test_normal_idx) > 1:
                    relative_test_ratio = self.test_ratio / val_test_ratio
                    val_normal_idx, test_normal_idx = train_test_split(
                        val_test_normal_idx,
                        test_size=min(relative_test_ratio, 0.99),
                        random_state=self.seed,
                    )
                else:
                    val_normal_idx = val_test_normal_idx
                    test_normal_idx = []
            else:
                train_normal_idx = normal_indices
                val_normal_idx = []
                test_normal_idx = []

            # Filter train_normal to only include samples from train_sources
            train_sources_lower = [s.lower() for s in self.train_sources]
            for idx in train_normal_idx:
                if df.loc[idx, "dataset_type"].lower() in train_sources_lower:
                    df.loc[idx, "split"] = "train_normal"
                else:
                    # Non-train-source normal samples go to val_mix
                    val_normal_idx.append(idx)

            for idx in val_normal_idx:
                df.loc[idx, "split"] = "val_mix"
            for idx in test_normal_idx:
                df.loc[idx, "split"] = "test_mix"

        # Split anomaly samples (only into val and test)
        if len(anomaly_indices) > 0:
            if len(anomaly_indices) > 1:
                relative_test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio)
                val_anomaly_idx, test_anomaly_idx = train_test_split(
                    anomaly_indices,
                    test_size=min(relative_test_ratio, 0.99),
                    random_state=self.seed,
                )
            else:
                val_anomaly_idx = anomaly_indices
                test_anomaly_idx = []

            for idx in val_anomaly_idx:
                df.loc[idx, "split"] = "val_mix"
            for idx in test_anomaly_idx:
                df.loc[idx, "split"] = "test_mix"

        # Handle any remaining unassigned samples
        unassigned = df["split"].isna()
        if unassigned.any():
            print(f"Warning: {unassigned.sum()} samples were not assigned to any split")
            df.loc[unassigned, "split"] = "unassigned"

        return df

    def run(self, output_path: str) -> pd.DataFrame:
        """
        Run the full indexing pipeline.

        Args:
            output_path: Path to save the metadata CSV

        Returns:
            DataFrame with full metadata including splits
        """
        print(f"Scanning dataset root: {self.dataset_root}")
        df = self.scan_dataset()

        print("Creating train/val/test splits...")
        df = self.create_splits(df)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        df.to_csv(output_path, index=False)
        print(f"Saved metadata to: {output_path}")

        return df

    @staticmethod
    def print_split_summary(df: pd.DataFrame) -> str:
        """Print and return a summary of the dataset splits."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("DATASET SPLIT SUMMARY")
        lines.append("=" * 60)

        # Overall stats
        lines.append(f"\nTotal samples: {len(df)}")
        lines.append(f"Datasets: {sorted(df['dataset_type'].unique())}")
        lines.append(f"Defect types: {df['defect_type'].nunique()}")

        # Per-split breakdown
        lines.append("\n" + "-" * 60)
        lines.append("Samples per split:")
        lines.append("-" * 60)

        for split in sorted(df["split"].unique()):
            split_df = df[df["split"] == split]
            normal_count = (split_df["label"] == "normal").sum()
            anomaly_count = (split_df["label"] == "anomaly").sum()
            lines.append(f"  {split:15s}: {len(split_df):6d} total "
                        f"(normal: {normal_count:5d}, anomaly: {anomaly_count:5d})")

        # Per-dataset breakdown
        lines.append("\n" + "-" * 60)
        lines.append("Samples per dataset:")
        lines.append("-" * 60)

        for dataset_type in sorted(df["dataset_type"].unique()):
            dataset_df = df[df["dataset_type"] == dataset_type]
            lines.append(f"\n  {dataset_type}:")
            for split in sorted(dataset_df["split"].unique()):
                split_df = dataset_df[dataset_df["split"] == split]
                lines.append(f"    {split:15s}: {len(split_df):5d}")

        lines.append("\n" + "=" * 60)

        summary = "\n".join(lines)
        print(summary)
        return summary


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """CLI entry point for dataset indexing."""
    parser = argparse.ArgumentParser(
        description="Index dataset and create train/val/test splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override dataset root path from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output metadata CSV path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Override validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Override test split ratio",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Path to save split summary text file",
    )

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Warning: Config file not found: {args.config}, using defaults")
        config = {}

    # Apply overrides
    dataset_root = args.dataset_root or config.get("dataset_root", "./data")
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    split_ratios = config.get("split_ratios", {})
    val_ratio = args.val_ratio if args.val_ratio is not None else split_ratios.get("val_ratio", 0.15)
    test_ratio = args.test_ratio if args.test_ratio is not None else split_ratios.get("test_ratio", 0.15)
    train_sources = config.get("train_sources", ["Kolektor", "MVTec"])
    neu_dataset_name = config.get("neu_dataset_name", "NEU")
    output_config = config.get("output", {})
    output_path = args.output or output_config.get("metadata_file", "outputs/metadata.csv")
    summary_path = args.save_summary or output_config.get("split_summary_file")

    # Run indexer
    indexer = DatasetIndexer(
        dataset_root=dataset_root,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        train_sources=train_sources,
        neu_dataset_name=neu_dataset_name,
    )

    df = indexer.run(output_path)
    summary = indexer.print_split_summary(df)

    # Save summary if requested
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Saved split summary to: {summary_path}")

    return df


if __name__ == "__main__":
    main()
