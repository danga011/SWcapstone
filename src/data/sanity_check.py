#!/usr/bin/env python3
"""
Sanity check script: Load samples from metadata and verify data integrity.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Optional torch import
try:
    import torch
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_sample(image_path: str, use_torch: bool = False):
    """Load a single image and optionally apply transforms."""
    img = Image.open(image_path).convert("RGB")

    if use_torch and HAS_TORCH:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(img)
    else:
        # Return as numpy array
        img_resized = img.resize((224, 224))
        return np.array(img_resized)


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: load samples and print shapes/labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="outputs/metadata.csv",
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to load",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Filter by specific split (e.g., train_normal, val_mix)",
    )

    args = parser.parse_args()

    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        print("Run 'python -m src.data.index_dataset' first to create metadata.")
        return 1

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(df)} samples")

    # Filter by split if specified
    if args.split:
        df = df[df["split"] == args.split]
        if len(df) == 0:
            print(f"Error: No samples found for split '{args.split}'")
            return 1
        print(f"Filtered to {len(df)} samples in split '{args.split}'")

    # Sample random images
    random.seed(args.seed)
    num_samples = min(args.num_samples, len(df))
    sample_indices = random.sample(range(len(df)), num_samples)
    samples = df.iloc[sample_indices]

    print("\n" + "=" * 80)
    print(f"Loading {num_samples} random samples...")
    print(f"PyTorch available: {HAS_TORCH}")
    print("=" * 80)

    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = row["path"]

        # Check if file exists
        if not Path(img_path).exists():
            print(f"\n[{i+1}/{num_samples}] FILE NOT FOUND: {img_path}")
            continue

        # Load image
        try:
            data = load_sample(img_path, use_torch=HAS_TORCH)

            print(f"\n[{i+1}/{num_samples}]")
            print(f"  Path:         {img_path}")
            print(f"  Dataset:      {row['dataset_type']}")
            print(f"  Defect type:  {row['defect_type']}")
            print(f"  Label:        {row['label']}")
            print(f"  Split:        {row['split']}")
            print(f"  Original:     {row['width']}x{row['height']}")

            if HAS_TORCH:
                print(f"  Tensor shape: {tuple(data.shape)} (C, H, W)")
                print(f"  Tensor dtype: {data.dtype}")
                print(f"  Value range:  [{data.min():.3f}, {data.max():.3f}]")
            else:
                print(f"  Array shape:  {data.shape} (H, W, C)")
                print(f"  Array dtype:  {data.dtype}")
                print(f"  Value range:  [{data.min()}, {data.max()}]")

        except Exception as e:
            print(f"\n[{i+1}/{num_samples}] ERROR loading {img_path}: {e}")

    print("\n" + "=" * 80)
    print("Sanity check complete!")
    print("=" * 80)

    # Print summary statistics
    full_df = pd.read_csv(metadata_path)
    print("\nDataset statistics:")
    print(f"  Total samples:     {len(full_df)}")
    print(f"  Splits:            {sorted(full_df['split'].unique())}")
    print("\nSamples per split:")
    print(full_df.groupby(['split', 'label']).size().unstack(fill_value=0))

    return 0


if __name__ == "__main__":
    exit(main())
