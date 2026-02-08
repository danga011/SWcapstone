#!/usr/bin/env python3
"""
Standalone PaDiM training script.

This script maintains backward compatibility with existing commands:
    python src/train.py

For new experiments, use:
    python -m src.experiments.run_experiments --configs configs/exp_padim.yaml
"""

import argparse
import os
import pickle
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import AnomalyDataset
from src.models.heatmap import PaDiM


def main():
    parser = argparse.ArgumentParser(
        description="Train PaDiM model for anomaly detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset config",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="outputs/metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/padim_stats.pkl",
        help="Output path for model",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "wide_resnet50_2"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PaDiM Training")
    print("=" * 60)
    print(f"Metadata: {args.metadata}")
    print(f"Backbone: {args.backbone}")
    print(f"Image size: {args.image_size}")
    print(f"Output: {args.output}")
    print()

    # Create dataset and dataloader
    train_dataset = AnomalyDataset(
        metadata_csv=args.metadata,
        split="train_normal",
        image_size=args.image_size,
    )

    print(f"Training samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create and train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = PaDiM(
        backbone=args.backbone,
        layers=["layer1", "layer2", "layer3"],
        image_size=args.image_size,
        device=device,
    )

    model.fit(train_loader, save_path=args.output)

    print()
    print("=" * 60)
    print(f"Model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
