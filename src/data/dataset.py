#!/usr/bin/env python3
"""PyTorch Dataset for anomaly detection experiments."""

from pathlib import Path
from typing import Optional, List, Tuple, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection with metadata-based loading."""

    def __init__(
        self,
        metadata_csv: str,
        split: Optional[str] = None,
        splits: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        label_column: str = "label",
        image_size: int = 224,
    ):
        """
        Args:
            metadata_csv: Path to metadata CSV file
            split: Single split name to filter (e.g., "train_normal")
            splits: List of splits to include (overrides split if provided)
            transform: Optional torchvision transform
            label_column: Column name for labels
            image_size: Image size for default transform
        """
        self.df = pd.read_csv(metadata_csv)
        self.label_column = label_column
        self.image_size = image_size

        # Filter by split(s)
        if splits is not None:
            self.df = self.df[self.df["split"].isin(splits)].reset_index(drop=True)
        elif split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        # Create label mapping
        self.label_map = {"normal": 0, "anomaly": 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["path"]
        label_str = row[self.label_column]
        label = self.label_map.get(label_str, 0)

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, img_path

    def get_paths(self) -> List[str]:
        """Return all image paths."""
        return self.df["path"].tolist()

    def get_labels(self) -> List[int]:
        """Return all labels as integers."""
        return [self.label_map.get(l, 0) for l in self.df[self.label_column].tolist()]


def create_dataloaders(
    metadata_csv: str,
    train_split: str = "train_normal",
    val_split: str = "val_mix",
    test_split: str = "test_mix",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = AnomalyDataset(
        metadata_csv, split=train_split, transform=train_transform, image_size=image_size
    )
    val_dataset = AnomalyDataset(
        metadata_csv, split=val_split, transform=eval_transform, image_size=image_size
    )
    test_dataset = AnomalyDataset(
        metadata_csv, split=test_split, transform=eval_transform, image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
