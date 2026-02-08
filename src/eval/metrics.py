#!/usr/bin/env python3
"""Evaluation metrics for anomaly detection experiments."""

import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels (0: normal, 1: anomaly)
        y_pred: Predicted labels
        y_scores: Anomaly scores/probabilities

    Returns:
        Dictionary of metrics
    """
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5  # Default for single-class

    # PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        pr_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Normal", "Anomaly"],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Plot confusion matrix and return as PIL Image.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    return img


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> Image.Image:
    """Plot ROC curve and return as PIL Image."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    return img


def plot_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
) -> Image.Image:
    """Plot Precision-Recall curve and return as PIL Image."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    return img


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Create heatmap overlay on image.

    Args:
        image: Original image [H, W, 3] (0-255 uint8 or 0-1 float)
        heatmap: Anomaly heatmap [H, W]
        alpha: Overlay transparency
        colormap: Matplotlib colormap name

    Returns:
        Overlay image [H, W, 3]
    """
    # Normalize image to 0-1
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Normalize heatmap to 0-1
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_norm)[:, :, :3]  # Remove alpha channel

    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap_colored

    return (overlay * 255).astype(np.uint8)


def plot_sample_predictions(
    images: List[np.ndarray],
    heatmaps: List[np.ndarray],
    labels: List[int],
    predictions: List[int],
    scores: List[float],
    paths: List[str],
    num_samples: int = 10,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Plot grid of sample predictions with heatmaps.
    """
    num_samples = min(num_samples, len(images))

    # Sample indices
    indices = np.random.choice(len(images), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        img = images[idx]
        heatmap = heatmaps[idx]
        label = labels[idx]
        pred = predictions[idx]
        score = scores[idx]
        path = Path(paths[idx]).name

        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n{path[:30]}...")
        axes[i, 0].axis("off")

        # Heatmap
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title(f"Heatmap\nScore: {score:.3f}")
        axes[i, 1].axis("off")

        # Overlay
        overlay = create_heatmap_overlay(img, heatmap)
        axes[i, 2].imshow(overlay)
        label_str = "Anomaly" if label == 1 else "Normal"
        pred_str = "Anomaly" if pred == 1 else "Normal"
        color = "green" if label == pred else "red"
        axes[i, 2].set_title(f"Overlay\nTrue: {label_str}, Pred: {pred_str}", color=color)
        axes[i, 2].axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    return img
