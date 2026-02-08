#!/usr/bin/env python3
"""
Main entrypoint for running cascade anomaly detection experiments.

Usage:
    python -m src.experiments.run_experiments --configs configs/exp_*.yaml
    python -m src.experiments.run_experiments --configs configs/exp_padim.yaml --force
"""

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import AnomalyDataset, create_dataloaders
from src.eval.metrics import (
    compute_metrics,
    create_heatmap_overlay,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_sample_predictions,
)
from src.models.gate import GateModel, GateModelTrainer, create_gate_model
from src.models.heatmap import create_heatmap_model


class ExperimentRunner:
    """Runner for cascade anomaly detection experiments."""

    def __init__(
        self,
        config: Dict[str, Any],
        force_retrain: bool = False,
        device: str = "auto",
    ):
        self.config = config
        self.force_retrain = force_retrain

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.experiment_id = config["experiment_id"]
        self.artifacts_dir = Path(config.get("artifacts_dir", f"models/{self.experiment_id}"))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        print(f"Experiment: {self.experiment_id}")
        print(f"Device: {self.device}")
        print(f"Artifacts dir: {self.artifacts_dir}")

    def _get_gate_model_path(self) -> Path:
        return self.artifacts_dir / "gate_model.pt"

    def _get_heatmap_model_path(self) -> Path:
        return self.artifacts_dir / "heatmap_model.pkl"

    def train_gate_model(self, train_loader: DataLoader, val_loader: DataLoader) -> GateModel:
        """Train or load gate model."""
        gate_config = self.config.get("gate_model")
        if gate_config is None:
            return None

        model_path = self._get_gate_model_path()

        if model_path.exists() and not self.force_retrain:
            print(f"Loading existing gate model from {model_path}")
            model = create_gate_model(gate_config)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model.to(self.device)

        print("Training gate model...")
        model = create_gate_model(gate_config)
        train_config = self.config.get("training", {}).get("gate", {})

        trainer = GateModelTrainer(
            model=model,
            device=str(self.device),
            lr=train_config.get("lr", 0.001),
            weight_decay=train_config.get("weight_decay", 0.0001),
        )

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_config.get("epochs", 20),
            save_path=str(model_path),
        )

        return model

    def train_heatmap_model(self, train_loader: DataLoader):
        """Train or load heatmap model."""
        heatmap_config = self.config["heatmap_model"]
        model_path = self._get_heatmap_model_path()

        if model_path.exists() and not self.force_retrain:
            print(f"Loading existing heatmap model from {model_path}")
            model = create_heatmap_model(heatmap_config, device=str(self.device))
            model.load(str(model_path))
            return model

        print("Training heatmap model...")
        model = create_heatmap_model(heatmap_config, device=str(self.device))
        model.fit(train_loader, save_path=str(model_path))

        return model

    @torch.no_grad()
    def evaluate(
        self,
        gate_model: Optional[GateModel],
        heatmap_model,
        eval_loader: DataLoader,
        split_name: str,
    ) -> Dict[str, Any]:
        """Run evaluation on a split."""
        print(f"\nEvaluating on {split_name}...")

        if gate_model is not None:
            gate_model.eval()

        t_low = self.config["thresholds"]["T_low"]
        t_high = self.config["thresholds"]["T_high"]

        all_labels = []
        all_gate_probs = []
        all_heatmap_scores = []
        all_final_scores = []
        all_predictions = []
        all_heatmaps = []
        all_images = []
        all_paths = []
        latencies = []
        heatmap_calls = 0
        total_samples = 0

        # Denormalization transform for visualization
        denorm = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            ),
        ])

        for images, labels, paths in tqdm(eval_loader, desc=f"Evaluating {split_name}"):
            batch_size = images.size(0)
            total_samples += batch_size
            images_device = images.to(self.device)

            start_time = time.time()

            if gate_model is None:
                # No gate model - run heatmap on all samples
                gate_probs = np.full(batch_size, 0.5)  # Neutral probability
                run_heatmap_mask = np.ones(batch_size, dtype=bool)
            else:
                # Run gate model
                gate_probs = gate_model.predict_proba(images_device).cpu().numpy()
                # Determine which samples need heatmap
                run_heatmap_mask = (gate_probs >= t_low) & (gate_probs <= t_high)

            # Run heatmap model for uncertain samples
            batch_heatmap_scores = np.zeros(batch_size)
            batch_heatmaps = np.zeros((batch_size, 224, 224))

            if run_heatmap_mask.any():
                heatmap_calls += run_heatmap_mask.sum()
                uncertain_images = images_device[torch.from_numpy(run_heatmap_mask)]
                h_scores, h_maps = heatmap_model.predict(uncertain_images)
                batch_heatmap_scores[run_heatmap_mask] = h_scores
                batch_heatmaps[run_heatmap_mask] = h_maps

            # Combine scores: use gate prob for confident predictions, heatmap score for uncertain
            final_scores = np.where(
                gate_probs < t_low,
                gate_probs,  # Confident normal
                np.where(
                    gate_probs > t_high,
                    gate_probs,  # Confident anomaly
                    # Normalize heatmap score to probability-like range
                    0.5 + 0.5 * (batch_heatmap_scores / (batch_heatmap_scores.max() + 1e-8))
                )
            )

            # Final predictions based on threshold 0.5
            predictions = (final_scores >= 0.5).astype(int)

            end_time = time.time()
            latencies.extend([(end_time - start_time) / batch_size * 1000] * batch_size)

            # Store results
            all_labels.extend(labels.numpy().tolist())
            all_gate_probs.extend(gate_probs.tolist())
            all_heatmap_scores.extend(batch_heatmap_scores.tolist())
            all_final_scores.extend(final_scores.tolist())
            all_predictions.extend(predictions.tolist())
            all_heatmaps.extend(batch_heatmaps.tolist())
            all_paths.extend(paths)

            # Store denormalized images for visualization
            for img in images:
                img_denorm = denorm(img).permute(1, 2, 0).numpy()
                img_denorm = np.clip(img_denorm, 0, 1)
                all_images.append((img_denorm * 255).astype(np.uint8))

        # Convert to arrays
        all_labels = np.array(all_labels)
        all_final_scores = np.array(all_final_scores)
        all_predictions = np.array(all_predictions)
        all_heatmaps = np.array(all_heatmaps)
        latencies = np.array(latencies)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_predictions, all_final_scores)

        # Add latency metrics
        metrics["avg_latency_ms"] = float(np.mean(latencies))
        metrics["p95_latency_ms"] = float(np.percentile(latencies, 95))
        metrics["heatmap_call_rate"] = float(heatmap_calls / total_samples)

        # Generate plots
        cm_img = plot_confusion_matrix(
            all_labels, all_predictions,
            title=f"Confusion Matrix - {split_name}",
            save_path=str(self.artifacts_dir / f"confusion_matrix_{split_name}.png"),
        )

        roc_img = plot_roc_curve(
            all_labels, all_final_scores,
            title=f"ROC Curve - {split_name}",
            save_path=str(self.artifacts_dir / f"roc_curve_{split_name}.png"),
        )

        pr_img = plot_pr_curve(
            all_labels, all_final_scores,
            title=f"PR Curve - {split_name}",
            save_path=str(self.artifacts_dir / f"pr_curve_{split_name}.png"),
        )

        # Sample predictions plot
        sample_img = plot_sample_predictions(
            images=all_images,
            heatmaps=all_heatmaps,
            labels=all_labels.tolist(),
            predictions=all_predictions.tolist(),
            scores=all_final_scores.tolist(),
            paths=all_paths,
            num_samples=10,
            save_path=str(self.artifacts_dir / f"sample_predictions_{split_name}.png"),
        )

        return {
            "metrics": metrics,
            "plots": {
                "confusion_matrix": cm_img,
                "roc_curve": roc_img,
                "pr_curve": pr_img,
                "sample_predictions": sample_img,
            },
            "raw_results": {
                "labels": all_labels,
                "predictions": all_predictions,
                "scores": all_final_scores,
                "heatmaps": all_heatmaps,
                "paths": all_paths,
            },
        }

    def run(self) -> Dict[str, Any]:
        """Run the full experiment."""
        # Setup MLflow
        mlflow.set_experiment(self.experiment_id)

        with mlflow.start_run(run_name=f"{self.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log config
            mlflow.log_params({
                "experiment_id": self.experiment_id,
                "seed": self.config["seed"],
                "gate_model": (self.config.get("gate_model") or {}).get("name", "none"),
                "heatmap_model": self.config["heatmap_model"]["name"],
                "heatmap_backbone": self.config["heatmap_model"].get("backbone", "resnet18"),
                "t_low": self.config["thresholds"]["T_low"],
                "t_high": self.config["thresholds"]["T_high"],
            })

            # Set seed
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])

            # Create dataloaders
            metadata_csv = self.config["metadata_csv"]
            image_size = self.config["heatmap_model"].get("image_size", 224)

            train_loader, val_loader, _ = create_dataloaders(
                metadata_csv=metadata_csv,
                train_split="train_normal",
                val_split="val_mix",
                test_split="test_mix",
                batch_size=self.config.get("training", {}).get("batch_size", 32),
                num_workers=self.config.get("training", {}).get("num_workers", 4),
                image_size=image_size,
            )

            # Train gate model (if configured)
            gate_model = self.train_gate_model(train_loader, val_loader)

            # Create normal-only loader for heatmap training
            train_normal_dataset = AnomalyDataset(
                metadata_csv=metadata_csv,
                split="train_normal",
                image_size=image_size,
            )
            train_normal_loader = DataLoader(
                train_normal_dataset,
                batch_size=self.config.get("training", {}).get("heatmap", {}).get("batch_size", 32),
                shuffle=False,
                num_workers=self.config.get("training", {}).get("num_workers", 4),
            )

            # Train heatmap model
            heatmap_model = self.train_heatmap_model(train_normal_loader)

            # Evaluate on each configured split
            all_results = {}
            eval_splits = self.config.get("eval_splits", ["test_mix"])

            for split_name in eval_splits:
                eval_dataset = AnomalyDataset(
                    metadata_csv=metadata_csv,
                    split=split_name,
                    image_size=image_size,
                )

                if len(eval_dataset) == 0:
                    print(f"Skipping {split_name} - no samples found")
                    continue

                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.get("training", {}).get("batch_size", 32),
                    shuffle=False,
                    num_workers=self.config.get("training", {}).get("num_workers", 4),
                )

                results = self.evaluate(gate_model, heatmap_model, eval_loader, split_name)
                all_results[split_name] = results

                # Log metrics to MLflow
                for metric_name, value in results["metrics"].items():
                    mlflow.log_metric(f"{split_name}_{metric_name}", value)

                # Log plots to MLflow
                for plot_name, plot_img in results["plots"].items():
                    mlflow.log_image(plot_img, f"{split_name}_{plot_name}.png")

            # Log artifacts directory
            mlflow.log_artifacts(str(self.artifacts_dir), "model_artifacts")

            # Print summary
            print("\n" + "=" * 60)
            print(f"EXPERIMENT RESULTS: {self.experiment_id}")
            print("=" * 60)

            for split_name, results in all_results.items():
                print(f"\n{split_name}:")
                for metric_name, value in results["metrics"].items():
                    print(f"  {metric_name}: {value:.4f}")

            return all_results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiments(
    config_paths: List[str],
    force_retrain: bool = False,
    device: str = "auto",
) -> pd.DataFrame:
    """Run multiple experiments and generate summary."""
    all_results = []

    for config_path in config_paths:
        print(f"\n{'='*60}")
        print(f"Loading config: {config_path}")
        print("=" * 60)

        config = load_config(config_path)
        runner = ExperimentRunner(
            config=config,
            force_retrain=force_retrain,
            device=device,
        )

        results = runner.run()

        # Collect results for summary
        for split_name, split_results in results.items():
            row = {
                "experiment_id": config["experiment_id"],
                "split": split_name,
                "config_path": config_path,
                **split_results["metrics"],
            }
            all_results.append(row)

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)

    # Save summary
    summary_path = Path("outputs/reports/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run cascade anomaly detection experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="Config file paths (supports glob patterns like configs/exp_*.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if artifacts exist",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )

    args = parser.parse_args()

    # Expand glob patterns
    config_paths = []
    for pattern in args.configs:
        expanded = glob.glob(pattern)
        if expanded:
            config_paths.extend(expanded)
        else:
            config_paths.append(pattern)

    config_paths = sorted(set(config_paths))
    print(f"Found {len(config_paths)} config files:")
    for p in config_paths:
        print(f"  - {p}")

    # Run experiments
    run_experiments(
        config_paths=config_paths,
        force_retrain=args.force,
        device=args.device,
    )


if __name__ == "__main__":
    main()
