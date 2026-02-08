#!/usr/bin/env python3
"""
Threshold sweep for cascade anomaly detection experiments.

Evaluates different T_low/T_high threshold combinations to find optimal
operating points that balance recall, heatmap_call_rate, and latency.

Usage:
    python -m src.experiments.threshold_sweep --config configs/threshold_sweep.yaml
"""

import argparse
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AnomalyDataset
from src.eval.metrics import compute_metrics
from src.models.gate import create_gate_model
from src.models.heatmap import create_heatmap_model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ThresholdSweepRunner:
    """Runner for threshold sweep evaluation."""

    def __init__(
        self,
        experiment_config: Dict[str, Any],
        device: str = "auto",
    ):
        self.config = experiment_config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.experiment_id = experiment_config["experiment_id"]
        self.artifacts_dir = Path(experiment_config.get("artifacts_dir", f"models/{self.experiment_id}"))

        self.gate_model = None
        self.heatmap_model = None

    def load_models(self) -> bool:
        """Load pre-trained gate and heatmap models. Returns True if successful."""
        gate_path = self.artifacts_dir / "gate_model.pt"
        heatmap_path = self.artifacts_dir / "heatmap_model.pkl"

        if not gate_path.exists():
            print(f"  Warning: Gate model not found at {gate_path}")
            return False

        if not heatmap_path.exists():
            print(f"  Warning: Heatmap model not found at {heatmap_path}")
            return False

        # Load gate model
        gate_config = self.config.get("gate_model")
        if gate_config:
            self.gate_model = create_gate_model(gate_config)
            checkpoint = torch.load(gate_path, map_location=self.device)
            self.gate_model.load_state_dict(checkpoint["model_state_dict"])
            self.gate_model = self.gate_model.to(self.device)
            self.gate_model.eval()

        # Load heatmap model
        heatmap_config = self.config["heatmap_model"]
        self.heatmap_model = create_heatmap_model(heatmap_config, device=str(self.device))
        self.heatmap_model.load(str(heatmap_path))

        return True

    @torch.no_grad()
    def evaluate_with_thresholds(
        self,
        eval_loader: DataLoader,
        t_low: float,
        t_high: float,
        use_calibrated: bool = True,
        temperature: float = 10.0,
    ) -> Dict[str, float]:
        """Evaluate with specific threshold values.

        Args:
            eval_loader: DataLoader for evaluation
            t_low: Lower threshold - below this, classify as normal
            t_high: Upper threshold - above this, classify as anomaly
            use_calibrated: If True, use calibrated sigmoid scores instead of softmax
            temperature: Temperature for calibrated scoring (only if use_calibrated=True)
        """
        all_labels = []
        all_final_scores = []
        all_predictions = []
        latencies = []
        heatmap_calls = 0
        total_samples = 0

        for images, labels, paths in eval_loader:
            batch_size = images.size(0)
            total_samples += batch_size
            images_device = images.to(self.device)

            start_time = time.time()

            # Run gate model
            if self.gate_model is not None:
                if use_calibrated:
                    gate_scores = self.gate_model.get_calibrated_scores(
                        images_device, temperature=temperature
                    ).cpu().numpy()
                else:
                    gate_scores = self.gate_model.predict_proba(images_device).cpu().numpy()
            else:
                gate_scores = np.full(batch_size, 0.5)

            # Determine which samples need heatmap (uncertain region)
            run_heatmap_mask = (gate_scores >= t_low) & (gate_scores <= t_high)

            # Run heatmap model for uncertain samples
            batch_heatmap_scores = np.zeros(batch_size)

            if run_heatmap_mask.any():
                heatmap_calls += run_heatmap_mask.sum()
                uncertain_images = images_device[torch.from_numpy(run_heatmap_mask)]
                h_scores, _ = self.heatmap_model.predict(uncertain_images)
                batch_heatmap_scores[run_heatmap_mask] = h_scores

            # Combine scores
            final_scores = np.where(
                gate_scores < t_low,
                gate_scores,  # Confident normal
                np.where(
                    gate_scores > t_high,
                    gate_scores,  # Confident anomaly
                    0.5 + 0.5 * (batch_heatmap_scores / (batch_heatmap_scores.max() + 1e-8))
                )
            )

            predictions = (final_scores >= 0.5).astype(int)

            end_time = time.time()
            latencies.extend([(end_time - start_time) / batch_size * 1000] * batch_size)

            all_labels.extend(labels.numpy().tolist())
            all_final_scores.extend(final_scores.tolist())
            all_predictions.extend(predictions.tolist())

        # Convert to arrays
        all_labels = np.array(all_labels)
        all_final_scores = np.array(all_final_scores)
        all_predictions = np.array(all_predictions)
        latencies = np.array(latencies)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_predictions, all_final_scores)

        # Add latency and call rate metrics
        metrics["avg_latency_ms"] = float(np.mean(latencies))
        metrics["p95_latency_ms"] = float(np.percentile(latencies, 95))
        metrics["heatmap_call_rate"] = float(heatmap_calls / total_samples)

        return metrics


def run_threshold_sweep(sweep_config_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the threshold sweep and return results."""
    sweep_config = load_config(sweep_config_path)

    t_high_values = sweep_config["sweep"]["T_high"]
    t_low_values = sweep_config["sweep"]["T_low"]
    use_calibrated = sweep_config["sweep"].get("use_calibrated", True)
    temperature = sweep_config["sweep"].get("temperature", 50.0)
    target_experiments = sweep_config["target_experiments"]
    splits_to_eval = sweep_config.get("splits_to_eval", ["test_mix"])
    objective = sweep_config.get("objective", {})

    results = []

    for exp_config_path in target_experiments:
        print(f"\n{'='*60}")
        print(f"Processing: {exp_config_path}")
        print("=" * 60)

        exp_config = load_config(exp_config_path)
        experiment_id = exp_config["experiment_id"]

        runner = ThresholdSweepRunner(exp_config)

        if not runner.load_models():
            print(f"  Skipping {experiment_id} - models not found")
            continue

        print(f"  Models loaded successfully")

        # Create dataloaders for each split
        metadata_csv = exp_config["metadata_csv"]
        image_size = exp_config["heatmap_model"].get("image_size", 224)
        batch_size = exp_config.get("training", {}).get("batch_size", 32)

        for split_name in splits_to_eval:
            try:
                eval_dataset = AnomalyDataset(
                    metadata_csv=metadata_csv,
                    split=split_name,
                    image_size=image_size,
                )

                if len(eval_dataset) == 0:
                    print(f"  Skipping {split_name} - no samples")
                    continue

                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                )

                print(f"\n  Evaluating on {split_name} ({len(eval_dataset)} samples)")
                print(f"  Sweeping {len(t_high_values)} T_high x {len(t_low_values)} T_low = {len(t_high_values) * len(t_low_values)} combinations")

                # Sweep all threshold combinations
                for t_high, t_low in tqdm(
                    list(product(t_high_values, t_low_values)),
                    desc=f"  {experiment_id}/{split_name}",
                ):
                    # Skip invalid combinations where t_low > t_high
                    if t_low > t_high:
                        continue

                    metrics = runner.evaluate_with_thresholds(
                        eval_loader, t_low, t_high,
                        use_calibrated=use_calibrated,
                        temperature=temperature,
                    )

                    results.append({
                        "experiment_id": experiment_id,
                        "model": exp_config.get("gate_model", {}).get("name", "unknown"),
                        "split": split_name,
                        "T_low": t_low,
                        "T_high": t_high,
                        **metrics,
                    })

            except Exception as e:
                print(f"  Error processing {split_name}: {e}")
                continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo results generated!")
        return pd.DataFrame(), pd.DataFrame()

    # Save full results
    output_config = sweep_config.get("output", {})
    results_path = Path(output_config.get("results_csv", "outputs/reports/threshold_sweep_results.csv"))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved full results to {results_path}")

    # Select best operating points
    best_results = select_best_operating_points(results_df, objective)

    best_path = Path(output_config.get("best_csv", "outputs/reports/threshold_sweep_best.csv"))
    best_results.to_csv(best_path, index=False)
    print(f"Saved best operating points to {best_path}")

    return results_df, best_results


def select_best_operating_points(
    results_df: pd.DataFrame,
    objective: Dict[str, Any],
) -> pd.DataFrame:
    """Select best operating point for each experiment based on objective criteria."""
    constraint = objective.get("constraint", {})
    call_rate_min = constraint.get("heatmap_call_rate_min", 0.20)
    call_rate_max = constraint.get("heatmap_call_rate_max", 0.60)

    best_points = []

    for exp_id in results_df["experiment_id"].unique():
        exp_results = results_df[results_df["experiment_id"] == exp_id].copy()

        # Filter for test_mix split (primary evaluation split)
        exp_results = exp_results[exp_results["split"] == "test_mix"]

        if exp_results.empty:
            continue

        # Apply constraint: heatmap_call_rate within range
        constrained = exp_results[
            (exp_results["heatmap_call_rate"] >= call_rate_min) &
            (exp_results["heatmap_call_rate"] <= call_rate_max)
        ]

        if constrained.empty:
            # Fallback: find closest to constraint range
            print(f"  Warning: No points in call_rate range [{call_rate_min}, {call_rate_max}] for {exp_id}")
            exp_results["call_rate_dist"] = exp_results["heatmap_call_rate"].apply(
                lambda x: min(abs(x - call_rate_min), abs(x - call_rate_max))
            )
            constrained = exp_results.nsmallest(5, "call_rate_dist")

        # Sort by: recall (desc), distance from 0.40 call_rate (asc), latency (asc)
        constrained = constrained.copy()
        constrained["call_rate_distance"] = abs(constrained["heatmap_call_rate"] - 0.40)

        best = constrained.sort_values(
            by=["recall", "call_rate_distance", "avg_latency_ms"],
            ascending=[False, True, True],
        ).iloc[0]

        best_points.append(best.to_dict())

    return pd.DataFrame(best_points)


def print_top_candidates(results_df: pd.DataFrame, n: int = 10):
    """Print top N candidates sorted by objective criteria."""
    print("\n" + "=" * 100)
    print("TOP CANDIDATES (sorted by: recall desc, |call_rate-0.40| asc, latency asc)")
    print("=" * 100)

    # Filter for test_mix only
    test_results = results_df[results_df["split"] == "test_mix"].copy()

    if test_results.empty:
        print("No test_mix results found")
        return

    # Add distance from ideal call rate
    test_results["call_rate_dist"] = abs(test_results["heatmap_call_rate"] - 0.40)

    # Sort by criteria
    sorted_results = test_results.sort_values(
        by=["recall", "call_rate_dist", "avg_latency_ms"],
        ascending=[False, True, True],
    )

    # Display top N
    display_cols = [
        "experiment_id", "T_low", "T_high",
        "accuracy", "precision", "recall", "f1",
        "heatmap_call_rate", "avg_latency_ms",
    ]

    print(sorted_results.head(n)[display_cols].to_string(index=False))


def print_best_operating_points(best_df: pd.DataFrame):
    """Print the selected best operating points."""
    print("\n" + "=" * 100)
    print("SELECTED OPERATING POINTS")
    print("=" * 100)

    display_cols = [
        "experiment_id", "T_low", "T_high",
        "accuracy", "precision", "recall", "f1",
        "heatmap_call_rate", "avg_latency_ms",
    ]

    available_cols = [c for c in display_cols if c in best_df.columns]
    print(best_df[available_cols].to_string(index=False))


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run threshold sweep for cascade anomaly detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/threshold_sweep.yaml",
        help="Path to threshold sweep config file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to display",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("THRESHOLD SWEEP")
    print("=" * 60)
    print(f"Config: {args.config}")

    results_df, best_df = run_threshold_sweep(args.config)

    if not results_df.empty:
        print_top_candidates(results_df, n=args.top)

    if not best_df.empty:
        print_best_operating_points(best_df)


if __name__ == "__main__":
    main()
