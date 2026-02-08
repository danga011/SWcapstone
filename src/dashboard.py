#!/usr/bin/env python3
"""
Streamlit dashboard for anomaly detection results visualization.

Usage:
    streamlit run src/dashboard.py

Features:
    - View experiment results summary
    - Browse sample predictions with heatmaps
    - Compare experiments
    - View MLflow metrics
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Page config
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)


def load_summary():
    """Load experiment summary CSV."""
    summary_path = Path("outputs/reports/summary.csv")
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def load_experiment_images(experiment_id: str, split: str = "test_mix"):
    """Load experiment visualization images."""
    artifacts_dir = Path(f"models/{experiment_id}")
    images = {}

    image_names = [
        f"confusion_matrix_{split}.png",
        f"roc_curve_{split}.png",
        f"pr_curve_{split}.png",
        f"sample_predictions_{split}.png",
    ]

    for name in image_names:
        path = artifacts_dir / name
        if path.exists():
            images[name.replace(f"_{split}.png", "")] = Image.open(path)

    return images


def main():
    st.title("🔍 Anomaly Detection Dashboard")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Summary", "Experiment Details", "Compare Experiments"],
    )

    # Load summary
    summary_df = load_summary()

    if page == "Summary":
        st.header("Experiment Summary")

        if summary_df is None:
            st.warning("No experiment results found. Run experiments first:")
            st.code("./scripts/run_all_experiments.sh")
            return

        # Display summary table
        st.subheader("Results Table")
        st.dataframe(
            summary_df.style.format({
                "accuracy": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1": "{:.4f}",
                "auroc": "{:.4f}",
                "pr_auc": "{:.4f}",
                "avg_latency_ms": "{:.2f}",
                "p95_latency_ms": "{:.2f}",
                "heatmap_call_rate": "{:.2%}",
            }),
            use_container_width=True,
        )

        # Key metrics comparison
        st.subheader("Key Metrics Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Best AUROC", f"{summary_df['auroc'].max():.4f}")
            best_auroc_exp = summary_df.loc[summary_df['auroc'].idxmax(), 'experiment_id']
            st.caption(f"Experiment: {best_auroc_exp}")

        with col2:
            st.metric("Best F1", f"{summary_df['f1'].max():.4f}")
            best_f1_exp = summary_df.loc[summary_df['f1'].idxmax(), 'experiment_id']
            st.caption(f"Experiment: {best_f1_exp}")

        # Latency comparison
        st.subheader("Latency Analysis")
        latency_df = summary_df[['experiment_id', 'split', 'avg_latency_ms', 'p95_latency_ms', 'heatmap_call_rate']]
        st.bar_chart(latency_df.set_index('experiment_id')[['avg_latency_ms', 'p95_latency_ms']])

    elif page == "Experiment Details":
        st.header("Experiment Details")

        if summary_df is None:
            st.warning("No experiments found.")
            return

        # Select experiment
        experiments = summary_df['experiment_id'].unique().tolist()
        selected_exp = st.selectbox("Select Experiment", experiments)

        # Select split
        splits = summary_df[summary_df['experiment_id'] == selected_exp]['split'].unique().tolist()
        selected_split = st.selectbox("Select Split", splits)

        # Display metrics
        exp_data = summary_df[
            (summary_df['experiment_id'] == selected_exp) &
            (summary_df['split'] == selected_split)
        ].iloc[0]

        st.subheader("Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUROC", f"{exp_data['auroc']:.4f}")
        col2.metric("F1", f"{exp_data['f1']:.4f}")
        col3.metric("Precision", f"{exp_data['precision']:.4f}")
        col4.metric("Recall", f"{exp_data['recall']:.4f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Latency", f"{exp_data['avg_latency_ms']:.2f} ms")
        col2.metric("P95 Latency", f"{exp_data['p95_latency_ms']:.2f} ms")
        col3.metric("Heatmap Call Rate", f"{exp_data['heatmap_call_rate']:.2%}")

        # Load and display images
        st.subheader("Visualizations")
        images = load_experiment_images(selected_exp, selected_split)

        if images:
            col1, col2 = st.columns(2)
            if "confusion_matrix" in images:
                col1.image(images["confusion_matrix"], caption="Confusion Matrix")
            if "roc_curve" in images:
                col2.image(images["roc_curve"], caption="ROC Curve")

            col1, col2 = st.columns(2)
            if "pr_curve" in images:
                col1.image(images["pr_curve"], caption="PR Curve")

            if "sample_predictions" in images:
                st.subheader("Sample Predictions")
                st.image(images["sample_predictions"], caption="Sample Predictions with Heatmaps")
        else:
            st.info("No visualization images found for this experiment.")

    elif page == "Compare Experiments":
        st.header("Compare Experiments")

        if summary_df is None:
            st.warning("No experiments found.")
            return

        # Multi-select experiments
        experiments = summary_df['experiment_id'].unique().tolist()
        selected_exps = st.multiselect(
            "Select Experiments to Compare",
            experiments,
            default=experiments[:2] if len(experiments) >= 2 else experiments,
        )

        if len(selected_exps) < 2:
            st.warning("Select at least 2 experiments to compare.")
            return

        # Filter data
        compare_df = summary_df[summary_df['experiment_id'].isin(selected_exps)]

        # Comparison table
        st.subheader("Comparison Table")
        pivot_df = compare_df.pivot(
            index='experiment_id',
            columns='split',
            values=['auroc', 'f1', 'avg_latency_ms', 'heatmap_call_rate'],
        )
        st.dataframe(pivot_df, use_container_width=True)

        # Metric comparison charts
        st.subheader("Metric Comparison")

        metric = st.selectbox(
            "Select Metric",
            ['auroc', 'f1', 'precision', 'recall', 'pr_auc', 'avg_latency_ms'],
        )

        chart_df = compare_df[['experiment_id', 'split', metric]].pivot(
            index='experiment_id',
            columns='split',
            values=metric,
        )
        st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
