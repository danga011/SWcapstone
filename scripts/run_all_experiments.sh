#!/usr/bin/env bash
# Run all core experiment configurations
# Usage: ./scripts/run_all_experiments.sh [--force]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "Cascade Anomaly Detection - Run All Experiments"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for --force flag
FORCE_FLAG=""
if [[ "$1" == "--force" ]]; then
    FORCE_FLAG="--force"
    echo "Force retraining enabled"
fi

# Ensure MLflow tracking directory exists
mkdir -p runs/mlruns

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="file://${PROJECT_ROOT}/runs/mlruns"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo ""

# Run experiments
echo "Running all experiments..."
python3 -m src.experiments.run_experiments \
    --configs \
        configs/exp_padim.yaml \
        configs/exp_gate_effnetb0_patchcore_r18.yaml \
        configs/exp_gate_mnv3_patchcore_r18.yaml \
        configs/exp_gate_effnetb0_patchcore_wrn50.yaml \
    $FORCE_FLAG

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Summary: outputs/reports/summary.csv"
echo "  - MLflow runs: runs/mlruns"
echo "  - Model artifacts: models/"
echo ""
echo "To view MLflow UI:"
echo "  mlflow ui --backend-store-uri file://${PROJECT_ROOT}/runs/mlruns"
