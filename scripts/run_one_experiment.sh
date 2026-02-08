#!/usr/bin/env bash
# Run a single experiment configuration
# Usage: ./scripts/run_one_experiment.sh <config_path> [--force]
# Example: ./scripts/run_one_experiment.sh configs/exp_padim.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check arguments
if [[ -z "$1" ]]; then
    echo "Usage: $0 <config_path> [--force]"
    echo "Example: $0 configs/exp_padim.yaml"
    echo ""
    echo "Available configs:"
    ls -1 configs/exp_*.yaml
    exit 1
fi

CONFIG_PATH="$1"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check for --force flag
FORCE_FLAG=""
if [[ "$2" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

echo "=============================================="
echo "Cascade Anomaly Detection - Run Experiment"
echo "=============================================="
echo "Config: $CONFIG_PATH"
echo ""

# Ensure MLflow tracking directory exists
mkdir -p runs/mlruns

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="file://${PROJECT_ROOT}/runs/mlruns"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo ""

# Run experiment
python3 -m src.experiments.run_experiments \
    --configs "$CONFIG_PATH" \
    $FORCE_FLAG

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - outputs/reports/summary.csv"
echo "  - runs/mlruns"
