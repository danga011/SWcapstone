#!/bin/bash
# Run threshold sweep for cascade anomaly detection experiments
# This script finds optimal T_low/T_high operating points

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running threshold sweep..."
echo "Project root: $PROJECT_ROOT"
echo ""

python -m src.experiments.threshold_sweep --config configs/threshold_sweep.yaml "$@"

echo ""
echo "Results saved to:"
echo "  - outputs/reports/threshold_sweep_results.csv (all combinations)"
echo "  - outputs/reports/threshold_sweep_best.csv (selected operating points)"
