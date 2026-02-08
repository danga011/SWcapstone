#!/usr/bin/env bash
# Script to index datasets and create train/val/test splits
# Usage: ./scripts/01_index_dataset.sh [dataset_root]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default dataset root, can be overridden by argument
DATASET_ROOT="${1:-./final_dataset}"

echo "=============================================="
echo "Cascade Anomaly Detection - Dataset Indexing"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo ""

# Check if dataset root exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "Error: Dataset root '$DATASET_ROOT' does not exist."
    echo "Please ensure your dataset follows the structure:"
    echo "  dataset_root/dataset_type/<DATASET_NAME>/defect_type/<DEFECT_TYPE>/<LABEL>/"
    echo ""
    echo "Example:"
    echo "  final_dataset/"
    echo "  └── dataset_type/"
    echo "      ├── Kolektor/"
    echo "      │   └── defect_type/"
    echo "      │       └── surface_defect/"
    echo "      │           ├── normal/"
    echo "      │           └── anomaly/"
    echo "      ├── MVTec/"
    echo "      │   └── defect_type/"
    echo "      │       └── metal_nut/"
    echo "      │           ├── normal/"
    echo "      │           └── anomaly/"
    echo "      └── NEU/"
    echo "          └── defect_type/"
    echo "              └── Crazing/"
    echo "                  └── anomaly/"
    echo ""
    exit 1
fi

# Run the indexing script
echo "Running dataset indexer..."
python3 -m src.data.index_dataset \
    --config configs/dataset.yaml \
    --dataset-root "$DATASET_ROOT" \
    --output outputs/metadata.csv \
    --save-summary outputs/split_summary.txt

echo ""
echo "=============================================="
echo "Dataset indexing complete!"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  - outputs/metadata.csv        (full metadata)"
echo "  - outputs/split_summary.txt   (split statistics)"
echo ""

# Print counts per split using pandas
echo "Quick verification - sample counts per split:"
python3 -c "
import pandas as pd
df = pd.read_csv('outputs/metadata.csv')
print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
"

echo ""
echo "To run sanity check:"
echo "  python -m src.data.sanity_check --num-samples 10"
