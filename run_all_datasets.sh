#!/bin/bash

# Specify GPU index (e.g., 0, 1, or 0,1)
# Or specify via command line: CUDA_VISIBLE_DEVICES=0 bash run_all_datasets.sh
# export CUDA_VISIBLE_DEVICES=0

# =================================================================================
# 🚀 Batch script to automatically run run_grid_search.py sequentially on all core datasets
# =================================================================================

# Full paths for all core datasets defined in datasets_overview.md
DATASETS=(
    # "datasets/h5ad/D1/RNA+ADT"             # Vertical,     1 batch,  26,169 cells, 17 cell types
    # "datasets/h5ad/D15/RNA+ATAC"           # Vertical,     1 batch,  10,717 cells, 13 cell types
    "datasets/h5ad/D18/RNA+ATAC"           # Vertical,     1 batch,   5,036 cells, 11 cell types
    # "datasets/h5ad/D53/RNA+ADT"            # Cross-batch,  2 batches, 140,951 cells, 31 cell types
    # "datasets/h5ad/D54/RNA+ADT"            # Cross-batch, 12 batches,  73,280 cells, 45 cell types
    # "datasets/h5ad/D56/RNA+ATAC"           # Cross-batch, 13 batches,  61,590 cells, 22 cell types
)

TOTAL=${#DATASETS[@]}
COUNT=1

echo "====================================================================="
echo "           scMMA Automated Grid Search (All Datasets)                "
echo "====================================================================="
echo "📊 A total of $TOTAL datasets will be passed sequentially to run_grid_search.py..."
echo "====================================================================="
sleep 3

for DATASET in "${DATASETS[@]}"
do
    echo ""
    echo "====================================================================="
    echo ">>> 🌟 Scheduling dataset [$COUNT/$TOTAL]: $DATASET"
    echo "====================================================================="
    
    # We modified run_grid_search.py to accept command-line arguments and append them to BASE_OVERRIDES
    # This leverages Hydra's override mechanism to dynamically change data.data_dir
    python run_grid_search.py "data.data_dir=$DATASET"
    
    # Capture exit code
    STATUS=$?
    if [ $STATUS -eq 130 ]; then
        echo "🛑 Keyboard interrupt (Ctrl+C) detected, exiting the global test queue."
        exit 130
    elif [ $STATUS -ne 0 ]; then
        echo "⚠️ Dataset $DATASET run may have exited abnormally. Resting for 10 seconds before the next one..."
        sleep 10
    else
        echo "✅ Grid search for dataset $DATASET completed successfully! Resting for 3 seconds before the next one..."
        sleep 3
    fi
    
    COUNT=$((COUNT + 1))
done

echo ""
echo "🎉 Congratulations! All dataset grid search tasks have finished running!"
