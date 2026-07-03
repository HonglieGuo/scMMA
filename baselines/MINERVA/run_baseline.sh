#!/bin/bash
# =========================================================================
# 🚀 scMultiBench Baseline Execution Script
# Method: MINERVA
# 
# Supported Modalities: RNA+ADT (CITE-seq)
# Supported Tasks: Vertical Integration, Cross-Batch, Zero-shot Transfer
# =========================================================================

# The datasets to evaluate on
DATASETS=${@:-"D1 D53 D54"}

PROJECT_ROOT=$(cd ../.. && pwd)
SCMULTIBENCH_DIR="$PROJECT_ROOT/baselines/scMultiBench"
OUTPUT_DIR="$PROJECT_ROOT/baselines/experiment_output"

echo "============================================================"
echo " Starting MINERVA Evaluation Pipeline "
echo "============================================================"

for DATASET in $DATASETS
do
    echo "------------------------------------------------------------"
    echo " Running MINERVA on $DATASET..."
    echo "------------------------------------------------------------"
    
    SAVE_DIR="$OUTPUT_DIR/MINERVA/$DATASET"
    mkdir -p "$SAVE_DIR"
    
    python main_MINERVA.py \
        --path1 "$PROJECT_ROOT/datasets/h5/$DATASET"/rna*.h5 \
        --path2 "$PROJECT_ROOT/datasets/h5/$DATASET"/adt*.h5 \
        --save_path "$SAVE_DIR"
        
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "❌ MINERVA failed on $DATASET. Skipping evaluation."
        continue
    fi
    
    # Resolve CTY files
    CTY_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/cty*.csv)

    echo "Running Evaluation..."
    python "$SCMULTIBENCH_DIR/eval_embedding.py" \
        --embedding_path "$SAVE_DIR/embedding.h5" \
        --cty_path "${CTY_FILES[@]}" \
        --save_path "$SAVE_DIR"
        
    python ../logger.py \
        --method "MINERVA" \
        --dataset "$DATASET" \
        --modality "RNA+ADT" \
        --task "Vertical/Cross-Batch" \
        --eval_dir "$SAVE_DIR"
        
    echo "✅ MINERVA completed on $DATASET!"
done
