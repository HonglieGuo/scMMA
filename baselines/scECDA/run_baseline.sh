#!/bin/bash
# =========================================================================
# 🚀 scMultiBench Baseline Execution Script
# Method: scECDA
# 
# Supported Modalities: RNA+ADT+ATAC
# Supported Tasks: Vertical Integration, Multi-omics Alignment
# =========================================================================

DATASETS=${@:-"D1 D15 D18 D22 D53 D54 D56 D59"}

PROJECT_ROOT=$(cd ../.. && pwd)
SCMULTIBENCH_DIR="$PROJECT_ROOT/baselines/scMultiBench"
OUTPUT_DIR="$PROJECT_ROOT/baselines/experiment_output"

echo "============================================================"
echo " Starting scECDA Evaluation Pipeline "
echo "============================================================"

for DATASET in $DATASETS
do
    echo "------------------------------------------------------------"
    echo " Running scECDA on $DATASET..."
    echo "------------------------------------------------------------"
    
    SAVE_DIR="$OUTPUT_DIR/scECDA/$DATASET"
    mkdir -p "$SAVE_DIR"
    
    # Resolve ADT files
    ADT_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/adt*.h5)
    # Resolve ATAC files (can be atac*.h5 or peak*.h5)
    ATAC_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/atac*.h5)
    if [ ! -e "${ATAC_FILES[0]}" ]; then
        ATAC_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/peak*.h5)
    fi

    # Build scECDA call (optionally omit missing modalities)
    CMD="python main_scECDA.py --path1 \"$PROJECT_ROOT/datasets/h5/$DATASET\"/rna*.h5 --save_path \"$SAVE_DIR\""
    if [ -e "${ADT_FILES[0]}" ]; then
        CMD="$CMD --path2 ${ADT_FILES[@]}"
    fi
    if [ -e "${ATAC_FILES[0]}" ]; then
        CMD="$CMD --path3 ${ATAC_FILES[@]}"
    fi
    
    eval $CMD
        
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "❌ scECDA failed on $DATASET. Skipping evaluation."
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
        --method "scECDA" \
        --dataset "$DATASET" \
        --modality "RNA+ADT+ATAC" \
        --task "Vertical" \
        --eval_dir "$SAVE_DIR"
        
    echo "✅ scECDA completed on $DATASET!"
done
