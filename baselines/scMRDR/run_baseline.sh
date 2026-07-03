#!/bin/bash
# =========================================================================
# 🚀 scMultiBench Baseline Execution Script
# Method: scMRDR
# 
# Supported Modalities: RNA+ATAC, RNA+ADT
# Supported Tasks: Unpaired Integration, Cross-omics translation, Diagonal
# =========================================================================

DATASETS=${@:-"D1 D15 D18 D53 D54 D56"}

PROJECT_ROOT=$(cd ../.. && pwd)
SCMULTIBENCH_DIR="$PROJECT_ROOT/baselines/scMultiBench"
OUTPUT_DIR="$PROJECT_ROOT/baselines/experiment_output"

echo "============================================================"
echo " Starting scMRDR Evaluation Pipeline "
echo "============================================================"

for DATASET in $DATASETS
do
    echo "------------------------------------------------------------"
    echo " Running scMRDR on $DATASET..."
    echo "------------------------------------------------------------"
    
    SAVE_DIR="$OUTPUT_DIR/scMRDR/$DATASET"
    mkdir -p "$SAVE_DIR"
    
    # Resolve MOD2 files (ADT or ATAC/Peak)
    MOD2_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/adt*.h5)
    if [ ! -e "${MOD2_FILES[0]}" ]; then
        MOD2_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/atac*.h5)
        if [ ! -e "${MOD2_FILES[0]}" ]; then
            MOD2_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/peak*.h5)
        fi
    fi
    
    python main_scMRDR.py \
        --path1 "$PROJECT_ROOT/datasets/h5/$DATASET"/rna*.h5 \
        --path2 "${MOD2_FILES[@]}" \
        --save_path "$SAVE_DIR"
        
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "❌ scMRDR failed on $DATASET. Skipping evaluation."
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
        --method "scMRDR" \
        --dataset "$DATASET" \
        --modality "Multi-Omics" \
        --task "Unpaired/Diagonal" \
        --eval_dir "$SAVE_DIR"
        
    echo "✅ scMRDR completed on $DATASET!"
done
