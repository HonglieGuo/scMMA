#!/bin/bash
# =========================================================================
# 🚀 scMultiBench Baseline Execution Script
# Method: FactVAE
# 
# Supported Modalities: RNA+ATAC
# Supported Tasks: Vertical Integration (Paired Multi-omics)
# =========================================================================

# The datasets to evaluate on (can be passed as arguments, otherwise defaults)
DATASETS=${@:-"D15 D18 D56"}

# Path variables
PROJECT_ROOT=$(cd ../.. && pwd)
SCMULTIBENCH_DIR="$PROJECT_ROOT/baselines/scMultiBench"
OUTPUT_DIR="$PROJECT_ROOT/baselines/experiment_output"

echo "============================================================"
echo " Starting FactVAE Evaluation Pipeline "
echo "============================================================"

for DATASET in $DATASETS
do
    echo "------------------------------------------------------------"
    echo " Running FactVAE on $DATASET..."
    echo "------------------------------------------------------------"
    
    SAVE_DIR="$OUTPUT_DIR/FactVAE/$DATASET"
    mkdir -p "$SAVE_DIR"
    
    # Resolve ATAC files (can be atac*.h5 or peak*.h5)
    ATAC_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/atac*.h5)
    if [ ! -e "${ATAC_FILES[0]}" ]; then
        ATAC_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/peak*.h5)
    fi

    python main_FactVAE.py \
        --path1 "$PROJECT_ROOT/datasets/h5/$DATASET"/rna*.h5 \
        --path2 "${ATAC_FILES[@]}" \
        --save_path "$SAVE_DIR"
        
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "❌ FactVAE failed on $DATASET. Skipping evaluation."
        continue
    fi
    
    # Resolve CTY files
    CTY_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/cty*.csv)

    # 2. Evaluate Embedding
    echo "Running Evaluation..."
    python "$SCMULTIBENCH_DIR/eval_embedding.py" \
        --embedding_path "$SAVE_DIR/embedding.h5" \
        --cty_path "${CTY_FILES[@]}" \
        --save_path "$SAVE_DIR" \
        --split_for_foscttm
        
    # 3. Log to CSV
    # We will use a shared logger or call a simple python snippet to append the JSON result to the CSV
    python ../logger.py \
        --method "FactVAE" \
        --dataset "$DATASET" \
        --modality "RNA+ATAC" \
        --task "Vertical" \
        --eval_dir "$SAVE_DIR"
        
    echo "✅ FactVAE completed on $DATASET!"
done
