#!/bin/bash
# =========================================================================
# 🚀 scMultiBench Baseline Execution Script
# Method: GLUE
# 
# Supported Modalities: RNA+ATAC
# Supported Tasks: Vertical Integration (Paired Multi-omics)
# =========================================================================

# The datasets to evaluate on (can be passed as arguments, otherwise defaults)
DATASETS=${@:-"D15"}

# Path variables
PROJECT_ROOT=$(cd ../.. && pwd)
SCMULTIBENCH_DIR="$PROJECT_ROOT/baselines/scMultiBench"
OUTPUT_DIR="$PROJECT_ROOT/baselines/experiment_output"

echo "============================================================"
echo " Starting GLUE Evaluation Pipeline "
echo "============================================================"

for DATASET in $DATASETS
do
    echo "------------------------------------------------------------"
    echo " Running GLUE on $DATASET..."
    echo "------------------------------------------------------------"
    
    SAVE_DIR="$OUTPUT_DIR/GLUE/$DATASET"
    mkdir -p "$SAVE_DIR"
    
    # Resolve ATAC files (can be atac*.h5 or peak*.h5)
    ATAC_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/atac*.h5)
    if [ ! -e "${ATAC_FILES[0]}" ]; then
        ATAC_FILES=("$PROJECT_ROOT/datasets/h5/$DATASET"/peak*.h5)
    fi

    python main_GLUE.py \
        --path1 "$PROJECT_ROOT/datasets/h5/$DATASET"/rna*.h5 \
        --path2 "${ATAC_FILES[@]}" \
        --save_path "$SAVE_DIR"
        
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "❌ GLUE failed on $DATASET (Status $STATUS). Logging as OOM/Failed."
        echo '{"time": "OOM", "Peak_Mem_GB": "OOM", "Params_M": "OOM", "Embedding_Dim": "OOM"}' > "$SAVE_DIR/metrics.json"
        python ../logger.py \
            --method "GLUE" \
            --dataset "$DATASET" \
            --modality "RNA+ATAC" \
            --task "Vertical" \
            --eval_dir "$SAVE_DIR"
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
    python ../logger.py \
        --method "GLUE" \
        --dataset "$DATASET" \
        --modality "RNA+ATAC" \
        --task "Vertical" \
        --eval_dir "$SAVE_DIR"
        
    echo "✅ GLUE completed on $DATASET!"
done
