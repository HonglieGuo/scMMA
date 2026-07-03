#!/bin/bash

# =================================================================================
# 🚀 Batch script to automatically run scMultiBench baselines on main datasets
# =================================================================================

# Switch to the directory where the script is located (baselines/scMultiBench) to ensure run_baseline.py is found
cd "$(dirname "$0")" || exit 1

# =========================================================================================
# 1. Configure datasets to run for each method (comment out to skip)
# =========================================================================================

# =========================================================================================
# 🌟 MOFA2
#    Supported modalities: RNA+ADT, RNA+ATAC (Bimodal)
# =========================================================================================
MOFA2_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 Multigrate
#    Supported modalities: RNA+ADT, RNA+ATAC (Bimodal)
# =========================================================================================
Multigrate_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 scMoMaT
#    Supported modalities: RNA+ADT, RNA+ATAC (Bimodal)
# =========================================================================================
scMoMaT_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 uniPort
#    Supported modalities: RNA+ADT, RNA+ATAC (Bimodal)
# =========================================================================================
uniPort_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 SMILE
#    Supported modalities: RNA+ADT, RNA+ATAC (Bimodal)
# =========================================================================================
SMILE_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 MultiMAP
#    Supported modalities: RNA+ADT, RNA+ATAC (Bimodal)
# =========================================================================================
MultiMAP_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 Seurat_v4
#    Supported modalities: RNA+ADT, RNA+ATAC
# =========================================================================================
Seurat_v4_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 MultiVI
#    Supported modalities: RNA+ATAC
# =========================================================================================
MultiVI_DATASETS=(
    # "D15"
    # "D18"
    # "D56"
)



# =========================================================================================
# 🌟 UINMF
#    Supported modalities: RNA+ADT, RNA+ATAC
# =========================================================================================
UINMF_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# Environment configuration
# =========================================================================================
declare -A METHOD_ENVS
METHOD_ENVS=(
    ["MOFA2"]="scmulti"
    ["scMoMaT"]="scmulti"
    ["Multigrate"]="env_multigrate"
    ["uniPort"]="scmulti"
    ["SMILE"]="scmulti"
    ["MultiMAP"]="scmulti"
    ["Seurat_v4"]="scmulti"
    ["MultiVI"]="scmulti"
    ["UINMF"]="scmulti"
)

# Automatically find conda.sh script on the system to activate environments
CONDA_SH_PATHS=(
    "/home/ghl/anaconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "/opt/anaconda3/etc/profile.d/conda.sh"
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/usr/local/anaconda3/etc/profile.d/conda.sh"
    "/usr/local/miniconda3/etc/profile.d/conda.sh"
)



CONDA_SH=""
for path in "${CONDA_SH_PATHS[@]}"; do
    if [ -f "$path" ]; then
        CONDA_SH="$path"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    echo "⚠️ Could not find conda.sh in common paths. Trying to use 'conda' command directly."
else
    echo "ℹ️ Found Conda initialization script: $CONDA_SH"
fi

# ==========================================
# 2. Execution Logic
# ==========================================

METHODS=(
    "MOFA2"
    "Multigrate"
    "scMoMaT"
    "uniPort"
    "SMILE"
    "MultiMAP"
    "Seurat_v4"
    "MultiVI"
    "UINMF"
)

echo "====================================================================="
echo "          scMultiBench Automated Test Queue          "
echo "====================================================================="
sleep 1

for METHOD in "${METHODS[@]}"
do
    # Dynamically get the corresponding dataset array for the method
    eval "CURRENT_DATASETS=(\"\${${METHOD}_DATASETS[@]}\")"
    
    # If the current method has no configured datasets, skip it
    if [ ${#CURRENT_DATASETS[@]} -eq 0 ]; then
        continue
    fi
    
    echo ""
    echo "====================================================================="
    echo ">>> 🌟 Preparing to run method: $METHOD"
    echo ">>> 📊 Datasets: ${CURRENT_DATASETS[*]}"
    echo "====================================================================="
    
    M_COUNT=1
    TOTAL_CURRENT_DATASETS=${#CURRENT_DATASETS[@]}
    
    for DATASET in "${CURRENT_DATASETS[@]}"
    do
        # Get the corresponding Conda environment for the current method
        ENV=${METHOD_ENVS[$METHOD]}
        if [ -z "$ENV" ]; then
            ENV="scmulti" # Default fallback
        fi

        echo ""
        echo ">>> 📅 Scheduling [$M_COUNT/$TOTAL_CURRENT_DATASETS]: Method $METHOD -> Dataset $DATASET (Env: $ENV)"
        
        # Compatibility prefix: For Multigrate, force CPU mode to avoid GPU architecture / PyTorch compatibility issues
        PREFIX_ENV=""
        if [ "$METHOD" = "Multigrate" ]; then
            PREFIX_ENV="CUDA_VISIBLE_DEVICES=\"\""
        fi
        
        if [ ! -z "$CONDA_SH" ]; then
            # Activate environment using source before executing
            echo "Activating conda environment '$ENV'..."
            CMD="source $CONDA_SH && conda activate $ENV && $PREFIX_ENV python run_baseline.py --method $METHOD --dataset $DATASET --conda-env $ENV"
            bash -c "$CMD"
        else
            # If conda.sh is not found, try global conda run
            CMD="$PREFIX_ENV conda run --no-capture-output -n $ENV python run_baseline.py --method $METHOD --dataset $DATASET --conda-env $ENV"
            echo "Executing: $CMD"
            eval $CMD
        fi
        
        STATUS=$?
        if [ $STATUS -eq 130 ]; then
            echo "🛑 Keyboard interrupt (Ctrl+C) detected, exiting the global test queue."
            exit 130
        elif [ $STATUS -ne 0 ]; then
            echo "⚠️ Method $METHOD on dataset $DATASET run may have exited abnormally. Resting for 5 seconds before continuing..."
            sleep 5
        else
            echo "✅ Method $METHOD -> Dataset $DATASET run completed! Resting for 3 seconds before next..."
            sleep 3
        fi
        
        M_COUNT=$((M_COUNT + 1))
    done
done

echo ""
echo "🎉 All Baseline tasks completed! Results are recorded in ../baseline_experiment_log.csv"
