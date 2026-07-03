#!/bin/bash
# =========================================================================
# 🚀 scMultiBench New Baselines Master Execution Script
# =========================================================================

# Switch to the directory where the script is located
cd "$(dirname "$0")" || exit 1

# Configure datasets to run for each method (comment out to skip)

# =========================================================================================
# 🌟 0. GLUE (Standalone version with real alignment metrics)
#    Supported modalities: RNA + ATAC
# =========================================================================================
GLUE_DATASETS=(
    # "D15"
    # "D18"
    # "D56"
)

# =========================================================================================
# 🌟 1. FactVAE
#    Supported modalities: RNA + ATAC (Strongly coupled dual-network architecture)
# =========================================================================================
FactVAE_DATASETS=(
    # "D15"
    # "D18"
    # "D56"
)

# =========================================================================================
# 🌟 2. MINERVA
#    Supported modalities: RNA + ADT (CITE-seq specialized, hardcoded Poisson distribution & preprocessing)
# =========================================================================================
MINERVA_DATASETS=(
    # "D1"
    # "D53"
    # "D54"
)

# =========================================================================================
# 🌟 3. scECDA
#    Supported modalities: RNA, RNA+ATAC, RNA+ADT, RNA+ATAC+ADT (Extremely flexible)
# =========================================================================================
scECDA_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D22"
    # "D53"
    # "D54"
    # "D56"
    # "D59"
)

# =========================================================================================
# 🌟 4. scMDCF
#    Supported modalities: RNA+ADT (CITE-seq) or RNA+ATAC (SNARE-seq)
# =========================================================================================
scMDCF_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

# =========================================================================================
# 🌟 5. scMRDR
#    Supported modalities: RNA+ATAC, RNA+ADT (Good at unpaired data integration, Diagonal/Cross-omics)
# =========================================================================================
scMRDR_DATASETS=(
    # "D1"
    # "D15"
    # "D18"
    # "D53"
    # "D54"
    # "D56"
)

echo "====================================================================="
echo "       New Baselines Automated Test Queue (Dataset-level control)    "
echo "====================================================================="

if [ ${#GLUE_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "====================================================================="
    echo ">>> 🌟 Preparing to run method: GLUE (Standalone)"
    echo ">>> 📊 Datasets: ${GLUE_DATASETS[*]}"
    echo "====================================================================="
    cd GLUE || exit 1
    bash run_baseline.sh "${GLUE_DATASETS[@]}"
    cd ..
fi


ALL_METHODS=("FactVAE" "MINERVA" "scECDA" "scMDCF" "scMRDR")

for METHOD in "${ALL_METHODS[@]}"
do
    # Dynamically get the corresponding DATASETS array
    DATASETS_VAR="${METHOD}_DATASETS[@]"
    DATASETS=("${!DATASETS_VAR}")
    
    if [ ${#DATASETS[@]} -gt 0 ]; then
        echo ""
        echo "====================================================================="
        echo ">>> 🌟 Preparing to run method: $METHOD"
        echo ">>> 📊 Datasets: ${DATASETS[*]}"
        echo "====================================================================="
        
        cd "$METHOD" || continue
        bash run_baseline.sh "${DATASETS[@]}"
        cd ..
    fi
done

echo ""
echo "🎉 Congratulations! All newly added methods have finished running! Results are recorded in baseline_experiment_log.csv"
