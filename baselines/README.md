# 🚀 scMMA Baselines Guide

This directory contains the environment setup and execution guide for all baseline methods compared in the **scMMA** project.

To completely avoid severe version conflicts between underlying libraries (such as `scvi-tools`, `anndata`, `scanpy`, `pytorch`, etc.) used by different multi-modal single-cell integration models, we recommend running the baseline methods using **two isolated Conda virtual environments**:

1. **`scmulti`**: The main integration environment, suitable for most baseline methods (such as **MOFA2**, **scMoMaT**, **GLUE**, **totalVI**, etc.).
2. **`env_multigrate`**: A dedicated isolated environment, specifically customized for **Multigrate**, with all core dependency versions locked to resolve API conflicts.

---

## 📂 Requirements

We have exported strictly dry-run tested and fully compatible dependency lists in the `scMultiBench` directory:

- 📝 **`scmulti` requirements list**: [requirements_scmulti.txt](file:///d:/code/research-projects/scMMA/baselines/scMultiBench/requirements_scmulti.txt)
- 📝 **`env_multigrate` requirements list**: [requirements_multigrate.txt](file:///d:/code/research-projects/scMMA/baselines/scMultiBench/requirements_multigrate.txt)

---

## 🛠️ Conda Environment Setup Guide

Please follow the steps below in your terminal to create and configure these two virtual environments:

### 1. Create and Configure the Main Environment `scmulti`

```bash
# Create and activate Python 3.10 virtual environment
conda create -n scmulti python=3.10 -y
conda activate scmulti

# Install Bedtools dependency (required for methods like GLUE)
conda install -c bioconda bedtools -y

# Install all Python dependencies using requirements
pip install -r scMultiBench/requirements_scmulti.txt
```

### 2. Create and Configure the Dedicated Environment `env_multigrate`

Because the older version of `scvi-tools` that `Multigrate` depends on has extensive API incompatibilities with newer versions of `anndata/scanpy/mudata`, we have built a strictly locked dependency chain:

```bash
# Create and activate Python 3.10 virtual environment
conda create -n env_multigrate python=3.10 -y
conda activate env_multigrate

# Install fully locked dependencies using requirements
pip install -r scMultiBench/requirements_multigrate.txt
```

> 💡 **Important Version Locking Notes (env_multigrate)**:
> 
> - Locked `setuptools==70.3.0` (higher versions remove `pkg_resources`, causing pytorch-lightning imports to fail).
> - Locked `anndata==0.9.2` and `mudata==0.2.3` (preventing new `SparseDataset` deprecation API from breaking multi-modal mudata loading).
> - Locked `scvi-tools==0.20.3` and `multigrate==0.0.2`.
> - Locked `numpy==1.26.4` (avoiding NumPy 2.x binary incompatibility errors).
> - Locked `jax==0.4.25` and `flax==0.8.5` (higher versions of JAX remove the `Device` API called by older scvi-tools).

---

## 🏃 Automated Batch Execution

We provide a one-click automated batch script `run_scmultibench_baslines.sh`. This script can **automatically detect your Conda installation path and switch to the correct Conda virtual environment based on different methods**, while avoiding potential GPU architecture compatibility issues.

### How to Run

```bash
# Enter the scMultiBench directory
cd scMultiBench

# Grant execution permissions and run the batch script
chmod +x run_scmultibench_baslines.sh
./run_scmultibench_baslines.sh
```

### Core Mechanism of the Automated Script

The script internally maintains an association mapping between methods and environments (`METHOD_ENVS`):

- **`MOFA2`** ➡️ `scmulti`
- **`scMoMaT`** ➡️ `scmulti`
- **`GLUE`** ➡️ `scmulti`
- **`Multigrate`** ➡️ `env_multigrate`

---

## 📝 Manual Execution Guide for Independent Methods

If you need to manually train and evaluate a specific method, you can activate the corresponding environment and schedule it using `run_baseline.py`.

### 1. MOFA2 (R & Python)

* 💡 **Required Environment**: `scmulti`

* The R package `MOFAPY2` and Python's `mofapy2` are pre-installed and bound in the main environment.
  
  ```bash
  conda activate scmulti
  cd scMultiBench
  python run_baseline.py --method MOFA2 --dataset D18
  ```

### 2. scMoMaT (Python)

* 💡 **Required Environment**: `scmulti`
  
  ```bash
  conda activate scmulti
  cd scMultiBench
  python run_baseline.py --method scMoMaT --dataset D18
  ```

### 3. GLUE (Graph-Linked Unified Embedding)

* 💡 **Required Environment**: `scmulti`

* ⚠️ **Data Dependency**: GLUE training requires the human reference genome GTF annotation file. We have integrated an **automatic download logic** inside `GLUE_human.py`. If the GTF is not detected locally, the script will automatically download the compressed file `gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz` (approx. 50MB) from the official EBI GENCODE FTP mirror.
  
  ```bash
  conda activate scmulti
  cd scMultiBench
  python run_baseline.py --method GLUE --dataset D18
  ```

### 4. Multigrate (Python Modality Integration)

* 💡 **Required Environment**: `env_multigrate`

* ⚠️ **GPU Compatibility Warning (CPU Mode Recommended)**:
  If you are using a very new architecture GPU (e.g., NVIDIA RTX 5090 D / sm_120 compute capability), the pre-installed PyTorch 2.5.1 CUDA version might not support this architecture and cause runtime crashes.
  
  - **Recommendation**: Force stable execution in **CPU mode** by prepending `CUDA_VISIBLE_DEVICES=""` to the run command.
  
  - **Upgrade PyTorch (Optional)**: If you wish to use the GPU during formal execution, you need to manually update PyTorch in the `env_multigrate` environment to a version that supports your GPU architecture (e.g., upgrade to torch for CUDA 12.4+ via official sources).
    
    ```bash
    conda activate env_multigrate
    cd scMultiBench
    # Force stable execution in CPU mode
    CUDA_VISIBLE_DEVICES="" python run_baseline.py --method Multigrate --dataset D18 --conda-env env_multigrate
    ```

---

## 📊 Experiment Outputs and Logging

- **Model Embeddings Results**: Saved by default in the path `baselines/experiment_output/<Method>/<Dataset>/embedding.h5`.
- **Performance Evaluation Result Log**: After every successful run, the system automatically evaluates using `eval_embedding.py` and records core integration metrics (including **NMI**, **ARI**, **ASW**, **cLISI**, **Overall** scores, as well as runtime and environment used) by appending them to:
  📝 **`baselines/baseline_experiment_log.csv`**
