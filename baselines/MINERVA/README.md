# MINERVA: **M**ultimodal **IN**tegration with self-sup**ERV**ised le**A**rning  
*A Generalizable Framework for Single-Cell Multiomics Analysis*<br>
<br>
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
![GitHub last commit](https://img.shields.io/github/last-commit/labomics/MINERVA)

---

## 📖 Introduction  
**MINERVA** is a versatile framework for single-cell multimodal data integration, specifically optimized for CITE-seq data. Our framework employs **six innovatively designed self-supervised learning (SSL) strategies**-categorized into bilevel masking, batch augmentation, and cell fusion—to achieve robust integrative analysis and cross-dataset generalization.

### Key Capabilities  
✅ **_De novo_ integration** of heterogeneous multi-omics datasets, especially for small-scale datasets  
✅ **Dimensionality reduction** for streamlined analysis  
✅ **Imputation** of missing features within- and cross-modality  
✅ **Batch correction**  
✅ **Zero-shot knowledge transfer** to unseen datasets without additional training or fine-tuning  
✅ **Instant cell-type identification**  

---

## 🗂️ Benchmark Datasets  
| Dataset<br>(Abbrev.)       | Species | Cells   | Proteins       | Batches | Accession ID |                Sample ratio: cell                 |
|       :--------------:     |  :---:  |  :---:  |    :-----:     |  :---:  |    :----:    |            :---------------------------           |
| CD45<sup>-</sup> dura mater<br>(DM)   | Mouse   | 6,697   | 168            | 1       | GSE191075    | 10%: 664<br>20%: 1,336<br>50%: 3,346<br>100%: 6,697 |
| Spleen & lymph nodes<br>(SLN)    | Mouse   | 29,338  | SLN111:111<br>SLN208:208 | 4       | GSE150599    | 10%: 2,339<br>20%: 4,678<br>50%: 11,731<br>100%: 23,470|
| Bone marrow mononuclear cell<br>(BMMC)| Human   | 90,261  | 134            | 12      | GSE194122    | 10%: 5,893<br>20%: 17,840<br>50%: 29,975<br>100%: 60,155|
| Immune cells  across lineages and tissues<br>(IMC)| Human   | 190,877 | 268            | 15      | GSE229791    | -                                                 |

---

## ⚙️ Installation  
### System Requirements  
- OS: Linux Ubuntu 18.04  
- Python 3.8.8 | R 4.1.0
- NVIDIA GPU

### Quick Setup  
```bash
# Create conda environment
conda create --name MINERVA python=3.8.8
conda activate MINERVA

# Install core packages
pip install torch==2.0.0
conda install -c conda-forge r-seurat=4.3.0

# Clone repository
git clone https://github.com/labomics/MINERVA.git
cd MINERVA
```

*Full dependency list:* [`others/Dependencies.txt`](others/Dependencies.txt)

---

## 🚀 Quick Start  
### 1. Data Preparation  
Perform quality control on each dataset and export the filtered data in **h5seurat** format for RNA and ADT modalities. Select variable features, generate the corresponding expression matrices, and split them by cell to create MINERVA inputs.

For demo data processing from `Example_data/`:
```bash
# Quality control
Rscript Preparation/1_rna_adt_filter.R dm_sub10_demo.rds dm_sub10
Rscript Preparation/1_rna_adt_filter.R sln_sub10_demo.rds sln_sub10

# Feature selection
Rscript Preparation/2_combine_subsets.R dm_sub10_demo.rds dm_sub10
Rscript Preparation/2_combine_subsets.R sln_sub10_demo.rds sln_sub10

# Generate MINERVA inputs
python Preparation/3_split_exp.py --task dm_sub10
python Preparation/3_split_exp.py --task sln_sub10
```

*Supports Seurat/Scanpy preprocessed data in h5seurat format. Once preprocessing is complete, split the matrices with **3_split_exp.py**.*

### 2. MINERVA Application  
#### Scenario A: _De Novo_ Integration  
*Corresponding to the integration of **Results 2-4** in the manuscript.*<br>
<br>
Execute the following commands to perform integration using SSL strategies:
```bash
# Integration with SSL strategies
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task dm_sub10 --pretext mask
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task sln_sub10 --pretext mask noise downsample
# Note: Cell fusion strategies require at least 2 batches
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task bmmc_sub10 --pretext mask noise downsample fusion
```

##### Output Extraction  
Trained model states are saved at specified epochs. To obtain the joint low-dimensional representations, intra- and inter-modality imputed expression profiles, and the batch-corrected matrix, run:
```bash
python MINERVA/run.py --task dm_sub10 --init_model sp_00000999 --actions predict_all
```

#### Scenario B: Zero-Shot Generalization to Novel Queries  
Two cases are provided:<br>
##### Case 1: Trained on two batches of SLN datasets, and tested the transfer performance on the remaining batches  
*This case corresponds to the generalization results in **Result 3**.*
```bash
# Split train/test datasets
mkdir -p ./result/preprocess/sln_sub10_train/{train,test}/

for dir in train test; do
    ln -sf ../../sln_sub10/feat ./result/preprocess/sln_sub10_train/$dir/
done

for i in 2 3; do
    ln -sf ../../sln_sub10/subset_$i ./result/preprocess/sln_sub10_train/train/subset_$((i-2))
done

ln -sf ../../sln_sub10/subset_{0,1} ./result/preprocess/sln_sub10_train/test/

# Train model
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task sln_sub10_train --pretext mask noise downsample --use_shm 2

# Transfer to unseen batches
python MINERVA/run.py --task sln_sub10_transfer --ref sln_sub10_train --rf_experiment e0 \
--experiment transfer --init_model sp_latest --init_from_ref 1 --action predict_all  --use_shm 3
```

##### Case 2: Construct reference atlas and transfer to novel cross-tissue datasets  
*This case corresponds to **Result 5**.*
```bash
# Reference atlas construction
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task imc_ref --pretext mask noise downsample fusion --use_shm 2

# Knowledge transfer to cross-tissues queries
python MINERVA/run.py --task imc_query --ref imc_ref --rf_experiment e0 \
--experiment transfer --init_model sp_latest --init_from_ref 1 --action predict_all --use_shm 3
```

### 3. Performance Evaluation
The output from both scenarios includes:
- **Input reconstructions**
- **Batch-corrected expression profiles**
- **Imputed matrices**
- **Cross-modality expression translations**
- **34-dimensional joint embeddings**
  - First 32 dimensions: Biological state
  - Last 2 dimensions: Technical bias

These embeddings can be imported using Python ("pd.read_csv") or R ("read.csv") to compute neighborhood graphs and perform clustering with Anndata or Seurat.<br>

*Example output paths*:  `dm_sub10/e0/default/predict/sp_latest/subset_0/{z,x_impu,x_bc,x_trans}`


**Quantitative evaluation scripts**:  
```bash
# Batch correction & biological conservation
python Evaluation/benchmark_batch_bio.py

# Modality alignment assessment
python Evaluation/benchmark_mod.py

# Comprehensive metric aggregation
python Evaluation/combine_metrics.py
```
---

## ⚡ Advanced Configuration  
### Key Parameters  
| Argument       | Description                          | Options                         |
|----------------|--------------------------------------|---------------------------------|
| `--pretext`    | SSL strategies                       | `mask`, `noise`, `downsample`, `fusion` |
| `--use_shm`    | Datasets partition mode              | `1` (all), `2` (train), `3` (test)      |
| `--actions`    | Post-training operations             | `predict_all`, `predict_joint`, etc.    |

*Full options:*  
```bash
python MINERVA/run.py -h
```

---

## 📜 License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
