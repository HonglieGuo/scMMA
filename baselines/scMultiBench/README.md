# scMultiBench — Baseline Method Script Collection

> Source: [PYangLab/scMultiBench](https://github.com/PYangLab/scMultiBench)
> Purpose: Provides execution scripts and a unified evaluation pipeline for baseline methods in the scMMA project.

---

## Directory Structure

```
scMultiBench/
├── tools_scripts/          # Execution scripts for various baseline methods
│   ├── totalVI/            # Python, RNA+ADT
│   ├── MOFA2/              # R+Python, Multi-modal
│   ├── Seurat_v4/          # R (WNN), Multi-modal
│   ├── ... (Total of 40 methods)
│   └── uniPort/
└── evaluation_pipelines/   # Unified evaluation scripts
    ├── scib_metrics/       # DR + Batch Correction + Clustering metrics
    ├── classification/     # Classification evaluation
    ├── clustering/         # Clustering evaluation
    ├── imputation/         # Imputation evaluation
    ├── fs/                 # Feature Selection evaluation
    └── spatial_registration/  # Spatial registration evaluation
```

---

## Method List (Categorized by Integration Type)

### Vertical Integration — Main Baseline Methods

| Method | Language | Modalities | Supported Tasks |
|------|------|---------|---------|
| totalVI | Python | RNA+ADT | DR, Clustering, Imputation |
| sciPENN | Python | RNA+ADT | DR, Imputation |
| Concerto | Python | Multi-modal | DR, Clustering |
| scMSI | Python | RNA+ADT | DR, Clustering |
| Matilda | Python | Multi-modal | DR, Classification, FS |
| MOFA2 | R+Python | Multi-modal | DR, Clustering, FS |
| Multigrate | Python | Multi-modal | DR, Clustering |
| UINMF | R | Multi-modal | DR, Clustering |
| scMoMaT | Python | Multi-modal | DR, Clustering, FS |
| Seurat_v4 (WNN) | R | Multi-modal | DR, Clustering |
| scMM | Python | Multi-modal | DR, Imputation |
| scMDC | Python | Multi-modal | DR, Clustering |
| moETM | Python | Multi-modal | DR, Imputation |
| VIMCCA | Python | Multi-modal | DR, Clustering |
| iPOLNG | Python | Multi-modal | DR, Clustering |
| MIRA | Python | RNA+ATAC | DR, Clustering |
| UnitedNet | Python | Multi-modal | DR, Imputation |
| scMVP | Python | RNA+ATAC | DR, Clustering |

### Diagonal Integration

| Method | Language | Notes |
|------|------|------|
| scBridge | Python | Cross-modal label transfer |
| Portal | Python | Fast batch correction |
| SCALEX | Python | Scalable batch correction |
| VIPCCA | Python | — |
| Seurat_v3 | R | CCA-based |
| Seurat_v5 | R | Bridge integration |
| MultiMAP | Python | — |
| sciCAN | Python | — |
| Conos | R | — |
| iNMF | R | LIGER |
| online iNMF | R | LIGER Online Version |
| scJoint | Python | — |
| GLUE | Python | Graph-linked multi-modal alignment |
| uniPort | Python | Optimal transport |

### Mosaic Integration

| Method | Language |
|------|------|
| MultiVI | Python |
| Cobolt | Python |
| StabMap | R |
| SMILE | Python |

### Spatial Registration

| Method | Language |
|------|------|
| PASTE | Python |
| PASTE2 | Python |
| SPIRAL | Python |
| GPSA | Python |

---

## Evaluation Pipeline Usage

### scib_metrics (DR + Batch Correction + Clustering)

```bash
cd evaluation_pipelines/scib_metrics
python scib_metric.py \
    --data_path "path/to/embedding.h5" \
    --cty_path "path/to/cty1.csv" "path/to/cty2.csv" \
    --save_path "path/to/results/"
```

Output metrics: NMI, ARI, ASW_label, ASW_batch, iLISI, cLISI, kBET, graph_conn, isolated_label_F1

### Classification

```bash
cd evaluation_pipelines/classification
python classification.py \
    --reference "path/to/ref_embedding.h5" \
    --query "path/to/query_embedding.h5" \
    --reference_cty "path/to/cty1.csv" \
    --query_cty "path/to/cty2.csv" \
    --save_path "path/to/results/"
```

Output: Overall Accuracy, Average Accuracy, F1, Sensitivity, Specificity
