# scMultiBench — 对比方法脚本集

> 来源：[PYangLab/scMultiBench](https://github.com/PYangLab/scMultiBench)
> 用途：为 scMMA 项目提供对比方法的运行脚本和统一评估 pipeline

---

## 目录结构

```
scMultiBench/
├── tools_scripts/          # 各对比方法的运行脚本
│   ├── totalVI/            # Python, RNA+ADT
│   ├── MOFA2/              # R+Python, 多模态
│   ├── Seurat_v4/          # R (WNN), 多模态
│   ├── ... (共 40 个方法)
│   └── uniPort/
└── evaluation_pipelines/   # 统一评估脚本
    ├── scib_metrics/       # DR + Batch Correction + Clustering 指标
    ├── classification/     # 分类评估
    ├── clustering/         # 聚类评估
    ├── imputation/         # 插补评估
    ├── fs/                 # Feature Selection 评估
    └── spatial_registration/  # 空间注册评估
```

---

## 方法清单（按整合类型分类）

### Vertical Integration（垂直整合）— 主要对比方法

| 方法 | 语言 | 适用模态 | 支持任务 |
|------|------|---------|---------|
| totalVI | Python | RNA+ADT | DR, Clustering, Imputation |
| sciPENN | Python | RNA+ADT | DR, Imputation |
| Concerto | Python | 多模态 | DR, Clustering |
| scMSI | Python | RNA+ADT | DR, Clustering |
| Matilda | Python | 多模态 | DR, Classification, FS |
| MOFA2 | R+Python | 多模态 | DR, Clustering, FS |
| Multigrate | Python | 多模态 | DR, Clustering |
| UINMF | R | 多模态 | DR, Clustering |
| scMoMaT | Python | 多模态 | DR, Clustering, FS |
| Seurat_v4 (WNN) | R | 多模态 | DR, Clustering |
| scMM | Python | 多模态 | DR, Imputation |
| scMDC | Python | 多模态 | DR, Clustering |
| moETM | Python | 多模态 | DR, Imputation |
| VIMCCA | Python | 多模态 | DR, Clustering |
| iPOLNG | Python | 多模态 | DR, Clustering |
| MIRA | Python | RNA+ATAC | DR, Clustering |
| UnitedNet | Python | 多模态 | DR, Imputation |
| scMVP | Python | RNA+ATAC | DR, Clustering |

### Diagonal Integration（对角整合）

| 方法 | 语言 | 备注 |
|------|------|------|
| scBridge | Python | 跨模态标签迁移 |
| Portal | Python | 快速批次校正 |
| SCALEX | Python | 可扩展批次校正 |
| VIPCCA | Python | — |
| Seurat_v3 | R | CCA-based |
| Seurat_v5 | R | Bridge 整合 |
| MultiMAP | Python | — |
| sciCAN | Python | — |
| Conos | R | — |
| iNMF | R | LIGER |
| online iNMF | R | LIGER 在线版 |
| scJoint | Python | — |
| GLUE | Python | 图引导多模态对齐 |
| uniPort | Python | 最优传输 |

### Mosaic Integration（马赛克整合）

| 方法 | 语言 |
|------|------|
| MultiVI | Python |
| Cobolt | Python |
| StabMap | R |
| SMILE | Python |

### Spatial Registration（空间注册）

| 方法 | 语言 |
|------|------|
| PASTE | Python |
| PASTE2 | Python |
| SPIRAL | Python |
| GPSA | Python |

---

## 评估 Pipeline 使用方法

### scib_metrics（DR + Batch Correction + Clustering）

```bash
cd evaluation_pipelines/scib_metrics
python scib_metric.py \
    --data_path "path/to/embedding.h5" \
    --cty_path "path/to/cty1.csv" "path/to/cty2.csv" \
    --save_path "path/to/results/"
```

输出指标：NMI, ARI, ASW_label, ASW_batch, iLISI, cLISI, kBET, graph_conn, isolated_label_F1

### Classification（分类）

```bash
cd evaluation_pipelines/classification
python classification.py \
    --reference "path/to/ref_embedding.h5" \
    --query "path/to/query_embedding.h5" \
    --reference_cty "path/to/cty1.csv" \
    --query_cty "path/to/cty2.csv" \
    --save_path "path/to/results/"
```

输出：Overall Accuracy, Average Accuracy, F1, Sensitivity, Specificity
