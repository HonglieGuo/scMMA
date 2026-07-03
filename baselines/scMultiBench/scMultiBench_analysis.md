# scMultiBench 仓库分析 — 对比方法 Pipeline 调研

> 仓库地址：https://github.com/PYangLab/scMultiBench

## 结论：✅ 非常适合用来跑对比实验

scMultiBench 仓库提供了 **40 个整合方法的完整运行脚本** 和 **统一的评估 pipeline**，可以大幅加速你的对比实验。

---

## 一、仓库结构概览

| 目录 | 内容 | 用途 |
|------|------|------|
| `tools_scripts/` | 每个对比方法一个子文件夹，包含运行脚本 | **跑各方法获得 embedding** |
| `evaluation_pipelines/` | 统一评估脚本 | **计算评估指标** |
| `data/` | 示例数据和数据预处理脚本 | 数据准备 |
| `figure_script/` | 论文图表绘制脚本 | 可视化（可选） |

---

## 二、`tools_scripts/` 中已有的方法（共 38 个子文件夹）

### 🟢 与 scMMA 直接相关的方法（Vertical Integration — 你的主要对比对象）

| 方法 | 发表年份 | 语言 | 模态 | 主要任务 | 备注 |
|------|----------|------|------|----------|------|
| **totalVI** | 2021 | Python | RNA+ADT | DR / Clustering / Imputation | scvi-tools 生态，最常用的对比方法之一 |
| **MOFA2 (MOFA+)** | 2020 | R+Python | 多模态 | DR / Clustering / Feature Selection | 经典因子分析方法 |
| **Multigrate** | 2022 | Python | 多模态 | DR / Clustering | scvi-tools 生态 |
| **scMoMaT** | 2023 | Python | 多模态 | DR / Clustering / Feature Selection | 支持 mosaic 整合 |
| **Seurat_v4 (WNN)** | 2021 | R | RNA+ADT / RNA+ATAC | DR / Clustering | 业界标准基准线 |
| **MIRA** | 2022 | Python | RNA+ATAC | DR / Clustering | ATAC 专用 |
| **VIMCCA** | 2023 | Python | 多模态 | DR / Clustering | — |
| **iPOLNG** | 2023 | Python | 多模态 | DR / Clustering | — |
| **scMSI** | 2023 | Python | RNA+ADT | DR / Clustering | — |
| **Matilda** | 2023 | Python | 多模态 | DR / Classification / FS | 同 PYangLab 出品 |
| **sciPENN** | 2022 | Python | RNA+ADT | DR / Imputation | — |
| **Concerto** | 2022 | Python | 多模态 | DR / Clustering | — |
| **UnitedNet** | 2023 | Python | 多模态 | DR / Imputation | — |
| **scMM** | 2022 | Python | 多模态 | DR / Imputation | 多模态 VAE |
| **scMDC** | 2022 | Python | 多模态 | DR / Clustering | — |
| **moETM** | 2023 | Python | 多模态 | DR / Imputation | — |
| **scMVP** | 2022 | Python | RNA+ATAC | DR / Clustering | — |
| **Cobolt** | 2022 | Python | RNA+ATAC | DR / Clustering | Mosaic 整合 |

### 🔵 Diagonal Integration 方法

| 方法 | 发表年份 | 语言 | 备注 |
|------|----------|------|------|
| **scBridge** | 2024 | Python | 跨模态标签迁移 |
| **Portal** | 2022 | Python | 快速批次校正 |
| **SCALEX** | 2022 | Python | 可扩展批次校正 |
| **VIPCCA** | 2021 | Python | — |
| **Seurat_v3** | 2019 | R | CCA-based 整合 |
| **Seurat_v5** | 2023 | R | Bridge 整合 |
| **MultiMAP** | 2021 | Python | — |
| **sciCAN** | 2022 | Python | — |
| **Conos** | 2019 | R | — |
| **iNMF** | 2019 | R | LIGER |
| **online iNMF** | 2021 | R | LIGER 在线版本 |
| **scJoint** | 2021 | Python | — |
| **GLUE** | 2022 | Python | 图引导多模态对齐 |
| **uniPort** | 2022 | Python | 最优传输 |

### 🟣 其他（Mosaic / Cross / Spatial）

| 方法 | 发表年份 | 类型 |
|------|----------|------|
| **MultiVI** | 2023 | Mosaic |
| **StabMap** | 2022 | Mosaic / Cross |
| **SMILE** | 2022 | Mosaic |
| **UINMF** | 2022 | Mosaic / Cross |
| **PASTE / PASTE2** | 2022/2023 | Spatial Registration |
| **SPIRAL** | 2023/2024 | Spatial Registration |
| **GPSA** | 2023 | Spatial Registration |

---

## 三、`evaluation_pipelines/` 统一评估框架

| 评估目录 | 对应任务 | 指标 |
|---------|---------|------|
| `scib_metrics/` | Dimension Reduction + Batch Correction + Clustering | NMI, ARI, ASW_label, ASW_batch, iLISI, cLISI, kBET, graph_conn, isolated_label_F1 等 |
| `classification/` | Cell Type Classification | Overall Accuracy, Average Accuracy, F1, Sensitivity, Specificity |
| `fs/` | Feature Selection | Specificity, Reproducibility (Pearson) |
| `imputation/` | Imputation | MSE, pFCS, pDES |
| `spatial_registration/` | Spatial Registration | PAA, SCS, LTARI |

> [!IMPORTANT]
> 评估 pipeline 中的 `scib_metrics/` 与你 scMMA 项目中使用的 NMI、ARI、ASW_label、cLISI、iLISI、ASW_batch 完全一致！可以保证指标的公平比较。

---

## 四、与 scMMA 数据集的对应关系

你的 scMMA 数据集来源就是 scMultiBench，因此数据格式兼容性很高：

| scMMA 数据集 | scMultiBench 中编号 | 主要模态 | 可用方法数 |
|-------------|-------------------|---------|-----------|
| D1 (PBMC CITE-seq) | D1 | RNA+ADT | ~18 (垂直整合) |
| D15 (10x Multiome) | D15 | RNA+ATAC | ~18 (垂直整合) |
| D18 (Neocortex) | D18 | RNA+ATAC | ~18 (垂直整合) |
| D22 (DOGMA, Single) | D22 | RNA+ADT+ATAC | ~10 (三模态方法较少) |
| D53 (WBC Atlas) | D53 | RNA+ADT | ~18 + ~14 (跨批次方法) |
| D54 (BMMC CITE-seq) | D54 | RNA+ADT | ~18 + ~14 |
| D56 (BMMC Multiome) | D56 | RNA+ATAC | ~18 + ~14 |
| D59 (DOGMA, Multi) | D59 | RNA+ADT+ATAC | ~10 |

---

## 五、使用建议

### 推荐工作流

1. **Clone 仓库**：`git clone https://github.com/PYangLab/scMultiBench.git`
2. **按方法安装环境**：每个方法可能需要独立 conda 环境（建议用 conda/mamba 分别创建）
3. **修改数据路径**：将 `tools_scripts/` 下各方法脚本中的数据路径指向你已有的数据集
4. **运行方法**：获得 embedding（通常保存为 `.h5` 文件）
5. **统一评估**：用 `evaluation_pipelines/scib_metrics/scib_metric.py` 计算指标

### ⚠️ 需要注意的点

- **环境隔离**：不同方法可能有依赖冲突（如 totalVI 需 scvi-tools，Seurat 需 R），建议为每个方法或一类方法创建独立 conda 环境
- **数据格式统一**：scMultiBench 的脚本通常读取 `.h5` 格式的 embedding 结果，确保你的 scMMA 输出也保存为同格式以便公平比较
- **GPU 需求**：部分深度学习方法（totalVI, MultiVI, GLUE 等）需要 GPU
- **三模态方法有限**：对于 D22 和 D59（三模态数据集），可用的对比方法数量明显较少

### 🎯 建议优先跑的对比方法

针对你的 **垂直整合 (Vertical Integration)** 任务，建议优先选择以下高影响力方法：

1. **totalVI** — 最常被引用的对比基准
2. **Seurat WNN (v4)** — 业界标准
3. **MOFA+** — 经典因子分析
4. **Multigrate** — scvi-tools 生态
5. **scMoMaT** — 多功能方法
6. **GLUE** — 如果涉及 RNA+ATAC 跨模态
