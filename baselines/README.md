# 🚀 scMMA Baselines 运行指南

本目录包含 **scMMA** 项目中所有对比方法（Baselines）的运行环境配置与执行指南。

为了完全避免多模态单细胞整合模型之间由于底层库（如 `scvi-tools`, `anndata`, `scanpy`, `pytorch` 等）严重的版本冲突，本项目的对比方法推荐使用**两套相互隔离的 Conda 虚拟环境**来运行：

1. **`scmulti`**：主整合环境，适用于大多数对比方法（如 **MOFA2**, **scMoMaT**, **GLUE**, **totalVI** 等）。
2. **`env_multigrate`**：专用隔离环境，专为 **Multigrate** 精准定制，锁定了所有核心依赖版本以解决 API 冲突。

---

## 📂 环境依赖要求（Requirements）

我们已经在 `scMultiBench` 目录下为您导出了经过严格 dry-run 测试、完全兼容的依赖清单文件：

- 📝 **`scmulti` 环境包清单**：[requirements_scmulti.txt](file:///d:/code/research-projects/scMMA/baselines/scMultiBench/requirements_scmulti.txt)
- 📝 **`env_multigrate` 环境包清单**：[requirements_multigrate.txt](file:///d:/code/research-projects/scMMA/baselines/scMultiBench/requirements_multigrate.txt)

---

## 🛠️ Conda 环境搭建指引

请在终端中按如下步骤创建并配置这两个虚拟环境：

### 1. 创建并配置主环境 `scmulti`

```bash
# 创建并激活 Python 3.10 虚拟环境
conda create -n scmulti python=3.10 -y
conda activate scmulti

# 安装 Bedtools 依赖 (GLUE 等方法必须)
conda install -c bioconda bedtools -y

# 使用 requirements 安装所有 Python 依赖包
pip install -r scMultiBench/requirements_scmulti.txt
```

### 2. 创建并配置专用环境 `env_multigrate`

由于 `Multigrate` 依赖的旧版 `scvi-tools` 与新版 `anndata/scanpy/mudata` 在 API 上存在大面积不兼容，我们构建了经过严格版本锁定的专属依赖链条：

```bash
# 创建并激活 Python 3.10 虚拟环境
conda create -n env_multigrate python=3.10 -y
conda activate env_multigrate

# 使用 requirements 安装完整锁定的依赖包
pip install -r scMultiBench/requirements_multigrate.txt
```

> 💡 **重要版本锁机制说明 (env_multigrate)**:
> 
> - 锁定了 `setuptools==70.3.0`（高版本 setuptools 移除了 `pkg_resources`，会导致 pytorch-lightning 导入失败）。
> - 锁定了 `anndata==0.9.2` 和 `mudata==0.2.3`（防止新版 `SparseDataset` 弃用 API 破坏多模态 mudata 加载）。
> - 锁定了 `scvi-tools==0.20.3` 和 `multigrate==0.0.2`。
> - 锁定了 `numpy==1.26.4`（避免 NumPy 2.x 的二进制不兼容错误）。
> - 锁定了 `jax==0.4.25` 和 `flax==0.8.5`（高版本 JAX 删除了旧版 scvi-tools 调用的 `Device` API）。

---

## 🏃 自动化批处理运行

我们为您提供了一键式自动化跑批脚本 `run_scmultibench_baslines.sh`。该脚本能够**自动检测您的 Conda 安装路径，并根据不同的方法自动切换到正确的 Conda 虚拟环境**运行，同时规避了 GPU 架构潜在的兼容性问题。

### 运行方式

```bash
# 进入 scMultiBench 目录
cd scMultiBench

# 赋予执行权限并运行批处理
chmod +x run_scmultibench_baslines.sh
./run_scmultibench_baslines.sh
```

### 自动化脚本核心机制

脚本内部维护了方法与环境的关联映射（`METHOD_ENVS`）：

- **`MOFA2`** ➡️ `scmulti`
- **`scMoMaT`** ➡️ `scmulti`
- **`GLUE`** ➡️ `scmulti`
- **`Multigrate`** ➡️ `env_multigrate`

---

## 📝 独立方法手动运行指南

如果您需要对某个具体的方法单独执行训练与评估，可以手动激活对应环境并使用 `run_baseline.py` 调度。

### 1. MOFA2 (R & Python)

* 💡 **所需环境**：`scmulti`

* R 包 `MOFAPY2` 和 Python 的 `mofapy2` 已经预装并在主环境中绑定。
  
  ```bash
  conda activate scmulti
  cd scMultiBench
  python run_baseline.py --method MOFA2 --dataset D18
  ```

### 2. scMoMaT (Python)

* 💡 **所需环境**：`scmulti`
  
  ```bash
  conda activate scmulti
  cd scMultiBench
  python run_baseline.py --method scMoMaT --dataset D18
  ```

### 3. GLUE (图引导多模态对齐)

* 💡 **所需环境**：`scmulti`

* ⚠️ **数据依赖**：GLUE 的训练需要人类参考基因组的 GTF 注释文件。我们已在 `GLUE_human.py` 内部集成了**自动下载逻辑**。如果本地未检测到 GTF，脚本会自动从 EBI GENCODE 官方 FTP 镜像下载压缩文件 `gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz`（约 50MB）并妥善放置。
  
  ```bash
  conda activate scmulti
  cd scMultiBench
  python run_baseline.py --method GLUE --dataset D18
  ```

### 4. Multigrate (Python 模态整合)

* 💡 **所需环境**：`env_multigrate`

* ⚠️ **显卡兼容性警告 (CPU 模式建议)**：
  如果您使用的是超新架构显卡（例如 NVIDIA RTX 5090 D / sm_120 计算能力），系统预装的 PyTorch 2.5.1 CUDA 版本可能不支持该架构并导致运行时崩溃。
  
  - **建议做法**：通过在运行命令前添加 `CUDA_VISIBLE_DEVICES=""` 强制以 **CPU 模式** 稳定运行。
  
  - **升级 PyTorch（可选）**：如果希望正式运行时能够使用 GPU，需要在 `env_multigrate` 环境中手动更新 PyTorch 到支持您显卡架构的版本（如通过官方源升级 CUDA 12.4+ 对应的 torch）。
    
    ```bash
    conda activate env_multigrate
    cd scMultiBench
    # 强制使用 CPU 模式稳定运行
    CUDA_VISIBLE_DEVICES="" python run_baseline.py --method Multigrate --dataset D18 --conda-env env_multigrate
    ```

---

## 📊 实验输出与日志记录

- **模型 Embeddings 结果**：默认会保存在 `baselines/experiment_output/<Method>/<Dataset>/embedding.h5` 路径中。
- **性能评估结果日志**：每次运行成功后，系统会自动使用 `eval_embedding.py` 进行评估，并将核心整合指标（包括 **NMI**, **ARI**, **ASW**, **cLISI**, **Overall** 等指标以及运行时间、所用环境）追加记录在：
  📝 **`baselines/baseline_experiment_log.csv`**
