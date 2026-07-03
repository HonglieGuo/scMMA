# Single-cell multiomics data alignment and integration based on enhanced contrastive learning and differential attention mechanism
## Environment Setup

To set up the environment, execute the following command:

```bash
pip install -r requirements.txt
```

------

## Input Data Format

For RNA, ATAC, and ADT data, the input type is AnnData format, where the observed attributes (`obs`) store the true cell type labels, specifically in `obs['cell_type']`. The data matrix should follow the format where rows represent cells and columns represent features.

------

## Model  Training

To perform model training, specify the file paths for each omics dataset and configure the necessary parameters. Ensure that the following directories are created in the current working directory:

- `models/`: For storing trained model parameters.
- `latent/`: For storing latent features generated during inference and prediction results.

Run the training script with:

```bash
python train.py
```

------

## Model Inference

For model inference, execute the following command:

```bash
python inference.py
```

------

## Data and Model Parameters

Due to GitHub's file size constraints, the dataset and trained model parameters are available on Baidu CloudDisk. Access the resources to download the large-scale datasets and model weights for this study.

Link: https://pan.baidu.com/s/1LfNgQrcX0S1iQIA8cNgBAA?pwd=jiad 

password: jiad 
