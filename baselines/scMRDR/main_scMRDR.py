import os
import sys
import time
import argparse
import json
import h5py
import numpy as np
import scipy.sparse as sp
import torch
import anndata as ad
import scanpy as sc

# Add scMRDR src directory to path for imports
sys.path.append(os.path.abspath('./src'))
from scMRDR.module import Integration

# Set random seeds
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def load_h5(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        features = np.array(f['matrix/features']).astype(str)
        barcodes = np.array(f['matrix/barcodes']).astype(str)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = features
    adata.obs_names = barcodes
    return adata

def preprocess_modality(adata, highly_genes=2000):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    n_top = min(highly_genes, adata.shape[1])
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
    return adata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", nargs="+", required=True, help="Path to RNA h5 (can be multiple)")
    parser.add_argument("--path2", nargs="+", required=True, help="Path to ATAC/ADT h5 (can be multiple)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (restored to official default)")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 1. Load data
    print("Loading data...")
    rna_adatas = []
    for i, p in enumerate(args.path1):
        adata = load_h5(p)
        adata.obs["batch"] = str(i)
        rna_adatas.append(adata)
    rna = ad.concat(rna_adatas, join="inner")
    
    mod2_adatas = []
    for i, p in enumerate(args.path2):
        adata = load_h5(p)
        adata.obs["batch"] = str(i)
        mod2_adatas.append(adata)
    mod2 = ad.concat(mod2_adatas, join="inner")
    
    # Ensure cell counts match
    assert rna.shape[0] == mod2.shape[0], "RNA and Mod2 cell counts mismatch!"
    
    # 2. Preprocess each modality separately
    print("Preprocessing data...")
    rna = preprocess_modality(rna, highly_genes=2000)
    mod2 = preprocess_modality(mod2, highly_genes=2000)
    
    n_cells = rna.shape[0]
    n_genes = rna.shape[1]
    n_features2 = mod2.shape[1]
    n_batches = len(args.path1)
    
    print(f"Shapes: RNA {rna.shape}, Mod2 {mod2.shape}")
    
    # 3. Construct unified mosaic AnnData object for scMRDR
    # Shape: (2 * n_cells, n_genes + n_features2)
    # We construct the count matrix:
    # [ rna_counts , zeros_mod2 ] -> modality 0
    # [ zeros_rna  , mod2_counts ] -> modality 1
    
    print("Building mosaic AnnData...")
    # Convert sparse matrices to dense arrays for simplicity of concatenation
    rna_x = rna.X.toarray()
    mod2_x = mod2.X.toarray()
    
    top_x = np.concatenate([rna_x, np.zeros((n_cells, n_features2), dtype=np.float32)], axis=1)
    bottom_x = np.concatenate([np.zeros((n_cells, n_genes), dtype=np.float32), mod2_x], axis=1)
    combined_x = np.concatenate([top_x, bottom_x], axis=0)
    
    # Create combined AnnData
    combined_adata = ad.AnnData(X=sp.csr_matrix(combined_x))
    
    # Metadata
    modalities = ["0"] * n_cells + ["1"] * n_cells
    batches = list(rna.obs["batch"].values) + list(mod2.obs["batch"].values)
    
    combined_adata.obs["modality"] = modalities
    combined_adata.obs["batch"] = batches
    
    # Feature list
    feature_list = {
        "0": list(range(n_genes)),
        "1": list(range(n_genes, n_genes + n_features2))
    }
    
    # 4. Initialize Integration class
    print("Initializing Integration...")
    model = Integration(
        data=combined_adata,
        layer=None,
        modality_key="modality",
        batch_key="batch" if n_batches > 1 else None,
        feature_list=feature_list,
        distribution="ZINB"
    )
    
    # 5. Setup model architecture
    model.setup(
        hidden_layers=[128, 128],
        latent_dim_shared=20,
        latent_dim_specific=20,
        beta=2,
        gamma=5,
        lambda_adv=5,
        dropout_rate=0.5
    )
    
    # 6. Train the model
    print("Training scMRDR...")
    model.train(
        epoch_num=args.epochs,
        batch_size=128,
        lr=1e-3,
        adaptlr=False,
        num_warmup=0,
        early_stopping=True,
        valid_prop=0.1,
        patience=15,
        random_state=seed
    )
    
    # 7. Extract latent representations
    print("Extracting cell embeddings...")
    z_shared, z_specific = model.inference(n_samples=1, update=False, returns=True)
    
    # Cell i's joint embedding is the concatenation of shared representation of its RNA and ATAC/ADT parts
    joint_emb = np.concatenate([z_shared[:n_cells], z_shared[n_cells:]], axis=1)
    
    # 8. Save results
    os.makedirs(args.save_path, exist_ok=True)
    emb_file = os.path.join(args.save_path, "embedding.h5")
    with h5py.File(emb_file, 'w') as f:
        f.create_dataset('data', data=joint_emb)
        
    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time:.1f}s. Saved embedding to {emb_file}")
    
    # Save metrics.json for logger
    metrics_file = os.path.join(args.save_path, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "time": elapsed_time,
            "Embedding_Dim": joint_emb.shape[1],
            "Actual_Epochs": args.epochs
        }, f)

if __name__ == "__main__":
    main()
