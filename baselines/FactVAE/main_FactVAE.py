import os
import time
import argparse
import h5py
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import scipy.sparse as sp

from FactVAE.model import VAE
from FactVAE.loader import dataInstance

# Set random seeds for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_h5(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        features = np.array(f['matrix/features']).astype(str)
        barcodes = np.array(f['matrix/barcodes']).astype(str)
    
    # Create AnnData object
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = features
    adata.obs_names = barcodes
    return adata

def load_multiple_h5(paths):
    adatas = []
    for path in paths:
        adatas.append(load_h5(path))
    if len(adatas) == 1:
        return adatas[0]
    return ad.concat(adatas, join='inner')

def preprocess_rna(rna):
    rna.layers["counts"] = rna.X.copy()
    
    # Highly variable genes (2000 genes)
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
    
    # Normalize & log1p
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    
    # PCA
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    
    # Filter to highly variable genes
    rna_hvg = rna[:, rna.var.highly_variable].copy()
    rna_hvg.var["highly_variable"] = True
    
    # CRITICAL FIX: ZINB loss requires raw non-negative counts.
    # Set .raw to the raw counts, NOT the scaled (negative) data.
    rna_counts = ad.AnnData(rna_hvg.layers["counts"])
    rna_counts.var_names = rna_hvg.var_names
    rna_counts.obs_names = rna_hvg.obs_names
    rna_counts.var = rna_hvg.var.copy()
    rna_hvg.raw = rna_counts
    
    # Compute size factors
    rna_hvg.obs["size_factors"] = np.asarray(rna_hvg.layers["counts"].sum(axis=1)).flatten() + 1e-6
    rna_hvg.obs["size_factors"] /= rna_hvg.obs["size_factors"].mean()
    
    return rna_hvg

def preprocess_atac(atac):
    atac.layers["counts"] = atac.X.copy()
    
    # Implement TF-IDF + PCA (LSI)
    # TF-IDF
    n_peaks = atac.shape[1]
    col_sums = np.asarray(atac.X.sum(axis=0)).flatten()
    idf = np.log(1.0 + atac.shape[0] / (col_sums + 1e-12))
    tf = atac.X.multiply(1.0 / (np.asarray(atac.X.sum(axis=1)).reshape(-1, 1) + 1e-12))
    tfidf = tf.multiply(idf)
    atac.obsm["X_lsi"] = tfidf.toarray()
    
    # Truncated SVD (LSI)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, random_state=seed)
    atac.obsm["X_lsi"] = svd.fit_transform(atac.obsm["X_lsi"])
    
    atac.raw = atac.copy()
    
    # Compute size factors
    atac.obs["size_factors"] = np.asarray(atac.layers["counts"].sum(axis=1)).flatten() + 1e-6
    atac.obs["size_factors"] /= atac.obs["size_factors"].mean()
    
    return atac

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", nargs='+', required=True, help="Path to RNA h5")
    parser.add_argument("--path2", nargs='+', required=True, help="Path to ATAC h5")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 1. Load data
    print("Loading data...")
    rna = load_multiple_h5(args.path1)
    atac = load_multiple_h5(args.path2)
    
    # 2. Preprocess data
    print("Preprocessing data...")
    rna = preprocess_rna(rna)
    atac = preprocess_atac(atac)
    
    # Add dummy cell types for loader compatibility
    rna.obs["cell_type"] = "dummy"
    atac.obs["cell_type"] = "dummy"
    
    # 3. Create guidance prior graph (dummy sparse matrix to avoid GTF requirement)
    print("Creating prior graph...")
    n_genes = rna.shape[1]
    n_peaks = atac.shape[1]
    rows = np.arange(n_genes)
    cols = np.arange(n_genes) % n_peaks
    data = np.ones(n_genes, dtype=np.float32)
    prior_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_genes, n_peaks))
    prior = torch.tensor(prior_sparse.toarray(), dtype=torch.float).to(device)
    
    # 4. Initialize Data Loader
    train_dataset = dataInstance(
        RNA_adata=rna,
        ATAC_adata=atac,
        RNA_label_colname="cell_type",
        ATAC_label_colname="cell_type",
        RNA_rep="X_pca",
        ATAC_rep="X_lsi"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True
    )
    
    # 5. Initialize FactVAE
    vae = VAE(
        layer_rna_e=[100, 512, 256], 
        layer_atac_e=[100, 256],
        zdim=50, 
        rna_enc_dropout=0.2, 
        atac_enc_dropout=0.2, 
        prior=prior, 
        beta=1,
        type_d_rna="ZINB", 
        type_d_atac="ZINB",
        gene_dim=n_genes, 
        peak_dim=n_peaks, 
        device=device
    ).to(device)
    
    genopts = {
        'vae': optim.Adam(vae.parameters(), lr=0.001),
        'enc_atac': optim.Adam(vae.ATAC_encoder.parameters(), lr=0.0005)
    }
    train_schedulers = {
        "vae": ExponentialLR(genopts["vae"], gamma=0.995),
        "enc_atac": ExponentialLR(genopts["enc_atac"], gamma=0.995)
    }
    
    # 6. Training loop
    print("Training FactVAE...")
    from tqdm import tqdm
    pbar = tqdm(range(args.epochs), desc="FactVAE Training", unit="epoch")
    for epoch in pbar:
        vae.train()
        for i, (index, rna_sample, atac_sample) in enumerate(train_loader):
            rna_train_data = rna_sample[0].float().to(device)
            atac_train_data = atac_sample[0].float().to(device)
            rna_raw = rna_sample[1].float().to(device)
            atac_raw = atac_sample[1].float().to(device)
            rna_size_factors = rna_sample[2].float().to(device)
            atac_size_factors = atac_sample[2].float().to(device)
            
            rna_loss, _, _, atac_loss, _, _ = vae.forward(
                rna_inputs=rna_train_data, rna_X=rna_raw, rna_scale_factor=rna_size_factors, rna_beta=0.001, rna_sigma=0.1,
                atac_inputs=atac_train_data, atac_X=atac_raw, atac_scale_factor=atac_size_factors, atac_beta=0.001, atac_sigma=0.002
            )
            
            loss_prior = vae.forward_prior_loss()
            loss = rna_loss + atac_loss + 1.0 * torch.exp(torch.clamp(loss_prior, max=20.0))
            
            genopts["vae"].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            genopts["vae"].step()
            
            loss_similar = vae.forward_similar_loss(rna_inputs=rna_train_data, atac_inputs=atac_train_data)
            genopts["enc_atac"].zero_grad()
            loss_similar.backward()
            torch.nn.utils.clip_grad_norm_(vae.ATAC_encoder.parameters(), max_norm=5.0)
            genopts["enc_atac"].step()
            
        train_schedulers["vae"].step()
        train_schedulers["enc_atac"].step()
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
    # 7. Get Embeddings
    print("Extracting cell embeddings...")
    vae.eval()
    with torch.no_grad():
        rna_latent, atac_latent = vae.get_emb(
            torch.tensor(rna.obsm["X_pca"]).to(device),
            torch.tensor(atac.obsm["X_lsi"], dtype=torch.float).to(device)
        )
    
    # Concatenate RNA and ATAC embeddings to form joint representation
    joint_emb = np.concatenate([rna_latent.cpu().numpy(), atac_latent.cpu().numpy()], axis=1)
    
    # 8. Save output
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
