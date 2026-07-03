import os
import sys
import time
import argparse
import json
import glob
import h5py
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import scanpy as sc
from sklearn.cluster import KMeans

# Add scMDCF subdirectory to path for imports
sys.path.append(os.path.abspath('.'))
from scMDCF.layer import scMDCF

# Set random seeds
seed = 0
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
    sc.pp.scale(adata)
    return adata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", nargs="+", required=True, help="Path to RNA h5 (can be multiple)")
    parser.add_argument("--path2", nargs="+", required=True, help="Path to ATAC/ADT h5 (can be multiple)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--epoch_pre", type=int, default=200, help="Number of pretrain epochs (restored to official default)")
    parser.add_argument("--epoch_alt", type=int, default=200, help="Number of alt training epochs (restored to official default)")
    args = parser.parse_args()
    
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load data
    print("Loading data...")
    rna_adatas = []
    for p in args.path1:
        rna_adatas.append(load_h5(p))
    rna = ad.concat(rna_adatas, join="inner")
    
    mod2_adatas = []
    for p in args.path2:
        mod2_adatas.append(load_h5(p))
    mod2 = ad.concat(mod2_adatas, join="inner")
    
    # Ensure cell counts match
    assert rna.shape[0] == mod2.shape[0], "RNA and Mod2 cell counts mismatch!"
    
    # 2. Preprocess
    print("Preprocessing data...")
    rna = preprocess_modality(rna, highly_genes=2000)
    mod2 = preprocess_modality(mod2, highly_genes=2000)
    
    rna_dim = rna.shape[1]
    mod2_dim = mod2.shape[1]
    n_cells = rna.shape[0]
    
    # Find number of classes/cell types from label files in directory
    dataset_dir = os.path.dirname(args.path1[0])
    label_files = glob.glob(os.path.join(dataset_dir, "cty*.csv"))
    if not label_files:
        label_files = glob.glob(os.path.join(dataset_dir, "cty.csv"))
    
    unique_labels = set()
    for lf in label_files:
        try:
            df = pd.read_csv(lf, header=None)
            col = 1 if df.shape[1] > 1 else 0
            unique_labels.update(df.iloc[1:, col].astype(str).tolist())
        except Exception as e:
            print(f"Warning reading label file {lf}: {e}")
            
    class_num = len(unique_labels) if len(unique_labels) > 0 else 15
    print(f"Number of classes detected: {class_num}")
    
    # 3. Setup model configs
    model_args = argparse.Namespace()
    model_args.enc1 = 512
    model_args.enc2 = 64
    model_args.zdim = 32
    model_args.alpha = 1.0
    model_args.gamma = 1.0
    model_args.lamb = 1.0
    model_args.n_clusters = class_num
    
    model_args.layere_omics1_view = [rna_dim, model_args.enc1, model_args.enc2]
    model_args.layere_omics2_view = [mod2_dim, model_args.enc1, model_args.enc2]
    model_args.layerd_omics1_view = [model_args.zdim, model_args.enc2, model_args.enc1, rna_dim]
    model_args.layerd_omics2_view = [model_args.zdim, model_args.enc2, model_args.enc1, mod2_dim]
    model_args.fusion_layer = [model_args.enc2 * 2, model_args.zdim]
    
    model = scMDCF(model_args).to(device)
    
    # Convert data to tensors
    x_rna = torch.from_numpy(rna.X).to(device).float()
    x_mod2 = torch.from_numpy(mod2.X).to(device).float()
    
    # 4. Stage 1: Pre-training
    print("Stage 1: Pre-training...")
    optimizer_pre = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    from tqdm import tqdm
    pbar_pre = tqdm(range(args.epoch_pre), desc="Stage 1: Pre-training", unit="epoch")
    for epoch in pbar_pre:
        model.train()
        z_RNA, z_ATAC, rec_RNA, rec_ATAC, z, _ = model(x_rna, x_mod2)
        loss_recrna = F.mse_loss(rec_RNA, x_rna)
        loss_recatac = F.mse_loss(rec_ATAC, x_mod2)
        cl_loss = model.crossview_contrastive_Loss(z_ATAC, z_RNA, lamb=model_args.lamb)
        loss = loss_recrna + loss_recatac + 0.1 * cl_loss
        
        optimizer_pre.zero_grad()
        loss.backward()
        optimizer_pre.step()
        
        pbar_pre.set_postfix({"Loss": f"{loss.item():.4f}"})
            
    # 5. Initialize Centroids using KMeans
    print("Initializing centroids...")
    model.eval()
    with torch.no_grad():
        _, _, _, _, z, _ = model(x_rna, x_mod2)
    kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=seed)
    kmeans.fit(z.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    
    # 6. Stage 2: Alternating Joint training
    print("Stage 2: Alternating training...")
    optimizer_alt = torch.optim.Adadelta(model.parameters(), lr=1e-3, rho=0.8)
    
    # Alternate loss weights: 0.1, 10.0, 1.0, 5.0
    weight1 = 0.1
    weight2 = 10.0
    weight3 = 1.0
    weight4 = 5.0
    
    pbar_alt = tqdm(range(args.epoch_alt), desc="Stage 2: Alternating", unit="epoch")
    for epoch in pbar_alt:
        model.train()
        z_RNA, z_ATAC, rec_RNA, rec_ATAC, z, q = model(x_rna, x_mod2)
        p = model.target_distribution(q).detach()
        
        loss_recrna = F.mse_loss(rec_RNA, x_rna)
        loss_recatac = F.mse_loss(rec_ATAC, x_mod2)
        cl_loss = model.crossview_contrastive_Loss(z_ATAC, z_RNA, lamb=model_args.lamb)
        loss_clu = model.cluster_loss(model_args, p, q)
        
        loss = weight1 * loss_recrna + weight2 * loss_recatac + weight3 * loss_clu + weight4 * cl_loss
        
        optimizer_alt.zero_grad()
        loss.backward()
        optimizer_alt.step()
        
        pbar_alt.set_postfix({"Loss": f"{loss.item():.4f}"})
            
    # 7. Extract Joint cell embeddings
    print("Extracting cell embeddings...")
    model.eval()
    with torch.no_grad():
        _, _, _, _, joint_emb, _ = model(x_rna, x_mod2)
    joint_emb = joint_emb.cpu().numpy()
    
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
            "Actual_Epochs": args.epoch_pre + args.epoch_alt
        }, f)

if __name__ == "__main__":
    main()
