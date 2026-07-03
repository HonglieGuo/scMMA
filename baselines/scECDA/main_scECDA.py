import os
import sys
import time
import argparse
import json
import glob
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import scanpy as sc

# Add scECDA to system path for imports
sys.path.append(os.path.abspath('.'))
from network3 import Network
from loss import ContrastiveWithEntropyLoss
from kmeans_pytorch import kmeans

# Set random seeds
seed = 10
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

def preprocess_modality(adata, omic_type, reserve_dim):
    if omic_type != 'adt':
        sc.pp.filter_genes(adata, min_cells=2)
        if adata.X.max() >= 16:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=reserve_dim, subset=True)
    else:
        if adata.X.max() >= 16:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
    return adata

def init_kmeans_plus(cluster_temp, k, device):
    n = cluster_temp.shape[0]
    if k > n or k <= 0:
        raise ValueError("k value invalid")

    is_selected = torch.zeros(n, dtype=torch.bool)
    selected_indices = torch.zeros(k, dtype=torch.long)

    i = torch.randint(n, (1,)).item()
    is_selected[i] = True
    selected_indices[0] = i

    for i in range(1, k):
        candidate_indices = torch.nonzero(~is_selected).squeeze()
        if candidate_indices.numel() == 0:
            break

        cluster_selected = cluster_temp[selected_indices[:i]]
        cluster_candidates = cluster_temp[candidate_indices]

        dists = torch.cdist(cluster_candidates, cluster_selected)
        min_dists = dists.sum(dim=1)

        selected_index = candidate_indices[torch.argmax(min_dists)]
        selected_indices[i] = selected_index
        is_selected[selected_index] = True

    return cluster_temp[selected_indices].to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", nargs="+", required=True, help="Path to RNA h5 (can be multiple)")
    parser.add_argument("--path2", nargs="+", default=None, help="Path to ADT h5 (can be multiple)")
    parser.add_argument("--path3", nargs="+", default=None, help="Path to ATAC h5 (can be multiple)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--mse_epochs", type=int, default=500, help="Number of pretrain epochs (restored to official default)")
    parser.add_argument("--con_epochs", type=int, default=100, help="Number of contrastive epochs (restored to official default)")
    args = parser.parse_args()
    
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Determine active modalities and load
    paths_lst = [args.path1, args.path2, args.path3]
    omic_types = ['rna', 'adt', 'atac']
    
    active_paths = []
    active_types = []
    
    # RNA is always path1
    active_paths.append(args.path1)
    active_types.append('rna')
    
    # If path2 is ATAC (SNARE-seq) or ADT (CITE-seq)
    if args.path2 is not None:
        active_paths.append(args.path2)
        # Check if path2 contains adt or atac
        path_str = args.path2[0].lower()
        if 'adt' in path_str:
            active_types.append('adt')
        else:
            active_types.append('atac')
            
    # If path3 is ATAC
    if args.path3 is not None:
        active_paths.append(args.path3)
        active_types.append('atac')
        
    view = len(active_paths)
    print(f"Active modalities: {active_types}")
    
    # Load and concatenate batches
    xs = []
    dims = []
    feature_dim_ls = []
    
    for v in range(view):
        adatas = []
        for p in active_paths[v]:
            adatas.append(load_h5(p))
        adata = ad.concat(adatas, join="inner")
        
        # Preprocess
        reserve_dim = 3000
        adata = preprocess_modality(adata, active_types[v], reserve_dim)
        
        # Extract data matrix
        try:
            x = adata.X.toarray().astype(np.float32)
        except:
            x = adata.X.astype(np.float32)
            
        xs.append(x)
        dims.append(x.shape[1])
        
        # Determine latent feature dim
        if active_types[v] == 'rna':
            feature_dim_ls.append(500)
        elif active_types[v] == 'atac':
            feature_dim_ls.append(500)
        elif active_types[v] == 'adt':
            feature_dim_ls.append(min(50, x.shape[1]))
            
    n_cells = xs[0].shape[0]
    
    # Find number of classes/cell types from label files in directory
    dataset_dir = os.path.dirname(active_paths[0][0])
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
    
    # Create dataset loader
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, xs):
            self.xs = xs
        def __len__(self):
            return self.xs[0].shape[0]
        def __getitem__(self, idx):
            return [torch.from_numpy(u[idx]) for u in self.xs], 0
            
    dataset = SimpleDataset(xs)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, drop_last=False
    )
    
    # 2. Instantiate Network
    model = Network(view, dims, feature_dim_ls, class_num, depth=5, noise=0.03, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0)
    criterion1 = ContrastiveWithEntropyLoss(256, class_num, temperature_l=1.0, device=device).to(device)
    
    # 3. Stage 1: Pretrain Autoencoders
    print("Stage 1: Pretraining...")
    from tqdm import tqdm
    pbar_pre = tqdm(range(args.mse_epochs), desc="Stage 1: Pretraining", unit="epoch")
    for epoch in pbar_pre:
        model.train()
        tot_loss = 0.
        for batch_idx, (batch_xs, _) in enumerate(data_loader):
            for v in range(view):
                batch_xs[v] = batch_xs[v].to(device)
            optimizer.zero_grad()
            xrs = model(batch_xs, True)
            loss_list = [F.mse_loss(xrs[v], batch_xs[v], reduction='mean') for v in range(view)]
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        pbar_pre.set_postfix({"Loss": f"{tot_loss/len(data_loader):.6f}"})
            
    # 4. Initialize Clustering Centers As
    print("Initializing centroids (As)...")
    model.eval()
    with torch.no_grad():
        for v in range(view):
            inputs_v = torch.from_numpy(xs[v]).to(device)
            hidden = model.encoders[v](inputs_v).cpu()
            if active_types[v] != 'rna':
                cluster_centers = init_kmeans_plus(hidden, class_num, device)
                model.As[v].data = cluster_centers
            else:
                cluster_ids_x, cluster_centers = kmeans(X=hidden, num_clusters=class_num, distance='cosine', device=device)
                model.As[v].data = cluster_centers.to(device)
                
    # 5. Stage 2: Contrastive training
    print("Stage 2: Contrastive training...")
    pbar_con = tqdm(range(args.con_epochs), desc="Stage 2: Contrastive", unit="epoch")
    for epoch in pbar_con:
        model.train()
        tot_loss = 0.
        for batch_idx, (batch_xs, _) in enumerate(data_loader):
            for v in range(view):
                batch_xs[v] = batch_xs[v].to(device)
            optimizer.zero_grad()
            xrs, P, Qs, Qs_drop, cls, glb_feature = model(batch_xs)
            
            loss_list = []
            for v in range(view):
                for w in range(v+1, view):
                    loss_list.append(0.5 * criterion1.forward_label(Qs[v], Qs[w]))
                    loss_list.append(0.5 * criterion1.forward_label(Qs[v], Qs_drop[w]))
                    loss_list.append(0.5 * criterion1.forward_label(Qs_drop[v], Qs[w]))
                loss_list.append(1.0 * F.kl_div(torch.log(P + 1e-12), Qs[v], reduction='batchmean'))
                loss_list.append(F.mse_loss(batch_xs[v], xrs[v], reduction='mean'))
                
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            model.copy_weight()
        pbar_con.set_postfix({"Loss": f"{tot_loss/len(data_loader):.6f}"})
            
    # 6. Extract joint global embeddings
    print("Extracting cell embeddings...")
    model.eval()
    all_glb = []
    with torch.no_grad():
        for batch_xs, _ in test_loader:
            for v in range(view):
                batch_xs[v] = batch_xs[v].to(device)
            _, _, _, _, _, glb = model(batch_xs)
            all_glb.append(glb.cpu().numpy())
    joint_emb = np.concatenate(all_glb, axis=0)
    
    # 7. Save results
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
            "Actual_Epochs": args.mse_epochs + args.con_epochs
        }, f)

if __name__ == "__main__":
    main()
