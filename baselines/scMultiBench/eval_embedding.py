import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import h5py

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.metrics import MetricsWrapper

def read_labels(label_paths):
    all_labels = []
    for path in label_paths:
        label_fs = pd.read_csv(path, header=None, index_col=False)
        if label_fs.shape[1] > 1:
            label_fs = label_fs.iloc[1:, 1] # Skip header, get 2nd column
        else:
            label_fs = label_fs.iloc[1:, 0] # Skip header, get 1st column
        all_labels.append(label_fs)
    all_labels = pd.concat(all_labels)
    all_labels_categorical = pd.Categorical(all_labels)
    codes = all_labels_categorical.codes
    return np.array(codes).astype('int32'), all_labels_categorical

def read_batches(label_paths):
    all_batchs = []
    for i, path in enumerate(label_paths):
        label_fs = pd.read_csv(path, header=None, index_col=False)
        if label_fs.shape[1] > 1:
            label_fs = label_fs.iloc[1:, 1]
        else:
            label_fs = label_fs.iloc[1:, 0]
        batch_temp = np.ones(len(label_fs)) + i
        all_batchs.append(batch_temp)
    return np.concatenate(all_batchs)

def load_embedding(data_paths):
    """Load embedding from one or multiple h5 files."""
    data = []
    for path in data_paths:
        with h5py.File(path, "r") as f:
            if 'data' in f:
                X = np.asarray(f['data'])
            elif 'matrix/data' in f:
                X = np.asarray(f['matrix/data'])
            else:
                X = np.asarray(f[list(f.keys())[0]]) # guess
            
            # Ensure shape is (cells, features)
            if X.shape[0] < X.shape[1]:
                X = X.transpose()
        data.append(X)
    return np.concatenate(data, axis=0)

def evaluate_embedding(embedding_paths, cty_paths, save_path=None, split_for_foscttm=False):
    print(f"Loading embeddings from {embedding_paths}...")
    X_emb = load_embedding(embedding_paths)
    
    print(f"Loading cell types from {cty_paths}...")
    cty_codes, cty_categorical = read_labels(cty_paths)
    batch_codes = read_batches(cty_paths)
    
    n_labels = len(cty_codes)
    n_cells = X_emb.shape[0]
    
    rna_emb = None
    mod2_emb = None
    
    if n_cells == 2 * n_labels:
        print("💡 Detected vertically concatenated embeddings (2N cells). Splitting for alignment metrics and duplicating labels...")
        rna_emb = X_emb[n_labels:, :] # Actually, for uniPort it's ATAC then RNA, but for FOSCTTM distance it's symmetric. Let's just split.
        mod2_emb = X_emb[:n_labels, :]
        
        # Duplicate labels so clustering metrics evaluate on all 2N points together!
        cty_codes = np.concatenate([cty_codes, cty_codes])
        batch_codes = np.concatenate([batch_codes, batch_codes])
        
    elif split_for_foscttm:
        print("💡 Detected horizontally concatenated embeddings (--split_for_foscttm). Splitting feature dimension...")
        half_dim = X_emb.shape[1] // 2
        rna_emb = X_emb[:, :half_dim]
        mod2_emb = X_emb[:, half_dim:]
    
    adata = ad.AnnData(X_emb)
    adata.obsm["X_emb"] = adata.X
    adata.obs["celltype"] = cty_codes
    adata.obs["celltype"] = adata.obs['celltype'].astype(str).astype('category')
    adata.obs["batch"] = batch_codes
    adata.obs["batch"] = adata.obs['batch'].astype(str).astype('category')
        
    print("Computing metrics via MetricsWrapper...")
    metrics = MetricsWrapper.compute_metrics(
        adata=adata,
        batch_key="batch",
        label_key="celltype",
        embedding_key="X_emb",
        rna_embeddings=rna_emb,
        mod2_embeddings=mod2_emb
    )
    
    results = {}
    results['Leiden_Res'] = metrics.get('leiden_res', np.nan)
    results['Embedding_Dim'] = X_emb.shape[1]
    
    results['NMI'] = metrics.get('nmi', np.nan)
    results['ARI'] = metrics.get('ari', np.nan)
    results['ASW_label'] = metrics.get('asw_label', np.nan)
    results['cLISI'] = metrics.get('clisi', np.nan)
    
    results['FOSCTTM'] = metrics.get('foscttm', np.nan)
    results['Match@5'] = metrics.get('match_at_5', np.nan)
    results['LTA'] = metrics.get('lta', np.nan)
    
    results['Batch_ASW'] = metrics.get('batch_asw', np.nan)
    results['iLISI'] = metrics.get('ilisi', np.nan)
    results['ASW_batch_raw'] = metrics.get('asw_batch_raw', np.nan)
    results['Overall'] = metrics.get('overall_score', np.nan)
    
    df_res = pd.DataFrame([results])
    print("\nResults:")
    print(df_res.T)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, "metrics.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"Saved metrics to {csv_path}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("eval_embedding")
    parser.add_argument('--embedding_path', nargs='+', required=True, help='Path to embedding.h5 (can be multiple)')
    parser.add_argument('--cty_path', nargs='+', required=True, help='Path to cty.csv (can be multiple)')
    parser.add_argument('--save_path', type=str, default=None, help='Directory to save metrics.csv')
    parser.add_argument('--split_for_foscttm', action='store_true', help='If provided, splits embedding in half to calculate FOSCTTM.')
    args = parser.parse_args()
    
    evaluate_embedding(args.embedding_path, args.cty_path, args.save_path, args.split_for_foscttm)
