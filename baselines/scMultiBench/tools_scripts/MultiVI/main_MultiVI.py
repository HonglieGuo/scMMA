import os
import scvi
import muon
import h5py
import random
import anndata
import argparse
import numpy as np
import mudata as md
import scanpy as sc
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
import torch
import json
from util import data_loader_multi_single, data_loader_multi_multi, split_dataset_by_modality, organize_multiome_datasets, sort_features_by_modality, filter_features, setup_anndata_for_multivi, get_normalized_expression, create_multivi_model, train_multivi_model, get_accessibility_estimates, save_multivi_model, load_multivi_model

random.seed(1)
parser = argparse.ArgumentParser("MultiVI")
parser.add_argument('--path1', nargs='+', default=['NULL'], help='path to RNA')
parser.add_argument('--path2', nargs='+', default=['NULL'], help='path to ATAC')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

def load_and_concat(paths, prefix):
    adatas = []
    for p in paths:
        if p == 'NULL': continue
        with h5py.File(p, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            adata = AnnData(X=X)
            adata.var_names = [f"{prefix}_{i}" for i in range(adata.shape[1])]
            adata.X = csr_matrix(np.matrix(adata.X))
            adatas.append(adata)
    if len(adatas) == 0:
        return None
    return anndata.concat(adatas, join='outer')

def run_MultiVI(path1_list, path2_list, n_epochs=500, lr=1e-3):
    rna = load_and_concat(path1_list, "RNA")
    atac = load_and_concat(path2_list, "ATAC")
    
    n_genes = rna.shape[1] if rna is not None else 0
    n_regions = atac.shape[1] if atac is not None else 0
    
    # We must explicitly add the modality column for MultiVI setup
    if rna is not None:
        rna.obs['modality'] = 'Gene Expression'
    if atac is not None:
        atac.obs['modality'] = 'Peaks'
        
    adatas = []
    if rna is not None: adatas.append(rna)
    if atac is not None: adatas.append(atac)
    
    if len(adatas) == 0: return None
    adata = anndata.concat(adatas, join='outer', fill_value=0)
    
    ordered_features = []
    if rna is not None: ordered_features.extend(rna.var_names)
    if atac is not None: ordered_features.extend(atac.var_names)
    adata = adata[:, ordered_features].copy()
    
    # Setup anndata for multivi
    scvi.model.MULTIVI.setup_anndata(adata, batch_key="modality")
    
    # Create the model
    model = scvi.model.MULTIVI(adata, n_genes=n_genes, n_regions=n_regions)
    
    try:
        params_m = sum(p.numel() for p in model.module.parameters()) / 1e6
    except:
        params_m = 0.0
        
    # In scvi v1.x kwargs might be passed to train instead of custom methods.
    model.train(max_epochs=n_epochs, lr=lr)
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    
    # save to metrics_ext.json
    ext_file = os.path.join(args.save_path, "metrics_ext.json")
    os.makedirs(args.save_path, exist_ok=True)
    with open(ext_file, "w") as f:
        json.dump({"Peak_Mem_GB": f"{peak_mem:.2f}", "Params_M": f"{params_m:.2f}"}, f)
        
    return model, adata

n_epochs = 500
lr = 1e-3
model, adata = run_MultiVI(args.path1, args.path2, n_epochs, lr)

latent_key = "X_multivi"
adata.obsm[latent_key] = model.get_latent_representation()
result = adata.obsm[latent_key]

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
    
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
