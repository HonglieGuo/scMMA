import os
import time
import json
import h5py
import anndata
import scglue
import argparse
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import networkx as nx
from scipy import sparse
from anndata import AnnData
import torch
from itertools import chain
from matplotlib import rcParams

scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

def load_rna(rna_path):
    with h5py.File(rna_path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        rna = AnnData(X=sparse.csr_matrix(X))
        rna.var_names = np.array(f['matrix/features']).astype(str)
        rna.obs_names = np.array(f['matrix/barcodes']).astype(str)
    return rna

def load_atac(atac_peaks_path):
    with h5py.File(atac_peaks_path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        atac = AnnData(X=sparse.csr_matrix(X))
        atac.var_names = np.array(f['matrix/features']).astype(str)
        atac.obs_names = np.array(f['matrix/barcodes']).astype(str)
    return atac

def convert_gene_name(rna):
    converted_gene_names = []
    for name in rna.var_names:
        if "-" in name:
            parts = name.split("-")
            converted_parts = [part.capitalize() for part in parts]
            converted_name = "-".join(converted_parts)
        else:
            converted_name = name.capitalize()

        if 'rik' in converted_name:
            converted_name = converted_name.upper()
            converted_name = converted_name.replace('RIK', 'Rik')  
        converted_gene_names.append(converted_name)
    rna.var_names = converted_gene_names
    return rna

def preprocess_rna(rna):
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    return rna

def preprocess_atac(atac):
    scglue.data.lsi(atac, n_components=100, n_iter=15)
    return atac

def main():
    parser = argparse.ArgumentParser("GLUE")
    parser.add_argument('--path1', nargs='+', required=True, help='path to train gene')
    parser.add_argument('--path2', nargs='+', required=True, help='path to train peak')
    parser.add_argument('--save_path', type=str, required=True, help='path to save the output data')
    args = parser.parse_args()

    start_time = time.time()

    # GLUE typically runs on single-batch data in this script form
    print("Loading data...")
    rna_adatas = []
    for p in args.path1:
        rna_adatas.append(load_rna(p))
    if len(rna_adatas) == 1:
        rna = rna_adatas[0]
    else:
        rna = ad.concat(rna_adatas)
        batch_labels = []
        for i, a in enumerate(rna_adatas):
            batch_labels.extend([str(i)] * a.shape[0])
        rna.obs["batch"] = pd.Categorical(batch_labels)
        
    atac_adatas = []
    for p in args.path2:
        atac_adatas.append(load_atac(p))
    if len(atac_adatas) == 1:
        atac = atac_adatas[0]
    else:
        atac = ad.concat(atac_adatas)
        batch_labels = []
        for i, a in enumerate(atac_adatas):
            batch_labels.extend([str(i)] * a.shape[0])
        atac.obs["batch"] = pd.Categorical(batch_labels)

    print("Preprocessing data (this might take a few minutes)...")
    rna = convert_gene_name(rna)
    rna.var_names = rna.var_names.str.upper()
    rna = preprocess_rna(rna)
    atac = preprocess_atac(atac)

    print("Resolving GTF annotations...")
    # Resolve GTF
    gtf_filename = "gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_paths = [
        os.path.join(script_dir, gtf_filename),
        os.path.join(script_dir, "..", "scMultiBench", gtf_filename), # Look in old location
        os.path.join(os.getcwd(), gtf_filename)
    ]
    
    found = False
    for p in potential_paths:
        if os.path.exists(p) and os.path.getsize(p) > 40_000_000:
            gtf_filename = p
            found = True
            break
            
    if not found:
        print(f"🔍 Could not find {gtf_filename}. Auto-downloading from EBI GENCODE...")
        url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz"
        try:
            import urllib.request
            gtf_filename = os.path.join(script_dir, "gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz")
            print(f"Downloading to: {gtf_filename}")
            urllib.request.urlretrieve(url, gtf_filename)
            print("\n✅ Download complete!")
        except Exception as e:
            print(f"\n❌ Auto-download failed: {e}")
            raise e

    scglue.data.get_gene_annotation(rna, gtf=gtf_filename, gtf_by="gene_name")

    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

    rna = rna[:, ~rna.var["chrom"].isna()]
    if "strand" in rna.var.columns:
        valid_strand = rna.var["strand"].isin(["+", "-"])
        rna = rna[:, valid_strand]

    print("Building RNA-anchored guidance graph...")
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    
    print("Configuring datasets for SCGLUE...")
    use_batch_rna = "batch" if "batch" in rna.obs else None
    use_batch_atac = "batch" if "batch" in atac.obs else None
    scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_layer="counts", use_rep="X_pca", use_batch=use_batch_rna)
    scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep="X_lsi", use_batch=use_batch_atac)
    
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()

    print("Starting SCGLUE training...")
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac}, guidance_hvf,
        fit_kws={"directory": "glue"}
    )

    print("Extracting cell embeddings...")
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    
    # -------------------------------------------------------------
    # CRITICAL CHANGE: Concatenate instead of averaging
    # -------------------------------------------------------------
    # result = (rna.obsm["X_glue"] + atac.obsm["X_glue"]) / 2 # Old behavior
    
    joint_emb = np.concatenate([rna.obsm["X_glue"], atac.obsm["X_glue"]], axis=1)
    
    os.makedirs(args.save_path, exist_ok=True)
    emb_file = os.path.join(args.save_path, "embedding.h5")
    with h5py.File(emb_file, 'w') as f:
        f.create_dataset('data', data=joint_emb)
        
    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time:.1f}s. Saved embedding to {emb_file}")
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    try:
        params_m = sum(p.numel() for p in glue.net.parameters()) / 1e6
    except:
        params_m = 0.0
    
    # Save metrics.json for logger
    metrics_file = os.path.join(args.save_path, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "time": elapsed_time,
            "Embedding_Dim": joint_emb.shape[1],
            "Actual_Epochs": "Auto", # SCGLUE decides internally
            "Peak_Mem_GB": f"{peak_mem:.2f}",
            "Params_M": f"{params_m:.2f}"
        }, f)

if __name__ == "__main__":
    main()
