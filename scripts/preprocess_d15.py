import os
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from pathlib import Path

def convert_h5_to_adata(h5_path, cty_df=None):
    """
    Reads the custom h5 format and converts it to an AnnData object.
    """
    print(f"Reading {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # data is stored as (features, cells) in float64
        data = f['matrix/data'][:]
        
        # cell barcodes
        barcodes = f['matrix/barcodes'][:]
        barcodes = [b.decode('utf-8') for b in barcodes]
        
        # features (genes or peaks)
        features = f['matrix/features'][:]
        features = [f.decode('utf-8') for f in features]
        
    print(f"Original data shape: {data.shape} (features, cells)")
    
    # Transpose data to (cells, features)
    data = data.T
    print(f"Transposed data shape: {data.shape} (cells, features)")
    
    # Convert to sparse CSR matrix to save memory/disk
    sparse_data = sparse.csr_matrix(data)
    
    # Create AnnData object
    adata = ad.AnnData(
        X=sparse_data,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=features)
    )
    
    if cty_df is not None:
        # Check alignment
        if len(cty_df) == len(barcodes):
            # assign cell types (assuming 'x' column or first column)
            label_col = cty_df.columns[0]
            adata.obs['cell_type'] = cty_df[label_col].values
            print(f"Assigned 'cell_type' from '{label_col}'.")
            # Create a mock batch column for the dataset
            adata.obs['batch'] = 'batch1'
        else:
            print(f"Warning: cell type DataFrame length ({len(cty_df)}) does not match number of cells ({len(barcodes)}).")
            
    print(adata)
    return adata

def main():
    base_dir = Path("datasets/raw_datasets/scMultiBench/D15")
    out_dir = Path("datasets/h5ad/D15/RNA+ATAC")
    
    rna_h5 = base_dir / "rna.h5"
    atac_h5 = base_dir / "atac.h5"
    cty_csv = base_dir / "cty.csv"
    
    print("Checking input files...")
    if not (rna_h5.exists() and atac_h5.exists() and cty_csv.exists()):
        print("Required files are missing in", base_dir)
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading labels from", cty_csv)
    cty_df = pd.read_csv(cty_csv)
    print(f"Labels shape: {cty_df.shape}")
    
    # 1. Process RNA
    print("\n--- Processing RNA ---")
    rna_adata = convert_h5_to_adata(rna_h5, cty_df=cty_df)
    rna_out_path = out_dir / "D15-RNA-counts.h5ad"
    print(f"Saving to {rna_out_path}")
    rna_adata.write_h5ad(rna_out_path)
    
    # 2. Process ATAC
    print("\n--- Processing ATAC ---")
    atac_adata = convert_h5_to_adata(atac_h5, cty_df=cty_df)
    atac_out_path = out_dir / "D15-ATAC-peaks.h5ad"
    print(f"Saving to {atac_out_path}")
    atac_adata.write_h5ad(atac_out_path)
    
    print("\n✅ Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
