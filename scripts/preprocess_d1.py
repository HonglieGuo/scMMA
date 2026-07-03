"""
Preprocess scMultiBench D1 dataset (RNA + ADT).
Converts raw .h5 and .csv into .h5ad for scMMA.
"""
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from pathlib import Path


def convert_h5_to_adata(h5_path, cty_df=None):
    """Reads the custom h5 format and converts it to an AnnData object."""
    print(f"Reading {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        data = f['matrix/data'][:]
        barcodes = [b.decode('utf-8') for b in f['matrix/barcodes'][:]]
        features = [feat.decode('utf-8') for feat in f['matrix/features'][:]]

    print(f"  Original shape: {data.shape} (features, cells)")
    data = data.T
    print(f"  Transposed shape: {data.shape} (cells, features)")

    sparse_data = sparse.csr_matrix(data)
    adata = ad.AnnData(
        X=sparse_data,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=features),
    )

    if cty_df is not None and len(cty_df) == len(barcodes):
        label_col = cty_df.columns[0]
        adata.obs['cell_type'] = cty_df[label_col].values
        adata.obs['batch'] = 'batch1'
        print(f"  Assigned cell_type from '{label_col}' ({adata.obs['cell_type'].nunique()} types)")

    print(f"  {adata}")
    return adata


def main():
    base_dir = Path("datasets/raw_datasets/scMultiBench/D1")
    out_dir  = Path("datasets/h5ad/D1/RNA+ADT")

    rna_h5  = base_dir / "rna.h5"
    adt_h5  = base_dir / "adt.h5"
    cty_csv = base_dir / "cty.csv"

    for p in (rna_h5, adt_h5, cty_csv):
        if not p.exists():
            print(f"❌ Missing: {p}")
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    cty_df = pd.read_csv(cty_csv)
    print(f"Labels shape: {cty_df.shape}")

    # RNA
    print("\n--- Processing RNA ---")
    rna = convert_h5_to_adata(rna_h5, cty_df)
    rna_out = out_dir / "D1-RNA-counts.h5ad"
    rna.write_h5ad(rna_out)
    print(f"  Saved to {rna_out}")

    # ADT
    print("\n--- Processing ADT ---")
    adt = convert_h5_to_adata(adt_h5, cty_df)
    adt_out = out_dir / "D1-ADT-counts.h5ad"
    adt.write_h5ad(adt_out)
    print(f"  Saved to {adt_out}")

    print("\n✅ D1 preprocessing completed successfully!")


if __name__ == "__main__":
    main()
