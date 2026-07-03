"""
Preprocess scMultiBench tri-modal datasets (D22, D59).

D22: Single-batch, RNA + ATAC + ADT
D59: Cross-batch (2 batches), RNA + ATAC + ADT  (ATAC files named peak*.h5)
"""
import os
import glob
import re
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from pathlib import Path


def read_single_h5(h5_path):
    """Read a single custom scMultiBench h5 file -> (data_T, barcodes, features)."""
    with h5py.File(h5_path, 'r') as f:
        data = f['matrix/data'][:]
        barcodes = [b.decode('utf-8') for b in f['matrix/barcodes'][:]]
        features = [feat.decode('utf-8') for feat in f['matrix/features'][:]]
    return data.T, barcodes, features


def convert_single(h5_path, cty_df=None, batch_label="batch1"):
    """Convert a single h5 file to AnnData."""
    data, barcodes, features = read_single_h5(h5_path)
    print(f"  Shape: {data.shape} (cells, features)")

    adata = ad.AnnData(
        X=sparse.csr_matrix(data),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=features),
    )
    if cty_df is not None and len(cty_df) == len(barcodes):
        label_col = cty_df.columns[0]
        adata.obs['cell_type'] = cty_df[label_col].values
        adata.obs['batch'] = batch_label
        print(f"  cell_type: {adata.obs['cell_type'].nunique()} types, batch: {batch_label}")
    return adata


def merge_batches(base_dir, prefix, cty_prefix="cty"):
    """Merge numbered batch files for a modality -> single AnnData with batch labels."""
    pattern = str(base_dir / f"{prefix}*.h5")
    h5_files = sorted(glob.glob(pattern))
    if not h5_files:
        print(f"  ❌ No files for pattern: {pattern}")
        return None

    def extract_num(p):
        m = re.search(rf'{prefix}(\d+)\.h5$', os.path.basename(p))
        return int(m.group(1)) if m else 0
    h5_files.sort(key=extract_num)

    all_data, all_barcodes, all_batches, all_cell_types = [], [], [], []
    shared_features = None

    for h5_path in h5_files:
        batch_num = extract_num(h5_path)
        batch_label = f"batch{batch_num}"
        data, barcodes, features = read_single_h5(h5_path)
        n_cells = data.shape[0]
        print(f"  Batch {batch_num}: {n_cells} cells, {len(features)} features")

        if shared_features is None:
            shared_features = features

        all_data.append(data)
        all_barcodes.extend([f"{batch_label}_{bc}" for bc in barcodes])
        all_batches.extend([batch_label] * n_cells)

        cty_path = base_dir / f"{cty_prefix}{batch_num}.csv"
        if cty_path.exists():
            cty_df = pd.read_csv(cty_path)
            cell_types = cty_df.iloc[:, 0].values.tolist()
            all_cell_types.extend(cell_types if len(cell_types) == n_cells else ["unknown"] * n_cells)
        else:
            all_cell_types.extend(["unknown"] * n_cells)

    merged = np.concatenate(all_data, axis=0)
    print(f"  Merged: {merged.shape}")

    adata = ad.AnnData(
        X=sparse.csr_matrix(merged),
        obs=pd.DataFrame(index=all_barcodes),
        var=pd.DataFrame(index=shared_features),
    )
    adata.obs['cell_type'] = all_cell_types
    adata.obs['batch'] = all_batches
    print(f"  Total: {adata.n_obs} cells, {adata.n_vars} features, "
          f"{adata.obs['cell_type'].nunique()} types, {adata.obs['batch'].nunique()} batches")
    return adata


# ─────────────────── D22: single-batch tri-modal ───────────────────
def process_d22():
    print("\n" + "=" * 60)
    print("Processing D22 (single-batch, RNA + ATAC + ADT)")
    print("=" * 60)

    base = Path("datasets/raw_datasets/scMultiBench/D22")
    out  = Path("datasets/h5ad/D22/RNA+ADT+ATAC")
    out.mkdir(parents=True, exist_ok=True)

    cty_df = pd.read_csv(base / "cty.csv")
    print(f"Labels: {cty_df.shape}")

    for name, src_file, out_name in [
        ("RNA",  "rna.h5",  "D22-RNA-counts.h5ad"),
        ("ATAC", "atac.h5", "D22-ATAC-peaks.h5ad"),
        ("ADT",  "adt.h5",  "D22-ADT-counts.h5ad"),
    ]:
        print(f"\n--- {name} ---")
        adata = convert_single(base / src_file, cty_df)
        out_path = out / out_name
        adata.write_h5ad(out_path)
        print(f"  ✅ Saved to {out_path}")


# ─────────────────── D59: cross-batch tri-modal ───────────────────
def process_d59():
    print("\n" + "=" * 60)
    print("Processing D59 (cross-batch 2 batches, RNA + ATAC + ADT)")
    print("=" * 60)

    base = Path("datasets/raw_datasets/scMultiBench/D59")
    out  = Path("datasets/h5ad/D59/RNA+ADT+ATAC")
    out.mkdir(parents=True, exist_ok=True)

    for name, prefix, out_name in [
        ("RNA",  "rna",  "D59-RNA-counts.h5ad"),
        ("ATAC", "peak", "D59-ATAC-peaks.h5ad"),   # Note: files named peak*.h5
        ("ADT",  "adt",  "D59-ADT-counts.h5ad"),
    ]:
        print(f"\n--- Merging {name} batches (prefix={prefix}) ---")
        adata = merge_batches(base, prefix)
        if adata is not None:
            out_path = out / out_name
            adata.write_h5ad(out_path)
            print(f"  ✅ Saved to {out_path}")


def main():
    process_d22()
    process_d59()
    print("\n✅ All tri-modal datasets preprocessed successfully!")


if __name__ == "__main__":
    main()
