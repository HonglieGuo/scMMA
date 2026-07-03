
import os
import anndata as ad
import muon as mu
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("datasets/h5ad/BMMC-p10")
    output_path = Path("datasets/bmmc_multiome.h5mu")
    
    print(f"Checking data in {data_dir}...")
    
    rna_path = data_dir / "BMMC-multiome-p10-RNA-counts.h5ad"
    atac_path = data_dir / "BMMC-multiome-p10-ATAC-peaks.h5ad"
    meta_path = data_dir / "metadata.csv"
    
    if not rna_path.exists() or not atac_path.exists():
        print("❌ Missing input files in datasets/h5ad/BMMC-p10/")
        return

    print("Loading RNA...")
    rna = ad.read_h5ad(rna_path)
    print("Loading ATAC...")
    atac = ad.read_h5ad(atac_path)
    
    # Ensure observation names match (intersection)
    common_obs = rna.obs_names.intersection(atac.obs_names)
    print(f"Common cells: {len(common_obs)}")
    
    rna = rna[common_obs].copy()
    atac = atac[common_obs].copy()

    # Fix for "ValueError: '_index' is a reserved name"
    # Ensure index names are safe
    for adata in [rna, atac]:
        if adata.obs.index.name == "_index":
            adata.obs.index.name = None
        if adata.var.index.name == "_index":
            adata.var.index.name = None
            
        # Also clean raw if present (sometimes loaded from h5ad)
        # The error "ValueError: '_index' is a reserved name" often comes from raw.var
        # Since we are building a clean object, dropping raw is often safest if it's just a duplicate of counts
        if adata.raw is not None:
            del adata.raw
    
    # Create MuData
    print("Creating MuData...")
    mdata = mu.MuData({"rna": rna, "atac": atac})
    
    # Load and attach metadata (cell_type, batch info, etc.)
    if meta_path.exists():
        print("Loading metadata...")
        meta = pd.read_csv(meta_path)
        
        if "barcode" in meta.columns:
            print("Setting index to 'barcode' column...")
            meta = meta.set_index("barcode")
        else:
            print("Warning: 'barcode' column not found, using default index.")

        # Filter metadata to common cells
        valid_indices = meta.index.intersection(common_obs)
        print(f"Metadata matches {len(valid_indices)}/{len(common_obs)} cells.")
        
        if len(valid_indices) > 0:
            # Reindex metadata to match common_obs order, fill missing with NaN
            meta = meta.reindex(common_obs)
            
            # Merge metadata into mdata.obs (IMPORTANT: this was commented out before!)
            # Select key columns for evaluation
            
            # Check for cell_type columns (support various naming conventions)
            cell_type_candidates = ['cell_type', 'celltype', 'cell_types', 'label', 'cell_type.l1', 'cell_type.l2']
            cell_type_col = None
            for col in cell_type_candidates:
                if col in meta.columns:
                    cell_type_col = col
                    break
            
            # Check for batch columns
            batch_candidates = ['batch', 'site', 'Site', 'donor', 'Donor', 'sample']
            batch_col = None
            for col in batch_candidates:
                if col in meta.columns:
                    batch_col = col
                    break
            
            # Add columns with standardized names
            if cell_type_col:
                mdata.obs['cell_type'] = meta[cell_type_col].values
                print(f"✅ Added 'cell_type' from '{cell_type_col}': {mdata.obs['cell_type'].nunique()} unique types")
            else:
                print("⚠️ No cell_type column found in metadata!")
                
            if batch_col:
                mdata.obs['batch'] = meta[batch_col].values
                print(f"✅ Added 'batch' from '{batch_col}': {mdata.obs['batch'].nunique()} unique batches")
            else:
                # Try to create batch from donor column if available
                if 'donor' in meta.columns or 'Donor' in meta.columns:
                    donor_col = 'donor' if 'donor' in meta.columns else 'Donor'
                    mdata.obs['batch'] = meta[donor_col].values
                    print(f"✅ Added 'batch' from '{donor_col}': {mdata.obs['batch'].nunique()} unique batches")
                else:
                    print("⚠️ No batch column found in metadata!")
                    
            print(f"✅ Metadata attached! obs columns: {list(mdata.obs.columns)}")
        else:
            print("⚠️ No matching cells found in metadata! Skipping metadata merge.")
    
    print(f"Saving to {output_path}...")
    mdata.write(output_path)
    print("✅ Done! Data is ready for scMMA.")

if __name__ == "__main__":
    main()
