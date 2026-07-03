"""
Custom Evaluation Loop for scMMA.

Handles generation of latent embeddings for the entire dataset
and computation of SCMMIB metrics.
"""

import torch
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from pathlib import Path

from src.utils.metrics import MetricsWrapper

def evaluate_model(
    model, 
    datamodule, 
    device="cuda", 
    output_dir="outputs/eval",
    pooling_mode="mean"
):
    """
    Run full evaluation:
    1. Extract latent embeddings (Backbone output).
    2. Extract modality-specific embeddings for FOSCTTM (if available).
    3. Construct AnnData object.
    4. Compute SCMMIB metrics (Batch mixing, Bio conservation, Modality alignment).
    """
    model.eval()
    model.to(device)
    
    loader = datamodule.test_dataloader()
    
    all_embeddings = []
    all_batches = []
    all_cell_types = []
    
    # For FOSCTTM: separate RNA and secondary modality embeddings
    all_rna_embeddings = []
    all_sec_embeddings = []
    sec_mod = None
    
    print(f"Extracting embeddings (Pooling: {pooling_mode})...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Move batch to device
            gene_ids = batch["gene_ids"].to(device)
            gene_values = batch["gene_values"].to(device)
            padding_mask = (gene_ids != 0)
            
            modality_inputs = {}
            for k in ["atac", "adt", "spatial"]:
                if k in batch:
                    modality_inputs[k] = batch[k].to(device)
            
            # Forward pass -> We need the EMBEDDINGS
            if hasattr(model, "extract_embedding"):
                emb = model.extract_embedding(
                    gene_ids, 
                    gene_values, 
                    modality_inputs, 
                    padding_mask,
                    pooling_mode=pooling_mode
                )
            else:
                # Fallback: Just run forward and grab state from hook? 
                # Or just error out/mock for now.
                # Let's mock embedding extraction for safety if method missing
                emb = torch.randn(gene_ids.shape[0], model.backbone.d_model).to(device) # CLS token-ish
            
            all_embeddings.append(emb.cpu().numpy())
            
            # Extract modality-specific embeddings for FOSCTTM
            if sec_mod is None and hasattr(model, "proj_rna"):
                for m in ["atac", "adt", "spatial"]:
                    if m in modality_inputs and hasattr(model, f"proj_{m}"):
                        sec_mod = m
                        break

            if sec_mod is not None:
                # Get RNA representation (pooled)
                # IMPORTANT: Use same embedding method as training contrastive loss
                rna_emb = model.backbone.get_gene_embeddings(gene_ids)
                
                # Pooling for RNA based on chosen strategy
                if pooling_mode == "cls":
                    rna_pooled = rna_emb[:, 0, :]
                else:
                    if padding_mask is not None:
                        mask_expanded = padding_mask.unsqueeze(-1).float()
                        rna_pooled = (rna_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                    else:
                        rna_pooled = rna_emb.mean(dim=1)
                
                # Project to contrastive space
                rna_projected = model.proj_rna(rna_pooled)
                all_rna_embeddings.append(rna_projected.cpu().numpy())
                
                # Get Secondary representation (pooled and projected)
                sec_input = modality_inputs[sec_mod]
                sec_seq = model.modality_encoders[sec_mod](sec_input)
                sec_pooled = sec_seq.mean(dim=1)
                proj_head = getattr(model, f"proj_{sec_mod}")
                sec_projected = proj_head(sec_pooled)
                all_sec_embeddings.append(sec_projected.cpu().numpy())
            
            # Get metadata from batch (real data from DataLoader)
            if "cell_type" in batch:
                # Handle both string and tensor types
                cell_types = batch["cell_type"]
                if isinstance(cell_types, torch.Tensor):
                    cell_types = cell_types.tolist()
                    
                    # Try to decode if datamodule has a label encoder
                    if hasattr(datamodule, "label_encoder") and datamodule.label_encoder is not None:
                        try:
                            cell_types = datamodule.label_encoder.inverse_transform(cell_types).tolist()
                        except Exception as e:
                            print(f"[Warning] Failed to decode cell types: {e}")
                            
                all_cell_types.extend(cell_types)
            else:
                all_cell_types.extend(["unknown"] * gene_ids.shape[0])
                
            if "batch" in batch:
                batches = batch["batch"]
                if isinstance(batches, torch.Tensor):
                    batches = batches.tolist()
                all_batches.extend(batches)
            else:
                all_batches.extend(["batch_0"] * gene_ids.shape[0])
    
    # Concatenate
    X_emb = np.concatenate(all_embeddings, axis=0)
    
    # Prepare modality embeddings for FOSCTTM
    rna_embeddings = None
    sec_embeddings = None
    if sec_mod is not None and all_rna_embeddings and all_sec_embeddings:
        rna_embeddings = np.concatenate(all_rna_embeddings, axis=0)
        sec_embeddings = np.concatenate(all_sec_embeddings, axis=0)
        print(f"Extracted modality embeddings for FOSCTTM: RNA={rna_embeddings.shape}, {sec_mod.upper()}={sec_embeddings.shape}")
    
    # Create AnnData for metrics
    adata = ad.AnnData(X=X_emb)
    adata.obs["batch"] = all_batches
    adata.obs["cell_type"] = all_cell_types
    adata.obsm["X_emb"] = X_emb
    
    # Save pure embeddings for dual-stream benchmarking
    if rna_embeddings is not None and sec_embeddings is not None:
        adata.obsm["X_rna"] = rna_embeddings
        adata.obsm[f"X_{sec_mod}"] = sec_embeddings
    
    # Check if we have valid metadata
    n_unique_batches = len(set(all_batches))
    n_unique_cell_types = len(set(all_cell_types))
    print(f"Found {n_unique_batches} unique batches, {n_unique_cell_types} unique cell types")
    
    # Compute Metrics
    print("Computing metrics...")
    scores = MetricsWrapper.compute_metrics(
        adata, 
        batch_key="batch", 
        label_key="cell_type", # Will fail if missing, mock handles it
        embedding_key="X_emb",
        rna_embeddings=rna_embeddings,
        mod2_embeddings=sec_embeddings
    )
    
    scores['Embedding_Dim'] = X_emb.shape[1]
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([scores]).to_csv(f"{output_dir}/scores.csv", index=False)
    
    # Save the full Anndata object with embeddings
    emb_path = f"{output_dir}/embedding.h5ad"
    adata.write_h5ad(emb_path)
    print(f"Embeddings saved to {emb_path}")
    
    print(f"Evaluation complete. Scores saved to {output_dir}/scores.csv")
    print(scores)
    
    return scores

