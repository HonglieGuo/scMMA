"""
Metrics interface for scMMA.

Computes biological conservation and batch correction metrics using sklearn.
Avoids dependency on scib which has compatibility issues with newer scanpy.
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
import scanpy as sc
import anndata as ad
from scipy.spatial.distance import cdist

# sklearn for clustering metrics
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)
from sklearn.preprocessing import LabelEncoder


def optimize_leiden_resolution(adata, label_col='cell_type'):
    """
    Dynamically search for optimal resolution to maximize NMI (aligned with scib standard).
    """
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.preprocessing import LabelEncoder
    
    if label_col not in adata.obs:
        sc.tl.leiden(adata, resolution=1.0, key_added="leiden", n_iterations=2)
        return 1.0

    print("Searching for optimal Leiden resolution (based on max NMI)...")
    best_nmi = -1
    best_res = 1.0
    best_cluster_labels = None
    
    # Encode true labels
    le = LabelEncoder()
    true_labels = le.fit_transform(adata.obs[label_col].values)
    
    for res in [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        sc.tl.leiden(adata, resolution=res, key_added="leiden_temp", n_iterations=2)
        pred_labels = LabelEncoder().fit_transform(adata.obs["leiden_temp"].values)
        
        nmi_val = normalized_mutual_info_score(true_labels, pred_labels)
        if nmi_val > best_nmi:
            best_nmi = nmi_val
            best_res = res
            best_cluster_labels = adata.obs['leiden_temp'].copy()

    print(f"✅ Optimal resolution found: {best_res} (NMI: {best_nmi:.4f})")
    adata.obs['leiden'] = best_cluster_labels
    
    if "leiden_temp" in adata.obs:
        del adata.obs["leiden_temp"]
        
    return float(best_res)


class MetricsWrapper:
    """Wrapper for biological conservation and batch correction metrics."""
    
    @staticmethod
    def compute_lisi(X: np.ndarray, labels: np.ndarray, k: int = 30) -> float:
        """
        Compute Local Inverse Simpson's Index (LISI).
        
        For cell-type LISI (cLISI): measures purity of neighborhoods by cell type.
        For integration LISI (iLISI): measures mixing of batches in neighborhoods.
        
        Args:
            X: Embedding matrix (N, D)
            labels: Labels for each cell (N,)
            k: Number of neighbors
            
        Returns:
            Average LISI score
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_cells = X.shape[0]
        k = min(k, n_cells - 1)
        
        # Fit KNN
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)[:, 1:]  # Exclude self
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        n_categories = len(le.classes_)
        
        lisi_scores = []
        for i in range(n_cells):
            neighbor_labels = encoded_labels[indices[i]]
            # Compute Simpson's Index: sum of squared proportions
            _, counts = np.unique(neighbor_labels, return_counts=True)
            proportions = counts / k
            simpson = np.sum(proportions ** 2)
            # Inverse Simpson's Index
            lisi = 1.0 / simpson if simpson > 0 else 1.0
            lisi_scores.append(lisi)
        
        return float(np.mean(lisi_scores))
    
    @staticmethod
    def compute_foscttm(
        rna_embeddings: np.ndarray, 
        atac_embeddings: np.ndarray,
        chunk_size: int = 5000
    ) -> float:
        """
        Compute FOSCTTM (Fraction of Samples Closer Than the True Match) using chunking.
        """
        n_cells = rna_embeddings.shape[0]
        foscttm_scores = []
        
        for start_idx in range(0, n_cells, chunk_size):
            end_idx = min(start_idx + chunk_size, n_cells)
            # Compute pairwise distances between chunk of RNA and all ATAC
            chunk_dist = cdist(rna_embeddings[start_idx:end_idx], atac_embeddings, metric='cosine')
            
            for local_i, global_i in enumerate(range(start_idx, end_idx)):
                true_dist = chunk_dist[local_i, global_i]
                closer_count = np.sum(chunk_dist[local_i, :] < true_dist)
                foscttm_scores.append(closer_count / n_cells)
                
        return float(np.mean(foscttm_scores))
    
    @staticmethod
    def compute_topk_accuracy(
        rna_embeddings: np.ndarray, 
        atac_embeddings: np.ndarray,
        k: int = 5,
        chunk_size: int = 5000
    ) -> float:
        """
        Compute Cross-Modal Top-k Accuracy using chunking.
        For each RNA cell, check if its true paired ATAC cell is in the top k nearest neighbors.
        """
        n_cells = rna_embeddings.shape[0]
        hits = 0
        
        for start_idx in range(0, n_cells, chunk_size):
            end_idx = min(start_idx + chunk_size, n_cells)
            chunk_dist = cdist(rna_embeddings[start_idx:end_idx], atac_embeddings, metric='cosine')
            
            if k < n_cells:
                top_k_indices = np.argpartition(chunk_dist, k, axis=1)[:, :k]
            else:
                top_k_indices = np.argsort(chunk_dist, axis=1)[:, :k]
                
            for local_i, global_i in enumerate(range(start_idx, end_idx)):
                if global_i in top_k_indices[local_i]:
                    hits += 1
                    
        return float(hits / n_cells)

    @staticmethod
    def compute_lta(
        rna_embeddings: np.ndarray, 
        atac_embeddings: np.ndarray,
        labels: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Compute Label Transfer Accuracy (LTA).
        Train a KNN on RNA embeddings and predict labels for ATAC embeddings.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        
        # Handle duplicated labels from vertical concatenation
        if len(labels) == 2 * rna_embeddings.shape[0]:
            labels = labels[:rna_embeddings.shape[0]]
            
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(rna_embeddings, labels)
        preds = knn.predict(atac_embeddings)
        
        return float(accuracy_score(labels, preds))
    
    @staticmethod
    def compute_metrics(
        adata: ad.AnnData,
        batch_key: str = "batch",
        label_key: str = "cell_type",
        embedding_key: str = "X_emb",
        rna_embeddings: Optional[np.ndarray] = None,
        mod2_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute integration metrics using sklearn (no scib dependency).
        
        Metrics computed:
        - NMI (Normalized Mutual Information): measures cluster-label agreement
        - ARI (Adjusted Rand Index): measures cluster-label agreement
        - ASW_label (Silhouette Score by cell type): bio conservation
        - cLISI (Cell-type LISI): cell type purity in neighborhoods (higher is better)
        - Batch_ASW (normalized): batch effect removal (higher is better)
        - iLISI (Integration LISI): batch mixing in neighborhoods (higher is better)
        - FOSCTTM: modality alignment accuracy (lower is better, requires paired data)
        
        Args:
            adata: AnnData object containing embeddings.
            batch_key: Column name for batch info.
            label_key: Column name for biological labels (cell type).
            embedding_key: Key in obsm where embeddings are stored.
            rna_embeddings: Optional RNA embeddings for FOSCTTM (N, D).
            mod2_embeddings: Optional Secondary modality embeddings for FOSCTTM (N, D).
            
        Returns:
            Dictionary of metric scores.
        """
        metrics = {}
        
        # Get embedding matrix
        if embedding_key in adata.obsm:
            X_emb = adata.obsm[embedding_key]
        else:
            X_emb = adata.X
            
        # Validate data availability
        has_valid_labels = (label_key in adata.obs.columns and 
                          len(adata.obs[label_key].unique()) > 1)
        
        if batch_key not in adata.obs or len(adata.obs[batch_key].unique()) <= 1:
            print(f"⚠️ Single batch detected. Skipping batch correction metrics (iLISI, ASW_batch).")
            has_valid_batches = False
        else:
            has_valid_batches = True
        
        if not has_valid_labels:
            print(f"⚠️ Warning: '{label_key}' missing or has ≤1 unique value. Bio metrics skipped.")
            
        # Build neighbors graph and run Leiden clustering for NMI/ARI
        if "neighbors" not in adata.uns:
            print("Computing neighbors graph...")
            sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
            
        # Run Leiden clustering
        print("Running Leiden clustering with adaptive resolution...")
        best_res = optimize_leiden_resolution(adata, label_col=label_key)
        metrics['leiden_res'] = best_res
        cluster_labels = adata.obs["leiden"].values
        
        # 1. Biological Conservation Metrics (requires valid cell_type labels)
        if has_valid_labels:
            try:
                # Encode labels to integers
                le = LabelEncoder()
                true_labels = le.fit_transform(adata.obs[label_key].values)
                pred_labels = LabelEncoder().fit_transform(cluster_labels)
                
                # NMI: Normalized Mutual Information
                nmi = normalized_mutual_info_score(true_labels, pred_labels)
                metrics['nmi'] = float(nmi)
                
                # ARI: Adjusted Rand Index
                ari = adjusted_rand_score(true_labels, pred_labels)
                metrics['ari'] = float(ari)
                
                # ASW_label: Silhouette score by cell type
                if len(np.unique(true_labels)) > 1:
                    asw_label = silhouette_score(X_emb, true_labels)
                    metrics['asw_label'] = float(asw_label)
                else:
                    metrics['asw_label'] = None
                
                # cLISI: Cell-type LISI (purity, higher is better)
                # We normalize: (max_LISI - LISI) / (max_LISI - 1)
                # where max_LISI = number of cell types
                raw_clisi = MetricsWrapper.compute_lisi(X_emb, adata.obs[label_key].values)
                n_cell_types = len(np.unique(true_labels))
                # Normalize so that 1 = perfect purity (LISI=1), 0 = max mixing (LISI=n_types)
                if n_cell_types > 1:
                    clisi_normalized = (n_cell_types - raw_clisi) / (n_cell_types - 1)
                    metrics['clisi'] = float(np.clip(clisi_normalized, 0, 1))
                else:
                    metrics['clisi'] = 1.0
                    
                asw_res = metrics['asw_label']
                asw_str = f"{asw_res:.4f}" if asw_res is not None else "N/A"
                print(f"✅ Bio conservation: NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, "
                      f"ASW_label={asw_str}, cLISI={metrics['clisi']:.4f}")
                
            except Exception as e:
                print(f"❌ Error computing bio conservation metrics: {e}")
                metrics['nmi'] = None
                metrics['ari'] = None
                metrics['asw_label'] = None
                metrics['clisi'] = None
        else:
            metrics['nmi'] = None
            metrics['ari'] = None
            metrics['asw_label'] = None
            metrics['clisi'] = None
            
        # 2. Batch Correction Metrics (requires valid batch info)
        if has_valid_batches:
            try:
                # Encode batch labels
                batch_labels = LabelEncoder().fit_transform(adata.obs[batch_key].values)
                
                # ASW_batch (raw): Silhouette score by batch
                asw_batch_raw = silhouette_score(X_emb, batch_labels)
                metrics['asw_batch_raw'] = float(asw_batch_raw)
                
                # Batch_ASW (normalized): Higher is better
                # Formula: (1 - asw_batch_raw) / 2, mapping [-1, 1] to [1, 0] then to [0, 1]
                # When asw_batch_raw = -1 (perfect mixing), batch_asw = 1
                # When asw_batch_raw = 1 (no mixing), batch_asw = 0
                batch_asw = (1 - asw_batch_raw) / 2
                metrics['batch_asw'] = float(batch_asw)
                
                # iLISI: Integration LISI (mixing, higher is better)
                # Normalize so that 1 = perfect mixing, 0 = no mixing
                raw_ilisi = MetricsWrapper.compute_lisi(X_emb, adata.obs[batch_key].values)
                n_batches = len(np.unique(batch_labels))
                if n_batches > 1:
                    ilisi_normalized = (raw_ilisi - 1) / (n_batches - 1)
                    metrics['ilisi'] = float(np.clip(ilisi_normalized, 0, 1))
                else:
                    metrics['ilisi'] = 1.0
                
                print(f"✅ Batch correction: Batch_ASW={metrics['batch_asw']:.4f}, "
                      f"iLISI={metrics['ilisi']:.4f}, ASW_batch_raw={metrics['asw_batch_raw']:.4f}")
                
            except Exception as e:
                print(f"❌ Error computing batch correction metrics: {e}")
                metrics['asw_batch_raw'] = None
                metrics['batch_asw'] = None
                metrics['ilisi'] = None
        else:
            metrics['asw_batch_raw'] = None
            metrics['batch_asw'] = None
            metrics['ilisi'] = None
        
        # 3. Modality Alignment (requires paired modality embeddings)
        if rna_embeddings is not None and mod2_embeddings is not None:
            try:
                foscttm = MetricsWrapper.compute_foscttm(rna_embeddings, mod2_embeddings)
                metrics['foscttm'] = float(foscttm)
                
                match_at_5 = MetricsWrapper.compute_topk_accuracy(rna_embeddings, mod2_embeddings, k=5)
                metrics['match_at_5'] = float(match_at_5)
                
                if has_valid_labels:
                    lta = MetricsWrapper.compute_lta(rna_embeddings, mod2_embeddings, adata.obs[label_key].values)
                    metrics['lta'] = float(lta)
                else:
                    metrics['lta'] = None
                    
                lta_str = f", LTA={metrics['lta']:.4f}" if metrics['lta'] is not None else ""
                print(f"✅ Modality alignment: FOSCTTM={metrics['foscttm']:.4f}, Match@5={metrics['match_at_5']:.4f}{lta_str}")
            except Exception as e:
                print(f"❌ Error computing alignment metrics: {e}")
                metrics['foscttm'] = None
                metrics['match_at_5'] = None
                metrics['lta'] = None
        else:
            metrics['foscttm'] = None
            metrics['match_at_5'] = None
            metrics['lta'] = None
            
        # 4. Overall integration score (weighted combination)
        # Bio: NMI, ARI, ASW_label, cLISI (all higher is better)
        # Batch: Batch_ASW, iLISI (all higher is better)
        valid_bio = [v for v in [metrics.get('nmi'), metrics.get('ari'), 
                                  metrics.get('asw_label'), metrics.get('clisi')] if v is not None]
        valid_batch = [v for v in [metrics.get('batch_asw'), metrics.get('ilisi')] if v is not None]
        
        if valid_bio and valid_batch:
            bio_score = np.mean(valid_bio)
            batch_score = np.mean(valid_batch)
            # Overall score: 0.6 * bio + 0.4 * batch (common weighting)
            metrics['overall_score'] = float(0.6 * bio_score + 0.4 * batch_score)
            print(f"✅ Overall integration score: {metrics['overall_score']:.4f}")
        elif valid_bio:
            metrics['overall_score'] = float(np.mean(valid_bio))
        else:
            metrics['overall_score'] = None
            
        return metrics

    @staticmethod
    def reconstruction_error(
        pred: Tensor, 
        target: Tensor, 
        mask: Optional[Tensor] = None
    ) -> float:
        """Compute MSE reconstruction error."""
        if mask is not None:
            mse = torch.nn.functional.mse_loss(pred[mask], target[mask])
        else:
            mse = torch.nn.functional.mse_loss(pred, target)
        return mse.item()

