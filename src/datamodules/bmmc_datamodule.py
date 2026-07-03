"""
BMMC (Bone Marrow Mononuclear Cells) DataModule.

Loads the BMMC Multiome dataset for training and evaluation.
Compatible with SCMMIB benchmarks.
"""

from pathlib import Path
from typing import Optional, Union, List
import hashlib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder # New import

import anndata as ad
import lightning.pytorch as pl
import muon as mu
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from scipy.sparse import issparse
from tqdm import tqdm

from .components.preprocessor import ScGPTTokenizer, GeneformerTokenizer, binning

log = logging.getLogger(__name__)


def sparse_collate_fn(batch):
    """
    Custom collate function that handles sparse ATAC tensors.
    
    Builds a batched 2D sparse tensor (Batch, N_peaks) from individual 
    1D sparse tensors (N_peaks,). This enables efficient pin_memory and
    GPU transfer while preserving sparsity.
    """
    elem = batch[0]
    result = {}
    
    for key in elem:
        if key == "atac" and elem[key].is_sparse:
            # Build batched 2D Sparse Tensor from 1D sparse samples
            batch_size = len(batch)
            n_peaks = batch[0][key].shape[0]
            
            all_indices = []
            all_values = []
            
            for i, item in enumerate(batch):
                sparse_t = item[key]
                # Get current sample's indices (1, nnz) and values
                idx = sparse_t.indices()  # (1, nnz)
                val = sparse_t.values()
                
                # Add batch dimension index: (1, nnz) filled with sample index i
                batch_idx = torch.full_like(idx, i)
                
                # Stack to (2, nnz): [batch_row, col]
                stacked_indices = torch.cat([batch_idx, idx], dim=0)
                
                all_indices.append(stacked_indices)
                all_values.append(val)
            
            # Concatenate all samples: indices (2, total_nnz), values (total_nnz,)
            final_indices = torch.cat(all_indices, dim=1)
            final_values = torch.cat(all_values)
            
            # Build batched 2D sparse tensor (Batch, N_peaks)
            result[key] = torch.sparse_coo_tensor(
                final_indices,
                final_values,
                size=(batch_size, n_peaks)
            ).coalesce()
        else:
            # Use default collate for all other keys (RNA tokens, masks, etc.)
            result[key] = default_collate([item[key] for item in batch])
    
    return result


class BMMCDataset(Dataset):
    """
    PyTorch Dataset for BMMC Multiome data.
    
    Supports precomputed tokenization cache for faster data loading.
    """
    
    def __init__(
        self,
        mdata: mu.MuData,
        tokenizer: Union[ScGPTTokenizer, GeneformerTokenizer],
        max_seq_len: int = 1200,
        split: str = "train",
        preprocessor_type: str = "scgpt",
        cache_dir: Optional[Path] = None,
        cell_indices: Optional[np.ndarray] = None,  # Original indices for cache lookup
    ):
        self.mdata = mdata
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.preprocessor_type = preprocessor_type
        self.cache_dir = cache_dir
        self.cell_indices = cell_indices  # Map local idx to global cache idx
        
        # Pre-fetch data to memory if possible
        self.rna = mdata.mod['rna']
        self.atac = mdata.mod['atac']
        
        # Pre-densify ATAC data to avoid per-sample toarray() overhead
        # This converts sparse matrix to dense numpy array once at init
        log.info(f"Pre-densifying ATAC data for {split} split ({self.atac.shape[0]} cells, {self.atac.shape[1]} peaks)...")
        if issparse(self.atac.X):
            self.atac_dense = self.atac.X.toarray().astype(np.float32)
        elif hasattr(self.atac.X, "toarray"):
            self.atac_dense = self.atac.X.toarray().astype(np.float32)
        else:
            self.atac_dense = np.asarray(self.atac.X, dtype=np.float32)
        log.info(f"ATAC densification complete: {self.atac_dense.shape}, {self.atac_dense.nbytes / 1e6:.1f} MB")
        
        # Gene names for tokenization
        self.gene_names = self.rna.var_names.tolist()
        
        # Get cell metadata for evaluation (cell_type, batch)
        # Try common column names in obs (including variations like cell_type.l1)
        self.cell_types = self._get_obs_column(['cell_type', 'celltype', 'cell_types', 'label', 'labels', 'cell_type.l1', 'cell_type.l2'])
        self.batches = self._get_obs_column(['batch', 'Batch', 'donor', 'Donor', 'sample', 'site', 'Site'])
        
        # Tokenization cache (loaded from disk or computed)
        self.token_cache: Optional[torch.Tensor] = None
        self.mask_cache: Optional[torch.Tensor] = None
        
        # Label Encoder (Shared)
        self.label_encoder = None
        if hasattr(mdata, "uns") and "cell_type_le" in mdata.uns:
             self.label_encoder = mdata.uns["cell_type_le"]
        
        if self.preprocessor_type == "scgpt":
            self.tokenized_genes = self.tokenizer.tokenize(self.gene_names)
        elif self.preprocessor_type == "geneformer" and cache_dir is not None:
            # Try to load from cache
            self._load_or_compute_geneformer_cache()
    
    def _get_obs_column(self, possible_names: List[str]) -> Optional[np.ndarray]:
        """Try to get a column from obs by trying multiple possible names."""
        obs = self.mdata.obs
        for name in possible_names:
            if name in obs.columns:
                return obs[name].values
        return None
    
    def _load_or_compute_geneformer_cache(self):
        """Load tokenization cache from disk, or compute and save if not found."""
        if self.cache_dir is None:
            return
        
        cache_file = self.cache_dir / f"geneformer_tokens_{self.max_seq_len}.pt"
        
        if cache_file.exists():
            # Load full cache
            log.info(f"Loading tokenization cache from {cache_file}")
            cache_data = torch.load(cache_file, weights_only=True)
            full_tokens = cache_data['tokens']
            full_masks = cache_data['masks']
            
            # Select only cells in this split using cell_indices
            if self.cell_indices is not None:
                self.token_cache = full_tokens[self.cell_indices]
                self.mask_cache = full_masks[self.cell_indices]
                log.info(f"Selected {len(self.token_cache)} cached tokenizations for {self.split} split")
            else:
                self.token_cache = full_tokens
                self.mask_cache = full_masks
                log.info(f"Loaded {len(self.token_cache)} cached tokenizations")
        else:
            # Cache should have been precomputed by BMMCDataModule
            log.warning(f"Cache file not found: {cache_file}. Falling back to on-the-fly tokenization.")
    
    def _compute_and_save_cache(self, cache_file: Path):
        """Compute tokenization for all samples and save to disk."""
        n_samples = len(self)
        tokens_list = []
        masks_list = []
        
        for idx in tqdm(range(n_samples), desc=f"Tokenizing {self.split}"):
            # Get RNA expression
            rna_expr = self.rna.X[idx]
            if issparse(rna_expr):
                rna_expr = rna_expr.toarray().flatten()
            elif hasattr(rna_expr, "toarray"):
                rna_expr = rna_expr.toarray().flatten()
            
            # Calculate total counts for CPM normalization
            n_counts = float(np.sum(rna_expr))
            
            # Tokenize
            gene_ids = self.tokenizer.tokenize_rank_value(
                self.gene_names,
                rna_expr,
                n_counts=n_counts,
                max_len=self.max_seq_len
            )
            
            # Create attention mask
            pad_id = self.tokenizer.pad_token_id
            attention_mask = (gene_ids != pad_id).long()
            
            tokens_list.append(gene_ids)
            masks_list.append(attention_mask)
        
        # Stack into tensors
        self.token_cache = torch.stack(tokens_list)
        self.mask_cache = torch.stack(masks_list)
        
        # Save to disk
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'tokens': self.token_cache,
            'masks': self.mask_cache,
        }, cache_file)
        log.info(f"Saved tokenization cache to {cache_file}")

        
    def __len__(self):
        return self.mdata.shape[0]
    
    def __getitem__(self, idx: int):
        # 1. Prepare RNA inputs
        if self.preprocessor_type == "geneformer":
            # Use cached tokenization if available (FAST PATH)
            if self.token_cache is not None:
                gene_ids = self.token_cache[idx]
                attention_mask = self.mask_cache[idx]
                gene_values = torch.ones_like(gene_ids, dtype=torch.float32)
            else:
                # Fallback to on-the-fly tokenization (SLOW PATH)
                rna_expr = self.rna.X[idx]
                if issparse(rna_expr):
                    rna_expr = rna_expr.toarray().flatten()
                elif hasattr(rna_expr, "toarray"):
                    rna_expr = rna_expr.toarray().flatten()
                
                n_counts = float(np.sum(rna_expr))
                
                gene_ids = self.tokenizer.tokenize_rank_value(
                    self.gene_names, 
                    rna_expr,
                    n_counts=n_counts,
                    max_len=self.max_seq_len
                )
                pad_id = self.tokenizer.pad_token_id
                attention_mask = (gene_ids != pad_id).long()
                gene_values = torch.ones_like(gene_ids, dtype=torch.float32)

        else:
            # scGPT strategy
            # Get dense vector
            rna_expr = self.rna.X[idx]
            if issparse(rna_expr):
                rna_expr = rna_expr.toarray().flatten()
            elif hasattr(rna_expr, "toarray"):
                rna_expr = rna_expr.toarray().flatten()
            
            # Slice to max_len matching tokenized list size
            rna_expr = rna_expr[:len(self.tokenized_genes)]
            
            # Binning (on the fly)
            gene_values = binning(rna_expr)
            
            # Slice to max_len (simple truncation for fixed list)
            gene_ids = self.tokenized_genes[:self.max_seq_len]
            gene_values = gene_values[:self.max_seq_len]
            
            # Padding
            pad_len = self.max_seq_len - len(gene_ids)
            if pad_len > 0:
                pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                pad_vals = torch.zeros((pad_len,), dtype=torch.long)
                
                gene_ids = torch.cat([gene_ids, pad_ids])
                gene_values = torch.cat([gene_values, pad_vals])
            
            gene_ids = gene_ids.long()
            gene_values = gene_values.long()
            attention_mask = (gene_ids != self.tokenizer.pad_token_id).long()

        # 2. Prepare ATAC inputs - Use pre-densified array (FAST PATH)
        atac_peaks = torch.from_numpy(self.atac_dense[idx])
        
        # Build result dict
        result = {
            "gene_ids": gene_ids.long(),
            "gene_values": gene_values.long(), # Ensure long for embeddings
            "attention_mask": attention_mask,
            "atac": atac_peaks,
            # Placeholder for future modalities
            # "adt": ...,
            # "spatial": ...
        }
        
        # Add metadata for evaluation
        if self.cell_types is not None:
            cell_type_str = self.cell_types[idx]
            result["cell_type_str"] = cell_type_str
            if self.label_encoder is not None:
                 result["cell_type"] = torch.tensor(self.label_encoder.transform([cell_type_str])[0], dtype=torch.long)
            else:
                 result["cell_type"] = -1 # Indicator of no label encoder
        else:
            result["cell_type_str"] = "unknown"
            result["cell_type"] = -1
            
        if self.batches is not None:
            result["batch"] = self.batches[idx]
        else:
            result["batch"] = "batch_0"
            
        return result


class BMMCDataModule(pl.LightningDataModule):
    """
    LightningDataModule for BMMC dataset.
    """
    
    def __init__(
        self,
        data_dir: str = "datasets/",
        vocab_path: str = "configs/vocab.json",
        gene_id_map_path: Optional[str] = None,  # Gene Symbol -> Ensembl ID mapping
        gene_median_path: Optional[str] = None,  # Gene -> Median expression for normalization
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,  # Accelerate CPU->GPU transfer
        persistent_workers: bool = True,  # Keep workers alive between epochs
        max_seq_len: int = 1200,
        preprocessor_type: str = "scgpt"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.vocab_path = vocab_path
        self.gene_id_map_path = gene_id_map_path
        self.gene_median_path = gene_median_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0  # Only if workers exist
        self.max_seq_len = max_seq_len
        self.preprocessor_type = preprocessor_type
        
        self.mdata: Optional[mu.MuData] = None
        self.train_dataset: Optional[BMMCDataset] = None
        self.val_dataset: Optional[BMMCDataset] = None
        self.test_dataset: Optional[BMMCDataset] = None
        
        self.label_encoder = LabelEncoder()
        self.num_classes = 0
        
    def prepare_data(self):
        """Download data if needed."""
        # TODO: Implement download logic from SCMMIB or other source
        pass
        
    
    def setup(self, stage: Optional[str] = None):
        """Load data and split."""
        if self.mdata is None:
            # Placeholder loading attempt. 
            # In practice: handle .h5mu or .h5ad files
            fpath = self.data_dir / "bmmc_multiome.h5mu"
            if fpath.exists():
                self.mdata = mu.read(str(fpath))
            else:
                print(f"Dataset {fpath} not found. Creating mock data.")
                # Create mock MuData for valid instantiation during scaffold verify
                rna = ad.AnnData(np.random.randn(100, 200)) # 100 cells, 200 genes
                rna.var_names = [f"Gene_{i}" for i in range(200)]
                
                # Mock ATAC with chromosome info
                n_peaks = 500
                atac = ad.AnnData(np.random.randn(100, n_peaks))
                # Create fake chr/start
                chrs = ['chr1'] * 200 + ['chr2'] * 200 + ['chrX'] * 100
                starts = np.arange(n_peaks)
                atac.var = pd.DataFrame({'chr': chrs, 'start': starts}, index=[f"Peak_{i}" for i in range(n_peaks)])
                
                self.mdata = mu.MuData({'rna': rna, 'atac': atac})

        # --- ATAC Processing: Sort and Index for ChromosomalEncoder ---
        if 'atac' in self.mdata.mod:
            atac = self.mdata.mod['atac']
            
            # Define parsing functions (always available)
            def parse_chr(name):
                name = str(name).replace(':', '-')
                parts = name.split('-')
                return parts[0]
                
            def parse_start(name):
                name = str(name).replace(':', '-')
                parts = name.split('-')
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
                return 0
            
            # 1. Standardize Chromosome Names (if needed) & Check Columns
            if 'chr' not in atac.var.columns:
                # Fallback: Try to parse from index (e.g., "chr1:1000-2000" or "chr1-1000-2000")
                print("Warning: 'chr' column not found in ATAC var. Attempting to parse index.")
                try:
                    atac.var['chr'] = atac.var_names.map(parse_chr)
                    atac.var['start'] = atac.var_names.map(parse_start)
                        
                except Exception as e:
                    print(f"Failed to parse ATAC var index: {e}. Using dummy chr1.")
                    atac.var['chr'] = 'chr1'
                    atac.var['start'] = np.arange(atac.shape[1])
            
            # 2. Filter Whitelist (chr1-22, X, Y)
            valid_chroms = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
            # Filter vars
            # Note: We need to filter the MuData object to keep observations consistent if needed
            # But usually we just filter features.
            
            # Identify valid peaks
            is_valid = atac.var['chr'].isin(valid_chroms)
            original_n_peaks = atac.shape[1]
            filt_atac = None
            
            if not is_valid.all():
                print(f"Filtering non-standard chromosomes. {original_n_peaks} -> {is_valid.sum()} peaks.")
                filt_atac = atac[:, is_valid].copy()
            else:
                filt_atac = atac.copy() # *Always* copy to ensure fresh object

            # Re-parse chr/start on FILTERED data (essential for correct sorting)
            filt_atac.var['chr'] = filt_atac.var_names.map(parse_chr)
            filt_atac.var['start'] = filt_atac.var_names.map(parse_start)
            print(f"Re-parsed chr/start on filtered ATAC ({filt_atac.shape[1]} peaks).")

            # 3. Sort by Chromosome and Start Position
            sorted_idx = filt_atac.var.sort_values(['chr', 'start']).index
            
            if not filt_atac.var_names.equals(sorted_idx):
                print("Sorting ATAC peaks by genomic position...")
                filt_atac = filt_atac[:, sorted_idx].copy()
            
            # CRITICAL: Force X to dense array to break HDF5 lazy loading
            if hasattr(filt_atac.X, 'toarray'):
                print("Converting sparse ATAC X to dense...")
                filt_atac.X = filt_atac.X.toarray()
            elif hasattr(filt_atac, 'file') or 'h5' in str(type(filt_atac.X)).lower():
                print("Converting HDF5-backed ATAC X to in-memory array...")
                filt_atac.X = np.array(filt_atac.X)
            
            # Update MuData: ALWAYS reconstruct to guarantee consistency
            # CRITICAL: Preserve original obs metadata (cell_type, batch, etc.) for Benchmark!
            print("Reconstructing MuData with processed ATAC...")
            original_obs = self.mdata.obs.copy()  # Save BEFORE reconstruction
            
            rna = self.mdata.mod['rna'].copy() # Also copy RNA for safety
            self.mdata = mu.MuData({'rna': rna, 'atac': filt_atac})
            self.mdata.update()
            
            # Restore original obs metadata (cell_type, batch, donor, etc.)
            for col in original_obs.columns:
                if col not in self.mdata.obs.columns:
                    self.mdata.obs[col] = original_obs[col].values
            print(f"Restored {len(original_obs.columns)} obs columns from original MuData.")
                
            # Refresh reference
            atac = self.mdata.mod['atac']

                
            # 4. Generate Chromosome Indices (Slice or List)
            # We need to map 'chr1' -> [0, 8000], 'chr2' -> [8000, 15000]
            self.chrom_indices = {}
            current_idx = 0
            
            # Group by chr. Since it's sorted, we can just iterate or use groupby
            # Ensure we maintain the sorted order of keys
            grouped = atac.var.groupby('chr', sort=False) 
            
            for chrom, group in grouped:
                n_peaks_in_chrom = len(group)
                start = current_idx
                end = current_idx + n_peaks_in_chrom
                self.chrom_indices[chrom] = [start, end] # List format for JSON serialization if needed
                current_idx = end
                
            print(f"Generated indices for {len(self.chrom_indices)} chromosomes.")
            
            # Verify Integrity
            assert current_idx == atac.shape[1], "Chromosome indices do not cover all peaks!"

        # Continue with tokenizer initialization and dataset creation
        self._setup_with_cache(stage)

    def _precompute_full_cache(self, tokenizer, cache_dir: Path):
        """Precompute tokenization cache for the entire dataset."""
        cache_file = cache_dir / f"geneformer_tokens_{self.max_seq_len}.pt"
        
        if cache_file.exists():
            log.info(f"Found existing tokenization cache: {cache_file}")
            return
        
        log.info(f"Precomputing tokenization cache for {self.mdata.shape[0]} cells...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        rna = self.mdata.mod['rna']
        gene_names = rna.var_names.tolist()
        n_samples = rna.shape[0]
        
        tokens_list = []
        masks_list = []
        
        for idx in tqdm(range(n_samples), desc="Tokenizing all cells"):
            # Get RNA expression
            rna_expr = rna.X[idx]
            if issparse(rna_expr):
                rna_expr = rna_expr.toarray().flatten()
            elif hasattr(rna_expr, "toarray"):
                rna_expr = rna_expr.toarray().flatten()
            
            n_counts = float(np.sum(rna_expr))
            
            gene_ids = tokenizer.tokenize_rank_value(
                gene_names,
                rna_expr,
                n_counts=n_counts,
                max_len=self.max_seq_len
            )
            
            pad_id = tokenizer.pad_token_id
            attention_mask = (gene_ids != pad_id).long()
            
            tokens_list.append(gene_ids)
            masks_list.append(attention_mask)
        
        # Save full cache
        full_cache = {
            'tokens': torch.stack(tokens_list),
            'masks': torch.stack(masks_list),
        }
        torch.save(full_cache, cache_file)
        log.info(f"Saved full tokenization cache to {cache_file}")

    def _setup_with_cache(self, stage: Optional[str] = None):
        """Internal setup that uses tokenization caching."""
        # Initialize Tokenizer
        if self.preprocessor_type == "geneformer":
            tokenizer = GeneformerTokenizer(
                self.vocab_path, 
                self.gene_id_map_path,
                self.gene_median_path
            )
            print("Using Geneformer Tokenizer")
        else:
            tokenizer = ScGPTTokenizer(self.vocab_path)
            tokenizer = ScGPTTokenizer(self.vocab_path)
            print("Using scGPT Tokenizer")
            
        # --- Label Encoding Setup ---
        # Collect all cell types to fit encoder
        if self.mdata is not None:
            # Try to find cell type column
            obs = self.mdata.obs
            ct_col = None
            for name in ['cell_type', 'celltype', 'cell_types', 'label', 'labels', 'cell_type.l1', 'cell_type.l2']:
                if name in obs.columns:
                    ct_col = name
                    break
            
            if ct_col:
                all_labels = obs[ct_col].astype(str).unique()
                self.label_encoder.fit(all_labels)
                self.num_classes = len(all_labels)
                print(f"🧩 Label Encoder fitted with {self.num_classes} classes: {all_labels[:5]}...")
                
                # Store LE in mdata.uns for datasets to access
                self.mdata.uns["cell_type_le"] = self.label_encoder
            else:
                print("⚠️ Warning: No cell_type column found for Label Encoding.")
        
        # Cache directory for tokenization (inside data directory)
        cache_dir = self.data_dir / ".cache"
        
        # Precompute tokenization cache for full dataset (Geneformer only)
        if self.preprocessor_type == "geneformer":
            self._precompute_full_cache(tokenizer, cache_dir)
            
        # Split logic with FIXED SEED for reproducibility
        n = self.mdata.shape[0]
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(n)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        self.train_dataset = BMMCDataset(
            self.mdata[train_idx], tokenizer, self.max_seq_len, 
            split="train", preprocessor_type=self.preprocessor_type,
            cache_dir=cache_dir, cell_indices=train_idx
        )

        self.val_dataset = BMMCDataset(
            self.mdata[val_idx], tokenizer, self.max_seq_len, 
            split="val", preprocessor_type=self.preprocessor_type,
            cache_dir=cache_dir, cell_indices=val_idx
        )

        self.test_dataset = BMMCDataset(
            self.mdata[test_idx], tokenizer, self.max_seq_len, 
            split="test", preprocessor_type=self.preprocessor_type,
            cache_dir=cache_dir, cell_indices=test_idx
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
