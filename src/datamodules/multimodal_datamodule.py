"""
Unified Multi-Modal DataModule for scMMA.

Loads single-cell multi-omics data from a directory of .h5ad files.
Automatically detects available modalities (RNA, ATAC, ADT) based on
filenames in the data directory.

Supported directory structures:
    datasets/h5ad/BMMC-p10/RNA+ATAC/
        BMMC-...-RNA-counts.h5ad
        BMMC-...-ATAC-counts.h5ad
    datasets/h5ad/BMMC-p10/RNA+ADT/
        BMMC-...-RNA-counts.h5ad
        BMMC-...-ADT-counts.h5ad
    datasets/HSPC/RNA+ADT/
        ...
    datasets/PBMC_CITE_seq/RNA+ADT/
        ...
"""

from pathlib import Path
from typing import Optional, Union, List, Dict
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


# ---------------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------------

def multimodal_collate_fn(batch):
    """
    Custom collate that handles sparse ATAC tensors gracefully.

    For dense inputs (gene_ids, gene_values, adt, etc.), uses default_collate.
    For sparse ATAC inputs, builds a batched 2D sparse COO tensor.
    """
    elem = batch[0]
    result = {}

    for key in elem:
        if key == "atac" and isinstance(elem[key], torch.Tensor) and elem[key].is_sparse:
            batch_size = len(batch)
            n_peaks = batch[0][key].shape[0]

            all_indices = []
            all_values = []

            for i, item in enumerate(batch):
                sparse_t = item[key]
                idx = sparse_t.indices()  # (1, nnz)
                val = sparse_t.values()
                batch_idx = torch.full_like(idx, i)
                stacked_indices = torch.cat([batch_idx, idx], dim=0)
                all_indices.append(stacked_indices)
                all_values.append(val)

            final_indices = torch.cat(all_indices, dim=1)
            final_values = torch.cat(all_values)
            result[key] = torch.sparse_coo_tensor(
                final_indices, final_values, size=(batch_size, n_peaks)
            ).coalesce()
        else:
            result[key] = default_collate([item[key] for item in batch])

    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal single-cell data (RNA + optional ATAC/ADT).

    Supports precomputed Geneformer tokenization cache for fast loading.
    """

    def __init__(
        self,
        mdata: mu.MuData,
        tokenizer: Union[ScGPTTokenizer, GeneformerTokenizer],
        max_seq_len: int = 1024,
        split: str = "train",
        preprocessor_type: str = "geneformer",
        cache_dir: Optional[Path] = None,
        cell_indices: Optional[np.ndarray] = None,
        label_col: str = "cell_type",
    ):
        self.mdata = mdata
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.preprocessor_type = preprocessor_type
        self.cache_dir = cache_dir
        self.cell_indices = cell_indices
        self.label_col = label_col

        # --- RNA ---
        self.rna = mdata.mod["rna"]
        self.gene_names = self.rna.var_names.tolist()

        # --- ATAC (optional) ---
        self.atac_dense = None
        if "atac" in mdata.mod:
            atac = mdata.mod["atac"]
            log.info(f"Pre-densifying ATAC for {split} ({atac.shape[0]} cells, {atac.shape[1]} peaks)...")
            if issparse(atac.X):
                self.atac_dense = atac.X.toarray().astype(np.float32)
            elif hasattr(atac.X, "toarray"):
                self.atac_dense = atac.X.toarray().astype(np.float32)
            else:
                self.atac_dense = np.asarray(atac.X, dtype=np.float32)
            log.info(f"ATAC densification complete: {self.atac_dense.shape}")

        # --- ADT (optional) ---
        self.adt_dense = None
        if "adt" in mdata.mod:
            adt = mdata.mod["adt"]
            log.info(f"Detected ADT for {split} ({adt.shape[0]} cells, {adt.shape[1]} proteins). CLR normalizing...")
            if issparse(adt.X):
                adt_raw = adt.X.toarray().astype(np.float32)
            elif hasattr(adt.X, "toarray"):
                adt_raw = adt.X.toarray().astype(np.float32)
            else:
                adt_raw = np.asarray(adt.X, dtype=np.float32)

            adt_log = np.log1p(adt_raw)
            adt_mean = np.mean(adt_log, axis=1, keepdims=True)
            self.adt_dense = adt_log - adt_mean
            log.info("ADT CLR normalization complete.")

        # --- Cell metadata ---
        self.cell_types = self._get_obs_column(
            [label_col, "cell_type", "celltype", "cell_types", "label",
             "labels", "celltype.l1", "celltype.l2", "celltype.l3"]
        )
        self.batches = self._get_obs_column(
            ["batch", "Batch", "donor", "Donor", "sample", "site", "Site"]
        )

        # --- Tokenization ---
        self.token_cache: Optional[torch.Tensor] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.scgpt_gene_ids_cache: Optional[torch.Tensor] = None
        self.scgpt_gene_values_cache: Optional[torch.Tensor] = None
        self.label_encoder = None
        if hasattr(mdata, "uns") and "cell_type_le" in mdata.uns:
            self.label_encoder = mdata.uns["cell_type_le"]

        if self.preprocessor_type == "scgpt" and cache_dir is not None:
            self._load_scgpt_cache()
        elif self.preprocessor_type == "geneformer" and cache_dir is not None:
            self._load_geneformer_cache()

    # ----- helpers -----

    def _get_obs_column(self, possible_names: List[str]) -> Optional[np.ndarray]:
        """Search for a metadata column across all obs DataFrames."""
        obs_dfs = [self.mdata.obs, self.mdata.mod["rna"].obs]
        for mod_name in ("adt", "atac"):
            if mod_name in self.mdata.mod:
                obs_dfs.append(self.mdata.mod[mod_name].obs)

        for obs_df in obs_dfs:
            for name in possible_names:
                if name in obs_df.columns:
                    return obs_df[name].values
        return None

    def _load_geneformer_cache(self):
        """Load precomputed tokenization cache, selecting cells for this split."""
        if self.cache_dir is None:
            return

        # Use vocab_path hash to differentiate caches for different foundation models
        import hashlib
        vocab_hash = hashlib.md5(str(self.tokenizer.vocab_path).encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"geneformer_tokens_{self.max_seq_len}_{vocab_hash}.pt"
        if cache_file.exists():
            log.info(f"Loading tokenization cache from {cache_file}")
            cache_data = torch.load(cache_file, weights_only=True)
            full_tokens = cache_data["tokens"]
            full_masks = cache_data["masks"]

            if self.cell_indices is not None:
                self.token_cache = full_tokens[self.cell_indices]
                self.mask_cache = full_masks[self.cell_indices]
                log.info(f"Selected {len(self.token_cache)} cached tokens for {self.split}")
            else:
                self.token_cache = full_tokens
                self.mask_cache = full_masks
                log.info(f"Loaded {len(self.token_cache)} cached tokens")
        else:
            log.warning(f"Cache not found: {cache_file}. Falling back to on-the-fly tokenization.")

    def _load_scgpt_cache(self):
        """Load precomputed scGPT tokenization cache, selecting cells for this split."""
        if self.cache_dir is None:
            return

        import hashlib
        vocab_hash = hashlib.md5(str(self.tokenizer.vocab_path).encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"scgpt_tokens_{self.max_seq_len}_{vocab_hash}.pt"
        if cache_file.exists():
            log.info(f"Loading scGPT tokenization cache from {cache_file}")
            cache_data = torch.load(cache_file, weights_only=True)
            full_gene_ids = cache_data["gene_ids"]
            full_gene_values = cache_data["gene_values"]

            if self.cell_indices is not None:
                self.scgpt_gene_ids_cache = full_gene_ids[self.cell_indices]
                self.scgpt_gene_values_cache = full_gene_values[self.cell_indices]
                log.info(f"Selected {len(self.scgpt_gene_ids_cache)} cached scGPT tokens for {self.split}")
            else:
                self.scgpt_gene_ids_cache = full_gene_ids
                self.scgpt_gene_values_cache = full_gene_values
                log.info(f"Loaded {len(self.scgpt_gene_ids_cache)} cached scGPT tokens")
        else:
            log.warning(f"scGPT cache not found: {cache_file}. Falling back to on-the-fly tokenization.")

    # ----- __getitem__ -----

    def __len__(self):
        return self.mdata.shape[0]

    def __getitem__(self, idx: int):
        # ---- 1. RNA tokens ----
        if self.preprocessor_type == "geneformer":
            if self.token_cache is not None:
                gene_ids = self.token_cache[idx]
                attention_mask = self.mask_cache[idx]
                gene_values = torch.ones_like(gene_ids, dtype=torch.float32)
            else:
                rna_expr = self.rna.X[idx]
                if issparse(rna_expr):
                    rna_expr = rna_expr.toarray().flatten()
                elif hasattr(rna_expr, "toarray"):
                    rna_expr = rna_expr.toarray().flatten()

                n_counts = float(np.sum(rna_expr))
                gene_ids = self.tokenizer.tokenize_rank_value(
                    self.gene_names, rna_expr, n_counts=n_counts, max_len=self.max_seq_len
                )
                pad_id = self.tokenizer.pad_token_id
                attention_mask = (gene_ids != pad_id).long()
                gene_values = torch.ones_like(gene_ids, dtype=torch.float32)
        else:
            # scGPT strategy: gene names → token IDs, expression → binned values
            if self.scgpt_gene_ids_cache is not None:
                # Fast path: use precomputed cache (array index only)
                gene_ids = self.scgpt_gene_ids_cache[idx]
                gene_values = self.scgpt_gene_values_cache[idx]
                attention_mask = (gene_ids != self.tokenizer.pad_token_id).long()
            else:
                # Fallback: on-the-fly tokenization
                rna_expr = self.rna.X[idx]
                if issparse(rna_expr):
                    rna_expr = rna_expr.toarray().flatten()
                elif hasattr(rna_expr, "toarray"):
                    rna_expr = rna_expr.toarray().flatten()

                # Non-zero gene filtering: only keep expressed genes (scGPT style)
                nonzero_mask = rna_expr > 0
                expressed_genes = [g for g, m in zip(self.gene_names, nonzero_mask) if m]
                expressed_values = rna_expr[nonzero_mask]

                # Tokenize expressed genes (only genes in vocab)
                gene_ids_list = []
                gene_vals_list = []
                for g, v in zip(expressed_genes, expressed_values):
                    if g in self.tokenizer.vocab:
                        gene_ids_list.append(self.tokenizer.vocab[g])
                        gene_vals_list.append(v)

                # Truncate to max_seq_len
                gene_ids_list = gene_ids_list[: self.max_seq_len]
                gene_vals_list = gene_vals_list[: self.max_seq_len]

                # Bin expression values
                if len(gene_vals_list) > 0:
                    binned_vals = binning(np.array(gene_vals_list, dtype=np.float64))
                else:
                    binned_vals = torch.zeros(0, dtype=torch.long)

                gene_ids = torch.tensor(gene_ids_list, dtype=torch.long)
                # gene_values as float for ContinuousValueEncoder
                gene_values = binned_vals.float()

                # Pad to max_seq_len
                pad_len = self.max_seq_len - len(gene_ids)
                if pad_len > 0:
                    pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                    pad_vals = torch.zeros((pad_len,), dtype=torch.float32)
                    gene_ids = torch.cat([gene_ids, pad_ids])
                    gene_values = torch.cat([gene_values, pad_vals])

                attention_mask = (gene_ids != self.tokenizer.pad_token_id).long()

        # ---- 2. Build result ----
        result: Dict[str, object] = {
            "gene_ids": gene_ids.long(),
            # scGPT uses float gene_values (binned), Geneformer uses long (dummy ones)
            "gene_values": gene_values.float() if self.preprocessor_type == "scgpt" else gene_values.long(),
            "attention_mask": attention_mask,
        }

        # ATAC (optional)
        if self.atac_dense is not None:
            result["atac"] = torch.from_numpy(self.atac_dense[idx])

        # ADT (optional)
        if self.adt_dense is not None:
            result["adt"] = torch.from_numpy(self.adt_dense[idx])

        # ---- 3. Metadata ----
        if self.cell_types is not None:
            cell_type_str = str(self.cell_types[idx])
            result["cell_type_str"] = cell_type_str
            if self.label_encoder is not None:
                result["cell_type"] = torch.tensor(
                    self.label_encoder.transform([cell_type_str])[0], dtype=torch.long
                )
            else:
                result["cell_type"] = -1
        else:
            result["cell_type_str"] = "unknown"
            result["cell_type"] = -1

        if self.batches is not None:
            result["batch"] = self.batches[idx]
        else:
            result["batch"] = "batch_0"

        return result


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class MultiModalDataModule(pl.LightningDataModule):
    """
    Unified LightningDataModule for arbitrary single-cell multi-omics datasets.

    Automatically detects available modalities (RNA, ATAC, ADT) from .h5ad
    files in the data directory. Supports any dataset layout following the
    naming convention: *-RNA-*.h5ad, *-ATAC-*.h5ad, *-ADT-*.h5ad.

    Examples:
        datasets/h5ad/BMMC-p10/RNA+ATAC/     → RNA + ATAC
        datasets/h5ad/BMMC-p10/RNA+ADT/      → RNA + ADT
        datasets/HSPC/RNA+ADT/      → RNA + ADT
        datasets/PBMC_CITE_seq/RNA+ADT/ → RNA + ADT
    """

    def __init__(
        self,
        data_dir: str = "datasets/",
        vocab_path: str = "models/pretrained/token_files/Geneformer/token_dictionary_geneformer.json",
        gene_id_map_path: Optional[str] = None,
        gene_median_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 16,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_seq_len: int = 1024,
        preprocessor_type: str = "geneformer",
        label_col: str = "cell_type",
        integration_task: str = "Vertical",
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
        self.persistent_workers = persistent_workers and num_workers > 0
        self.max_seq_len = max_seq_len
        self.preprocessor_type = preprocessor_type
        self.label_col = label_col
        self.integration_task = integration_task

        self.mdata: Optional[mu.MuData] = None
        self.train_dataset: Optional[MultiModalDataset] = None
        self.val_dataset: Optional[MultiModalDataset] = None
        self.test_dataset: Optional[MultiModalDataset] = None

        self.label_encoder = LabelEncoder()
        self.num_classes = 0

        # Populated during setup for ATAC chromosome processing
        self.chrom_indices: Optional[Dict] = None

    def prepare_data(self):
        pass

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        """Load data from .h5ad files and create train/val/test splits."""
        if self.mdata is not None:
            return  # Already loaded

        # --- 1. Auto-detect and load modalities ---
        rna_files = sorted(self.data_dir.glob("*RNA*.h5ad"))
        adt_files = sorted(self.data_dir.glob("*ADT*.h5ad"))
        atac_files = sorted(self.data_dir.glob("*ATAC*.h5ad"))

        if not rna_files:
            raise FileNotFoundError(f"No RNA .h5ad files found in {self.data_dir}")

        log.info(f"Loading RNA from {rna_files[0]}")
        mods: Dict[str, ad.AnnData] = {"rna": ad.read_h5ad(rna_files[0])}

        if adt_files:
            log.info(f"Loading ADT from {adt_files[0]}")
            mods["adt"] = ad.read_h5ad(adt_files[0])

        if atac_files:
            log.info(f"Loading ATAC from {atac_files[0]}")
            mods["atac"] = ad.read_h5ad(atac_files[0])

        self.mdata = mu.MuData(mods)
        if not hasattr(self.mdata, "uns"):
            self.mdata.uns = {}
        self.mdata.uns["label_col"] = self.label_col

        detected = ", ".join(mods.keys())
        log.info(f"Detected modalities: [{detected}] ({self.mdata.shape[0]} cells)")

        # --- 2. ATAC-specific processing (chromosome sorting) ---
        if "atac" in self.mdata.mod:
            self._process_atac()

        # --- 3. Tokenizer, labels, cache, splits ---
        self._setup_with_cache(stage)

    # ------------------------------------------------------------------
    # ATAC Processing
    # ------------------------------------------------------------------

    def _process_atac(self):
        """Sort ATAC peaks by genomic position and build chromosome indices."""
        atac = self.mdata.mod["atac"]

        def parse_chr(name):
            return str(name).replace(":", "-").split("-")[0]

        def parse_start(name):
            parts = str(name).replace(":", "-").split("-")
            return int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0

        # Standardize chromosome annotations
        if "chr" not in atac.var.columns:
            log.warning("'chr' column not in ATAC var. Parsing from index.")
            try:
                atac.var["chr"] = atac.var_names.map(parse_chr)
                atac.var["start"] = atac.var_names.map(parse_start)
            except Exception as e:
                log.warning(f"Failed to parse ATAC index: {e}. Using dummy chr1.")
                atac.var["chr"] = "chr1"
                atac.var["start"] = np.arange(atac.shape[1])

        # Filter valid chromosomes
        valid_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
        is_valid = atac.var["chr"].isin(valid_chroms)
        if not is_valid.all():
            log.info(f"Filtering non-standard chroms: {atac.shape[1]} → {is_valid.sum()} peaks.")
            filt_atac = atac[:, is_valid].copy()
        else:
            filt_atac = atac.copy()

        # Re-parse on filtered data
        filt_atac.var["chr"] = filt_atac.var_names.map(parse_chr)
        filt_atac.var["start"] = filt_atac.var_names.map(parse_start)

        # Sort by genomic position
        sorted_idx = filt_atac.var.sort_values(["chr", "start"]).index
        if not filt_atac.var_names.equals(sorted_idx):
            log.info("Sorting ATAC peaks by genomic position...")
            filt_atac = filt_atac[:, sorted_idx].copy()

        # Force dense
        if hasattr(filt_atac.X, "toarray"):
            filt_atac.X = filt_atac.X.toarray()
        elif hasattr(filt_atac, "file") or "h5" in str(type(filt_atac.X)).lower():
            filt_atac.X = np.array(filt_atac.X)

        # Reconstruct MuData with processed ATAC
        original_obs = self.mdata.obs.copy()
        rna = self.mdata.mod["rna"].copy()
        new_mods = {"rna": rna, "atac": filt_atac}
        if "adt" in self.mdata.mod:
            new_mods["adt"] = self.mdata.mod["adt"].copy()
        self.mdata = mu.MuData(new_mods)
        self.mdata.update()

        # Restore obs metadata
        for col in original_obs.columns:
            if col not in self.mdata.obs.columns:
                self.mdata.obs[col] = original_obs[col].values

        if not hasattr(self.mdata, "uns"):
            self.mdata.uns = {}
        self.mdata.uns["label_col"] = self.label_col

        # Build chromosome index map
        atac = self.mdata.mod["atac"]
        self.chrom_indices = {}
        current_idx = 0
        for chrom, group in atac.var.groupby("chr", sort=False):
            n = len(group)
            self.chrom_indices[chrom] = [current_idx, current_idx + n]
            current_idx += n
        log.info(f"Built chromosome indices for {len(self.chrom_indices)} chromosomes.")
        assert current_idx == atac.shape[1], "Chromosome indices do not cover all peaks!"

    # ------------------------------------------------------------------
    # Cache + Splits
    # ------------------------------------------------------------------

    def _precompute_full_cache(self, tokenizer, cache_dir: Path):
        """Precompute Geneformer tokenization cache for the entire dataset."""
        # Use vocab_path hash to differentiate caches for different foundation models
        import hashlib
        vocab_hash = hashlib.md5(str(tokenizer.vocab_path).encode()).hexdigest()[:8]
        cache_file = cache_dir / f"geneformer_tokens_{self.max_seq_len}_{vocab_hash}.pt"

        if cache_file.exists():
            log.info(f"Found existing tokenization cache: {cache_file}")
            return

        log.info(f"Precomputing tokenization cache for {self.mdata.shape[0]} cells...")
        cache_dir.mkdir(parents=True, exist_ok=True)

        rna = self.mdata.mod["rna"]
        gene_names = rna.var_names.tolist()
        n_samples = rna.shape[0]

        tokens_list = []
        masks_list = []

        for idx in tqdm(range(n_samples), desc="Tokenizing cells"):
            rna_expr = rna.X[idx]
            if issparse(rna_expr):
                rna_expr = rna_expr.toarray().flatten()
            elif hasattr(rna_expr, "toarray"):
                rna_expr = rna_expr.toarray().flatten()

            n_counts = float(np.sum(rna_expr))
            gene_ids = tokenizer.tokenize_rank_value(
                gene_names, rna_expr, n_counts=n_counts, max_len=self.max_seq_len
            )
            pad_id = tokenizer.pad_token_id
            attention_mask = (gene_ids != pad_id).long()

            tokens_list.append(gene_ids)
            masks_list.append(attention_mask)

        full_cache = {
            "tokens": torch.stack(tokens_list),
            "masks": torch.stack(masks_list),
        }
        torch.save(full_cache, cache_file)
        log.info(f"Saved tokenization cache to {cache_file}")

    def _precompute_scgpt_cache(self, tokenizer, cache_dir: Path):
        """Precompute scGPT tokenization cache (gene_ids + binned values) for the entire dataset."""
        import hashlib
        vocab_hash = hashlib.md5(str(tokenizer.vocab_path).encode()).hexdigest()[:8]
        cache_file = cache_dir / f"scgpt_tokens_{self.max_seq_len}_{vocab_hash}.pt"

        if cache_file.exists():
            log.info(f"Found existing scGPT tokenization cache: {cache_file}")
            return

        log.info(f"Precomputing scGPT tokenization cache for {self.mdata.shape[0]} cells...")
        cache_dir.mkdir(parents=True, exist_ok=True)

        rna = self.mdata.mod["rna"]
        gene_names = rna.var_names.tolist()
        n_samples = rna.shape[0]

        # Pre-build vectorized vocab mapping (avoids per-cell Python for-loop)
        vocab_mask = np.array([g in tokenizer.vocab for g in gene_names], dtype=bool)
        vocab_ids = np.array([tokenizer.vocab.get(g, 0) for g in gene_names], dtype=np.int64)
        pad_id = tokenizer.pad_token_id

        all_gene_ids = torch.full((n_samples, self.max_seq_len), pad_id, dtype=torch.long)
        all_gene_values = torch.zeros(n_samples, self.max_seq_len, dtype=torch.float32)

        for idx in tqdm(range(n_samples), desc="Tokenizing cells (scGPT)"):
            rna_expr = rna.X[idx]
            if issparse(rna_expr):
                rna_expr = rna_expr.toarray().flatten()
            elif hasattr(rna_expr, "toarray"):
                rna_expr = rna_expr.toarray().flatten()
            else:
                rna_expr = np.asarray(rna_expr).flatten()

            # Non-zero + in-vocab filter (vectorized, no Python loop)
            valid_mask = (rna_expr > 0) & vocab_mask
            valid_ids = vocab_ids[valid_mask]
            valid_vals = rna_expr[valid_mask]

            # Truncate to max_seq_len
            n_valid = min(len(valid_ids), self.max_seq_len)
            if n_valid == 0:
                continue
            valid_ids = valid_ids[:n_valid]
            valid_vals = valid_vals[:n_valid]

            # Bin expression values
            binned_vals = binning(valid_vals.astype(np.float64))

            all_gene_ids[idx, :n_valid] = torch.from_numpy(valid_ids)
            all_gene_values[idx, :n_valid] = binned_vals.float()

        full_cache = {
            "gene_ids": all_gene_ids,
            "gene_values": all_gene_values,
        }
        torch.save(full_cache, cache_file)
        log.info(f"Saved scGPT tokenization cache to {cache_file}")

    def _setup_with_cache(self, stage: Optional[str] = None):
        """Initialize tokenizer, label encoder, cache, and create split datasets."""
        # Tokenizer
        if self.preprocessor_type == "geneformer":
            tokenizer = GeneformerTokenizer(
                self.vocab_path, self.gene_id_map_path, self.gene_median_path
            )
            log.info(f"Using Geneformer Tokenizer ({len(tokenizer.vocab)} tokens)")
        else:
            tokenizer = ScGPTTokenizer(self.vocab_path)
            log.info("Using scGPT Tokenizer")

        # Compute vocab hit rate
        rna_genes = self.mdata.mod["rna"].var_names.tolist()
        hits = 0
        for gene in rna_genes:
            if self.preprocessor_type == "geneformer":
                ensembl_id = tokenizer.symbol_to_ensembl.get(gene, gene)
                if ensembl_id in tokenizer.vocab or gene in tokenizer.vocab:
                    hits += 1
            else:
                if gene in tokenizer.vocab:
                    hits += 1
        
        self.vocab_hit_rate = (hits / len(rna_genes)) * 100 if len(rna_genes) > 0 else 0
        log.info(f"🎯 Vocab Hit Rate: {self.vocab_hit_rate:.2f}% ({hits}/{len(rna_genes)} genes)")

        # Label Encoding
        if self.mdata is not None:
            ct_col = None
            candidates = [
                self.label_col, "cell_type", "celltype", "cell_types",
                "label", "labels", "celltype.l1", "celltype.l2", "celltype.l3",
            ]

            obs_dfs = [self.mdata.obs, self.mdata.mod["rna"].obs]
            for mod_name in ("adt", "atac"):
                if mod_name in self.mdata.mod:
                    obs_dfs.append(self.mdata.mod[mod_name].obs)

            for obs_df in obs_dfs:
                for name in candidates:
                    if name in obs_df.columns:
                        ct_col = name
                        used_obs = obs_df
                        break
                if ct_col:
                    break

            if ct_col:
                all_labels = used_obs[ct_col].astype(str).unique()
                self.label_encoder.fit(all_labels)
                self.num_classes = len(all_labels)
                log.info(f"🧩 Label Encoder: {self.num_classes} classes from '{ct_col}'")
                self.mdata.uns["cell_type_le"] = self.label_encoder
                self.has_labels = True
            else:
                log.warning(f"⚠️ Label column '{self.label_col}' (or fallbacks) not found. Entering Unsupervised Mode.")
                self.has_labels = False
                self.num_classes = 0

        # Detect number of batches for integration type auto-detection
        self.n_batches = 1
        batch_candidates = ["batch", "Batch", "donor", "Donor", "sample", "site", "Site"]
        for obs_df in [self.mdata.mod["rna"].obs, self.mdata.obs]:
            for col in batch_candidates:
                if col in obs_df.columns:
                    self.n_batches = obs_df[col].nunique()
                    log.info(f"🔢 Detected {self.n_batches} batches from '{col}' column")
                    break
            if self.n_batches > 1:
                break

        # Cache
        cache_dir = self.data_dir / ".cache"
        if self.preprocessor_type == "geneformer":
            self._precompute_full_cache(tokenizer, cache_dir)
        elif self.preprocessor_type == "scgpt":
            self._precompute_scgpt_cache(tokenizer, cache_dir)

        # Train/Val/Test Split (80/10/10, seed=42)
        n = self.mdata.shape[0]
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(n)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_idx = indices[:n_train]
        val_idx = indices[n_train: n_train + n_val]
        test_idx = indices[n_train + n_val:]

        ds_kwargs = dict(
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
            preprocessor_type=self.preprocessor_type,
            cache_dir=cache_dir,
            label_col=self.label_col,
        )

        self.train_dataset = MultiModalDataset(
            self.mdata[train_idx], split="train", cell_indices=train_idx, **ds_kwargs
        )
        self.val_dataset = MultiModalDataset(
            self.mdata[val_idx], split="val", cell_indices=val_idx, **ds_kwargs
        )
        self.test_dataset = MultiModalDataset(
            self.mdata[test_idx], split="test", cell_indices=test_idx, **ds_kwargs
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

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
