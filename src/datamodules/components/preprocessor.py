"""
Preprocessing logic for scMMA.

Includes:
1. Gene Activity Matrix (GAM) calculation from ATAC.
2. Tokenization and vocabulary alignment for scGPT.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


class ScGPTTokenizer:
    """
    Tokenizer for scGPT.

    Handles mapping between gene names and token IDs.
    Loads vocabulary from scGPT model vocab.json files.
    
    scGPT vocab.json format: {"gene_name": token_id, ...}
    Special tokens (<pad>, <cls>, <eoc>) are added automatically if missing.
    """

    def __init__(self, vocab_path: Union[str, Path]):
        self.vocab_path = Path(vocab_path)
        self.vocab: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        self._load_vocab()

    def _load_vocab(self):
        """Load vocabulary JSON and ensure special tokens exist."""
        if not self.vocab_path.exists():
            print(f"Warning: Vocab file {self.vocab_path} not found. Using mock vocab.")
            self.vocab = {"<pad>": 0, "<cls>": 1, "<eoc>": 2}
            self.id2token = {v: k for k, v in self.vocab.items()}
            return

        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)

        # Auto-add special tokens if missing (append after max existing ID)
        max_id = max(self.vocab.values()) if self.vocab else -1
        for special in ["<pad>", "<cls>", "<eoc>"]:
            if special not in self.vocab:
                max_id += 1
                self.vocab[special] = max_id
                print(f"[ScGPTTokenizer] Added special token '{special}' with ID {max_id}")

        self.id2token = {v: k for k, v in self.vocab.items()}
        print(f"[ScGPTTokenizer] Loaded vocabulary: {len(self.vocab)} tokens")

    def tokenize(
        self,
        gene_names: List[str],
        pad_to_length: Optional[int] = None
    ) -> Tensor:
        """
        Convert gene names to token IDs.
        Only includes genes that exist in the vocabulary.

        Args:
            gene_names: List of gene symbols.
            pad_to_length: Optional length to pad/truncate to.

        Returns:
            Tensor of token IDs.
        """
        ids = []
        for g in gene_names:
            if g in self.vocab:
                ids.append(self.vocab[g])
            # Skip unknown genes (scGPT style: only known genes are used)

        if pad_to_length:
            if len(ids) < pad_to_length:
                ids += [self.pad_token_id] * (pad_to_length - len(ids))
            else:
                ids = ids[:pad_to_length]

        return torch.tensor(ids, dtype=torch.long)

    @property
    def pad_token_id(self) -> int:
        return self.vocab.get("<pad>", 0)

    @property
    def mask_token_id(self) -> int:
        return self.vocab.get("<mask>", self.vocab.get("<cls>", 1))


class GeneformerTokenizer:
    """
    Tokenizer for Geneformer (Rank-Value Encoding).
    
    CRITICAL: 
    1. Geneformer uses Ensembl IDs (ENSG...), NOT gene symbols.
    2. Expression values are normalized by gene-specific medians before ranking.
    3. Genes are sorted by normalized expression (High -> Low).
    """
    
    def __init__(
        self, 
        vocab_path: Union[str, Path], 
        gene_id_map_path: Optional[str] = None,
        gene_median_path: Optional[str] = None
    ):
        """
        Args:
            vocab_path: Path to Geneformer's token_dictionary (EnsemblID -> TokenID).
            gene_id_map_path: (Optional) Path to GeneSymbol -> EnsemblID mapping JSON. 
                              If None, assumes dataset already uses Ensembl IDs.
            gene_median_path: (Optional) Path to Gene -> Median expression JSON.
                              Used for normalization before ranking.
        """
        self.vocab_path = Path(vocab_path)
        self.gene_id_map_path = Path(gene_id_map_path) if gene_id_map_path else None
        self.gene_median_path = Path(gene_median_path) if gene_median_path else None
        self.vocab = {}
        self.symbol_to_ensembl = {}
        self.gene_medians = {}
        self._load_vocab()
        self._load_gene_mapping()
        self._load_gene_medians()
        
    def _load_vocab(self):
        # 1. Load Token Dictionary (Ensembl -> ID)
        if self.vocab_path.suffix == ".pkl":
            import pickle
            try:
                with open(self.vocab_path, "rb") as f:
                    self.vocab = pickle.load(f)
            except:
                print("[WARN] Geneformer token dict not found. Using Mock.")
                self.vocab = {"<pad>": 0, "<mask>": 1, "<unk>": 2}
        elif self.vocab_path.suffix == ".json":
            # Support JSON format token dictionary
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    import json
                    self.vocab = json.load(f)
                print(f"[INFO] Loaded Geneformer token dictionary: {len(self.vocab)} tokens")
            except Exception as e:
                print(f"[WARN] Failed to load JSON token dict: {e}. Using Mock.")
                self.vocab = {"<pad>": 0, "<mask>": 1, "<unk>": 2}
        else:
             self.vocab = {"<pad>": 0, "<mask>": 1, "<unk>": 2}
    
    def _load_gene_mapping(self):
        """Load Gene Symbol -> Ensembl ID mapping."""
        if self.gene_id_map_path and self.gene_id_map_path.exists():
            try:
                with open(self.gene_id_map_path, "r", encoding="utf-8") as f:
                    import json
                    self.symbol_to_ensembl = json.load(f)
                print(f"[INFO] Loaded gene symbol mapping: {len(self.symbol_to_ensembl)} genes")
            except Exception as e:
                print(f"[WARN] Failed to load gene mapping: {e}")
                self.symbol_to_ensembl = {}
        else:
            # No mapping file provided, assume dataset uses Ensembl IDs directly
            self.symbol_to_ensembl = {}
    
    def _load_gene_medians(self):
        """Load Gene -> Median expression values for normalization."""
        if self.gene_median_path and self.gene_median_path.exists():
            try:
                with open(self.gene_median_path, "r", encoding="utf-8") as f:
                    import json
                    self.gene_medians = json.load(f)
                print(f"[INFO] Loaded gene median dictionary: {len(self.gene_medians)} genes")
            except Exception as e:
                print(f"[WARN] Failed to load gene medians: {e}")
                self.gene_medians = {}
        else:
            # No median file provided, skip normalization
            self.gene_medians = {}
        
    def tokenize_rank_value(
        self,
        gene_names: List[str],  # These might be symbols or Ensembl IDs
        expression_values: Union[np.ndarray, List[float]],
        n_counts: Optional[float] = None,  # Total counts per cell (for CPM normalization)
        max_len: int = 2048,
        target_sum: float = 10000.0,  # CPM target (matches Geneformer default)
    ) -> Tensor:
        """
        Rank-Value Encoding following official Geneformer preprocessing:
        
        Steps (matching official Geneformer tokenizer.py):
        1. Filter expressed genes (>0)
        2. CPM Normalization: X / n_counts * target_sum (if n_counts provided)
        3. Divide by gene-specific median (if gene_medians loaded)
        4. Sort by normalized expression (Descending)
        5. Map Gene Symbol -> Ensembl ID -> Token ID
        6. Truncate to max_len
        7. Pad to max_len
        
        Args:
            gene_names: List of gene names (symbols or Ensembl IDs)
            expression_values: Raw expression counts
            n_counts: Total counts in cell (sum of all genes). Required for proper normalization.
            max_len: Maximum sequence length (2048 for Geneformer V1, 4096 for V2)
            target_sum: Target sum for CPM normalization (default 10000)
        """
        # Convert to numpy for faster processing
        if isinstance(expression_values, list):
            expression_values = np.array(expression_values)
        
        # Build pairs with full normalization pipeline
        pairs = []
        for i, (name, val) in enumerate(zip(gene_names, expression_values)):
            if val > 0:
                # Get Ensembl ID for dictionary lookups
                ensembl_id = self.symbol_to_ensembl.get(name, name)
                
                # Skip genes not in vocabulary
                if ensembl_id not in self.vocab and name not in self.vocab:
                    continue
                
                # Step 1: CPM Normalization (X / n_counts * target_sum)
                if n_counts is not None and n_counts > 0:
                    cpm_val = (val / n_counts) * target_sum
                else:
                    cpm_val = val
                
                # Step 2: Divide by gene median
                if self.gene_medians:
                    median = self.gene_medians.get(ensembl_id, 1.0)
                    if median > 0:
                        normalized_val = cpm_val / median
                    else:
                        normalized_val = cpm_val
                else:
                    normalized_val = cpm_val
                    
                pairs.append((name, normalized_val, ensembl_id))
                
        # Step 3: Sort by normalized expression (Descending)
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Truncate to max_len
        if len(pairs) > max_len:
            pairs = pairs[:max_len]
            
        # Step 5: Map to Token IDs
        token_ids = []
        for gene, val, ensembl_id in pairs:
            # Ensembl -> Token ID
            tid = self.vocab.get(ensembl_id)
            if tid is not None:
                token_ids.append(tid)
        
        # Step 6: Padding
        if len(token_ids) < max_len:
            token_ids += [self.vocab.get("<pad>", 0)] * (max_len - len(token_ids))
            
        return torch.tensor(token_ids, dtype=torch.long)
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab.get("<pad>", 0)


class GAMPreprocessor:
    """
    Gene Activity Matrix (GAM) Calculator.
    
    Calculates gene activity scores from ATAC-seq peaks.
    Usually: Sum of peaks within gene body + promoter region (e.g. 2kb upstream).
    """
    
    def __init__(self, gene_coords_path: Optional[str] = None):
        """
        Args:
            gene_coords_path: Path to gene annotation (GTF/BED).
        """
        self.gene_coords_path = gene_coords_path
        
    def calculate_gam(
        self, 
        atac_peaks: Union[np.ndarray, Tensor], 
        peak_coords: List[str],
        gene_names: List[str]
    ) -> np.ndarray:
        """
        Compute GAM from ATAC data.
        
        Args:
            atac_peaks: (n_cells, n_peaks) matrix (sparse or dense).
            peak_coords: List of peak regions "chr:start-end".
            gene_names: Target genes to compute scores for.
            
        Returns:
            (n_cells, n_genes) Gene Activity Matrix.
        """
        # TODO: Implement real interval intersection (PyRanges or BedTools)
        # For now, return a placeholder identity-like mapping or mock
        print("Warning: GAM calculation is currently a placeholder logic.")
        
        n_cells = atac_peaks.shape[0]
        n_genes = len(gene_names)
        
        # Placeholder: Random projection to simulate GAM
        # In real implementation: Map peaks to genes based on overlap
        return np.random.randn(n_cells, n_genes)


def _digitize(x: np.ndarray, bins: np.ndarray, side: str = "both") -> np.ndarray:
    """
    Digitize the data into bins. Matches the official scGPT implementation.
    
    When side="both", spreads data uniformly when bins have same values
    (using random interpolation between left and right digitization).
    
    Args:
        x: 1D array of values to digitize.
        bins: 1D array of bin edges in increasing order.
        side: "one" for left-side only, "both" for interpolated.
    
    Returns:
        Digitized values as int64 array.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_digits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers
    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, Tensor],
    n_bins: int = 51
) -> Tensor:
    """
    Discretize continuous expression values into bins (for scGPT).
    
    Matches the official scGPT preprocessing:
    - Zero values → 0 (special bin for no expression)
    - Non-zero values → binned to [1, n_bins-1] using quantile-based edges
    
    Args:
        row: Expression values for one cell, shape (n_genes,).
        n_bins: Number of bins (default 51 as in scGPT).
        
    Returns:
        Binned integer values as Tensor.
    """
    if isinstance(row, Tensor):
        row = row.cpu().numpy()
    
    row = np.asarray(row, dtype=np.float64)
    
    if row.max() == 0:
        return torch.zeros(len(row), dtype=torch.long)
    
    if row.min() <= 0:
        # Has zeros: bin only non-zero values, keep zeros as 0
        non_zero_ids = row.nonzero()[0]
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        # All positive: bin everything
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    
    return torch.from_numpy(binned_row).long()
