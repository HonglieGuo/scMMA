"""
Training Entry Point for scMMA.

Uses Hydra for configuration management and PyTorch Lightning for training.
"""
# === Path initialization (ensure src package is importable) ===
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ==========================================

import torch
torch.set_float32_matmul_precision('medium')

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from src.models.unified_module import MultiModalAdapterModel
from src.utils.lora_utils import print_trainable_parameters



log = logging.getLogger(__name__)

def log_experiment_result(cfg: DictConfig, scores: dict, datamodule=None, actual_epochs: int = 0, log_file: str = "outputs/experiment_log.csv", pooling_mode: str = "auto", model=None, run_time_sec: float = 0.0, peak_mem_gb: float = 0.0):
    """
    Log experiment parameters and results to a permanent CSV file.
    """
    import pandas as pd
    import datetime
    import os
    import torch

    # 1. Extract Configs
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    lora_cfg = model_cfg.get("lora_config", {}) or {}

    # Helper to clean class names
    def get_name(cfg_node):
        name = cfg_node.get("_target_", "unknown").split(".")[-1] if isinstance(cfg_node, dict) or hasattr(cfg_node, "get") else "unknown"
        # Simplify names as requested
        return name.replace("DataModule", "").replace("Wrapper", "").replace("Fusion", "").replace("Encoder", "")

    # 2. Build Record (Order Dict for CSV Columns)
    record = {}
    
    # [Section 1] ID & Timestamp
    record["Times"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Backbone (inserted right after Times, before Dataset)
    # Extract model name from checkpoint_path, not _target_ class name
    backbone_path = str(model_cfg.get("backbone", {}).get("checkpoint_path", ""))
    backbone_lower = backbone_path.lower().replace("\\", "/")
    if "scgpt" in backbone_lower:
        record["Backbone"] = "scGPT"
    elif "geneformer" in backbone_lower:
        record["Backbone"] = "Geneformer"
    else:
        record["Backbone"] = backbone_path.split("/")[-1] if backbone_path else "Unknown"
    
    # Extract Dataset from data_dir path (e.g., datasets/h5ad/BMMC-p10/RNA+ADT -> BMMC-p10/RNA+ADT)
    data_dir = str(cfg.get("data", {}).get("data_dir", ""))
    clean_path = data_dir.replace("\\", "/").rstrip("/")
    path_parts = clean_path.split("/")
    if len(path_parts) >= 2:
        # Keep the last two parts: e.g. BMMC/RNA+ADT
        record["Dataset"] = "/".join(path_parts[-2:])
    elif len(path_parts) == 1 and path_parts[0]:
        record["Dataset"] = path_parts[0]
    else:
        record["Dataset"] = "Unknown"
        
    record["Pooling_Mode"] = pooling_mode
        
    if datamodule is not None and hasattr(datamodule, "vocab_hit_rate"):
        record["Vocab Hit Rate"] = f"{datamodule.vocab_hit_rate:.2f}%"
    else:
        record["Vocab Hit Rate"] = "N/A"
        
    if datamodule is not None and hasattr(datamodule, "has_labels"):
        record["Label Status"] = "Labeled" if datamodule.has_labels else "Unlabeled"
    else:
        record["Label Status"] = "N/A"

    mtp = model_cfg.get("mtp_weight", 0.0)
    cls = model_cfg.get("cls_weight", 0.0)
    con = model_cfg.get("contrastive_weight", 0.0)
    sup = model_cfg.get("supcon_weight", 0.0)
    record["Loss Weights"] = f"MTP:{mtp}|CLS:{cls}|Con:{con}|SupCon:{sup}"

    # Get Task Type (Vertical, Cross-batch, etc) from config
    # If datamodule has batch info, auto-detect cross-batch integration
    task_type = cfg.get("data", {}).get("integration_task", "Vertical")
    if datamodule is not None and hasattr(datamodule, "n_batches") and datamodule.n_batches > 1:
        task_type = "Cross-batch"
    record["Task Type"] = task_type

    # Extract Modality from data_dir path
    upper_dir = data_dir.upper()
    if "ADT" in upper_dir and "ATAC" in upper_dir:
        record["Modalities"] = "RNA+ADT+ATAC"
    elif "ADT" in upper_dir:
        record["Modalities"] = "RNA+ADT"
    elif "ATAC" in upper_dir:
        record["Modalities"] = "RNA+ATAC"
    else:
        record["Modalities"] = "Unknown"
        
    # [Section 2] Metrics - Explicit ordering for readability
    # Bio Conservation (higher is better)
    record["Leiden_Res"] = scores.get("leiden_res", 1.0)
    record["NMI"] = scores.get("nmi")
    record["ARI"] = scores.get("ari")
    record["ASW_label"] = scores.get("asw_label")
    record["cLISI"] = scores.get("clisi")
    
    # Batch Correction (higher is better after normalization)
    record["Batch_ASW"] = scores.get("batch_asw")
    record["iLISI"] = scores.get("ilisi")
    record["ASW_batch_raw"] = scores.get("asw_batch_raw")
    
    # Overall Score
    record["Overall"] = scores.get("overall_score")
    
    # Modality Alignment (lower is better for FOSCTTM, higher is better for Match@5 and LTA)
    record["FOSCTTM"] = scores.get("foscttm")
    record["Match@5"] = scores.get("match_at_5")
    record["LTA"] = scores.get("lta")
    
    record["Run_Time_Sec"] = run_time_sec
    record["Peak_Mem_GB"] = peak_mem_gb
        
    # [Section 3] Experiment Params
    record["Fusion"] = get_name(model_cfg.get("fusion_module", {}))
    
    proj_type = model_cfg.get("projection_head_type", "linear").upper()
    eval_proj = "Yes" if model_cfg.get("eval_apply_projection", False) else "No"
    record["Proj_Head"] = f"{proj_type} (Eval: {eval_proj})"
    record["Embedding_Dim"] = scores.get("Embedding_Dim", "N/A")
    
    atac_cfg_node = model_cfg.get("modality_encoders", {}).get("atac", None)
    if atac_cfg_node:
        record["ATAC_Encoder"] = get_name(atac_cfg_node)
    else:
        record["ATAC_Encoder"] = "N/A"
    
    adt_cfg_node = model_cfg.get("modality_encoders", {}).get("adt", None)
    if adt_cfg_node:
        adt_target = str(adt_cfg_node.get("_target_", ""))
        if "ADTEncoder" in adt_target:
            record["ADT_Encoder"] = "MLP"
        else:
            record["ADT_Encoder"] = get_name(adt_cfg_node)
    else:
        record["ADT_Encoder"] = "N/A"
    
    # [Section 4] Training Configs
    record["Max_Epochs"] = cfg.get("trainer", {}).get("max_epochs", "unknown")
    record["Actual_Epochs"] = actual_epochs
    record["Warmup_Epochs"] = model_cfg.get("warmup_epochs", 0)
    record["Accum_Grad"] = cfg.get("trainer", {}).get("accumulate_grad_batches", 1)
    record["Batch_Size"] = data_cfg.get("batch_size", "unknown")
    record["Max_Seq_Len"] = data_cfg.get("max_seq_len", "unknown")
    record["LR"] = model_cfg.get("learning_rate", "unknown")
    
    # [Section 5] Model Hyperparameters
    record["LoRA_r"] = lora_cfg.get("r", "N/A")
    record["Temperature"] = model_cfg.get("contrastive_temperature", 0.07)
    
    if model is not None:
        trainable_params = sum(p.numel() for p in set(model.parameters()) if p.requires_grad)
        total_params = sum(p.numel() for p in set(model.parameters()))
        record["Trainable Params"] = f"{trainable_params / 1e6:.1f} M"
        record["Total Params"] = f"{total_params / 1e6:.1f} M"
    else:
        record["Trainable Params"] = "N/A"
        record["Total Params"] = "N/A"

    # 3. Save to CSV
    df_new = pd.DataFrame([record])
    try:
        # Ensure dir exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        if os.path.exists(log_file):
            try:
                # Read existing to properly align and merge columns (handles insertion order changes)
                df_old = pd.read_csv(log_file)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                
                # Enforce the new column order based on the current record dictionary
                ordered_cols = list(record.keys())
                for col in df_combined.columns:
                    if col not in ordered_cols:
                        ordered_cols.append(col)
                df_combined = df_combined[ordered_cols]
                
                df_combined.to_csv(log_file, index=False)
            except pd.errors.EmptyDataError:
                # File exists but is empty
                df_new.to_csv(log_file, index=False)
        else:
            df_new.to_csv(log_file, index=False)
            
        log.info(f"📝 Experiment result logged to {log_file}")
        
    except Exception as e:
        log.error(f"❌ Failed to log experiment result: {e}")

@hydra.main(version_base="1.3", config_path="../configs", config_name="train_atac")
def train(cfg: DictConfig) -> None:
    """Main training routine."""
    
    # 1. Print Config
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)
    
    # 3. Instantiate DataModule & Auto-detect Dimensions
    log.info(f"Instantiating DataModule: <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # [Dynamic Adaptation] Load data immediately to check dimensions
    log.info("Loading data to auto-detect dimensions...")
    datamodule.setup(stage="fit")

    # [Dynamic Configuration Merge] Detect if dataset is unlabelled
    if hasattr(datamodule, "has_labels") and not datamodule.has_labels:
        log.warning("🚀 No cell type labels detected. Merging 'train_unlabel.yaml' for Unsupervised Integration.")
        unlabel_cfg_path = PROJECT_ROOT / "configs" / "train_unlabel.yaml"
        if unlabel_cfg_path.exists():
            unlabel_cfg = OmegaConf.load(unlabel_cfg_path)
            # Merge into existing cfg (overriding the relevant keys)
            cfg.model = OmegaConf.merge(cfg.model, unlabel_cfg.get("model", {}))
            cfg.logger = OmegaConf.merge(cfg.logger, unlabel_cfg.get("logger", {}))
            log.info("✅ Unsupervised parameters applied (SupCon/CLS disabled, RNA-ATAC Contrastive enabled).")
        else:
            log.error(f"❌ Configuration file {unlabel_cfg_path} not found! Check your configs directory.")
    
    # Check ATAC dimension
    if datamodule.mdata is not None and "atac" in datamodule.mdata.mod:
        actual_n_peaks = datamodule.mdata.mod["atac"].shape[1]
        log.info(f"✅ Auto-detected ATAC peaks: {actual_n_peaks}")
        
        # Override Config if exists
        # Override Config if exists
        if "atac" in cfg.model.modality_encoders:
            atac_cfg = cfg.model.modality_encoders.atac
            
            if atac_cfg is not None:
                # [CASE 1] Chromosomal Encoder requires `chrom_indices`
                if "ChromosomalEncoder" in atac_cfg.get("_target_", ""):
                     if hasattr(datamodule, "chrom_indices"):
                         log.info(f"🧬 Injecting chrom_indices: {len(datamodule.chrom_indices)} chromosomes found.")
                         # Convert to OmegaConf-compatible format (if needed, but Dict/List is fine)
                         atac_cfg.chrom_indices = datamodule.chrom_indices
                     else:
                         log.error("❌ ChromosomalEncoder selected but DataModule missing 'chrom_indices'! Ensure BMMCDataModule.setup() generates them.")

                # [CASE 2] Encoders requiring `n_peaks` (e.g., Conv1D, LatentPeak)
                if "n_peaks" in atac_cfg:
                    cfg_n_peaks = atac_cfg.n_peaks
                    if cfg_n_peaks != actual_n_peaks:
                        log.warning(f"⚠️ Config n_peaks ({cfg_n_peaks}) mismatch! Overriding with {actual_n_peaks}.")
                        atac_cfg.n_peaks = actual_n_peaks
                    else:
                        log.info(f"Config matches data dimension ({cfg_n_peaks}).")
                    
    # Check ADT dimension
    if datamodule.mdata is not None and "adt" in datamodule.mdata.mod:
        actual_n_proteins = datamodule.mdata.mod["adt"].shape[1]
        log.info(f"✅ Auto-detected ADT proteins: {actual_n_proteins}")
        if "adt" in cfg.model.modality_encoders:
            adt_cfg = cfg.model.modality_encoders.adt
            if adt_cfg is not None and "n_proteins" in adt_cfg:
                cfg_n_proteins = adt_cfg.n_proteins
                if cfg_n_proteins != actual_n_proteins:
                    log.warning(f"⚠️ Config n_proteins ({cfg_n_proteins}) mismatch! Overriding with {actual_n_proteins}.")
                    adt_cfg.n_proteins = actual_n_proteins
                else:
                    log.info(f"Config matches ADT dimension ({cfg_n_proteins}).")
    
    # 4. Instantiate Model Components
    log.info(f"Instantiating Backbone: <{cfg.model.backbone._target_}>")
    backbone = hydra.utils.instantiate(cfg.model.backbone)
    
    log.info("Instantiating Modality Encoders...")
    encoders = {}
    for name, conf in cfg.model.modality_encoders.items():
        if conf is not None:
            encoders[name] = hydra.utils.instantiate(conf)
            
    log.info(f"Instantiating Fusion Module: <{cfg.model.fusion_module._target_}>")
    fusion_module = hydra.utils.instantiate(cfg.model.fusion_module)
    
    # 4. Instantiate Unified Model
    log.info("Assembling MultiModalAdapterModel...")
    model = MultiModalAdapterModel(
        backbone=backbone,
        modality_encoders=encoders,
        fusion_module=fusion_module,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        use_lora=cfg.model.use_lora,
        lora_config=cfg.model.get("lora_config"),
        mask_ratio=cfg.model.mask_ratio,
        warmup_epochs=cfg.model.get("warmup_epochs", 0),
        num_classes=getattr(datamodule, "num_classes", 0),
        mtp_weight=cfg.model.get("mtp_weight", 1.0),
        cls_weight=cfg.model.get("cls_weight", 1.0),
        contrastive_weight=cfg.model.get("contrastive_weight", 0.1),
        supcon_weight=cfg.model.get("supcon_weight", 0.0),
        contrastive_temperature=cfg.model.get("contrastive_temperature", 0.07),
        projection_head_type=cfg.model.get("projection_head_type", "linear"),
        projection_dim=cfg.model.get("projection_dim", 256),
        eval_apply_projection=cfg.model.get("eval_apply_projection", False),
        weight_tying=cfg.model.get("weight_tying", False),
    )
    
    print_trainable_parameters(model)
    
    # 5. Callbacks & Logger
    import datetime
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    # ------------------ Unified Output Directory Detection ------------------
    backbone_path = str(cfg.model.backbone.get("checkpoint_path", ""))
    backbone_lower = backbone_path.lower().replace("\\", "/")
    if "scgpt" in backbone_lower:
        backbone_name = "scGPT"
    elif "geneformer" in backbone_lower:
        backbone_name = "Geneformer"
    else:
        # Fallback to class name if no checkpoint path
        backbone_name = cfg.model.backbone.get("_target_", "Unknown").split(".")[-1].replace("Wrapper", "")
    
    data_dir = str(cfg.get("data", {}).get("data_dir", ""))
    clean_path = data_dir.replace("\\", "/").rstrip("/")
    path_parts = clean_path.split("/")
    dataset_name = path_parts[-2] if len(path_parts) >= 2 else (path_parts[0] if path_parts[0] else "Unknown")
    
    run_identifier = f"{backbone_name}_{dataset_name}_{now_str}"
    unified_out_dir = f"outputs/{run_identifier}"
    Path(unified_out_dir).mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------------------------
    
    from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar  # Add import

    callbacks = [
        ModelCheckpoint(
            dirpath=unified_out_dir, 
            filename="best_model-epoch={epoch:02d}-val_loss={val_loss:.4f}",
            save_top_k=1, 
            monitor="val_loss", 
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=3,          # If loss doesn't improve for 3 epochs, stop
            verbose=True,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="step"),
        # Explicitly add Rich progress bar for better formatting
        RichProgressBar(leave=True),
    ]
    
    from lightning.pytorch.loggers import CSVLogger
    # To prevent version_0 nesting, we pass name="" and version=""
    logger = CSVLogger(
        save_dir=unified_out_dir,
        name="",
        version=""
    )
    
    # 6. Trainer
    log.info("Instantiating Trainer...")
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    
    # 7. Train
    log.info("Starting training...")
    import time
    start_time = time.time()
    from tqdm.contrib.logging import logging_redirect_tqdm
    # Redirect all Python logging through tqdm.write() during training
    # This prevents Hydra's StreamHandler from overwriting tqdm progress bars
    all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    all_loggers.append(logging.root)
    with logging_redirect_tqdm(loggers=all_loggers):
        trainer.fit(model=model, datamodule=datamodule)
    
    run_time_sec = time.time() - start_time
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    log.info(f"Training finished in {run_time_sec:.1f} seconds. Peak Memory: {peak_mem_gb:.2f} GB")
    
    # 8. End-to-End Evaluation (Benchmark)
    # Get Best Checkpoint
    checkpoint_callback = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
    best_model_path = checkpoint_callback.best_model_path
    
    if best_model_path:
        log.info(f"🏆 Best model found at: {best_model_path}")
        log.info("Loading best model weights for evaluation...")
        
        # Load weights into existing model (safest approach for complex composition)
        checkpoint = torch.load(best_model_path, weights_only=False)
        # Handle state_dict key mismatch if wrapper logic changed, usually direct load works
        model.load_state_dict(checkpoint["state_dict"])
        
        # Setup Test Data
        log.info("Setting up Test Data...")
        datamodule.setup(stage="test")
        
        # Run Evaluation
        from src.evaluation_loop import evaluate_model
        
        # Use unified output directory for evaluation files (embedding.h5ad, scores.csv)
        output_dir = unified_out_dir
        log.info(f"Running Benchmark... (Results -> {output_dir})")
        
        try:
            # Auto-select pooling_mode based on backbone type
            # Geneformer-30M uses Mean Pooling
            if cfg.model.pooling_mode == "auto":
                if "scgpt" in backbone_path:
                    cfg.model.pooling_mode = "cls"
                    log.info("Auto-set pooling mode to 'cls' for scGPT.")
                else:
                    cfg.model.pooling_mode = "mean"
                    log.info("Auto-set pooling mode to 'mean' for Geneformer.")
            
            pooling_mode = cfg.model.pooling_mode
            
            scores = evaluate_model(
                model=model,
                datamodule=datamodule,
                output_dir=output_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
                pooling_mode=pooling_mode
            )
            log.info(f"✅ Benchmark Complete! Score: {scores}")
            
            # --- 9. Log to Permanent Record ---
            try:
                custom_log_file = cfg.get("log_file", "outputs/experiment_log.csv")
                log_experiment_result(cfg, scores, datamodule=datamodule, actual_epochs=trainer.current_epoch + 1, log_file=custom_log_file, pooling_mode=pooling_mode, model=model, run_time_sec=run_time_sec, peak_mem_gb=peak_mem_gb)
            except Exception as e:
                log.error(f"Failed to write experiment log: {e}")
        except Exception as e:
            log.error(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        log.warning("No best model checkpoint found. Skipping evaluation.")

if __name__ == "__main__":
    train()
