"""
Benchmark Entry Point for scMMA.

Runs evaluation on the test set and calculates SCMMIB metrics.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
import lightning.pytorch as pl

from src.models.unified_module import MultiModalAdapterModel
from src.datamodules.bmmc_datamodule import BMMCDataModule
from src.evaluation_loop import evaluate_model

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def benchmark(cfg: DictConfig) -> None:
    """Main benchmarking routine."""
    
    # 1. Load Model from Checkpoint
    checkpoint_path = cfg.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        log.error(f"Checkpoint not found at {checkpoint_path}")
        # Note: In real usage, you'd load from checkpoint. 
        # For scaffold verification, we might instantiate fresh if chkpt missing
        log.warning("Instantiating fresh model instead (Testing mode)")
        
        # Instantiate fresh components (same as train.py)
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        encoders = {k: hydra.utils.instantiate(v) for k,v in cfg.model.modality_encoders.items()}
        fusion = hydra.utils.instantiate(cfg.model.fusion_module)
        
        model = MultiModalAdapterModel(
            backbone=backbone,
            modality_encoders=encoders,
            fusion_module=fusion,
            use_lora=cfg.model.use_lora
        )
    else:
        # Load from checkpoint
        # Strategy: Instantiate components first, then load state dict
        log.info(f"Loading weights from {checkpoint_path}")
        
        # Instantiate fresh components (same as train.py)
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        encoders = {k: hydra.utils.instantiate(v) for k,v in cfg.model.modality_encoders.items()}
        fusion = hydra.utils.instantiate(cfg.model.fusion_module)
        
        model = MultiModalAdapterModel(
            backbone=backbone,
            modality_encoders=encoders,
            fusion_module=fusion,
            use_lora=cfg.model.use_lora,
            lora_config=cfg.model.get("lora_config")
        )
        
        # Load state dict from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle potential mismatches
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(f"Missing keys: {len(missing)} (LoRA params expected if use_lora=False)")
        if unexpected:
            log.warning(f"Unexpected keys: {len(unexpected)}")

    # 2. DataModule
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    
    # 3. Run Evaluation Loop
    pooling_mode = cfg.get("pooling_mode", "mean")
    log.info(f"Starting Benchmark Evaluation (Pooling: {pooling_mode})...")
    scores = evaluate_model(
        model=model,
        datamodule=datamodule,
        output_dir=cfg.output_dir,
        pooling_mode=pooling_mode
    )
    
    log.info(f"Benchmark Scores: {scores}")

if __name__ == "__main__":
    benchmark()
