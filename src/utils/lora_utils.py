"""
LoRA / PEFT Utility Functions.

Helpers to inspect and manage Low-Rank Adaptation parameters.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

def get_lora_config(
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> Dict:
    """create standard LoRA config dict"""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"] 
        # Note: adjust based on backbone architecture names (e.g. scGPT uses FlashAttn or native Torch NN)
    
    return {
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": target_modules
    }

def enable_lora(
    model: nn.Module, 
    config: Optional[Dict] = None
) -> nn.Module:
    """
    Wrap model with PEFT LoRA.
    Alternative manual usage if not done in LightningModule.
    """
    from peft import get_peft_model, LoraConfig, TaskType
    
    # Ensure config is complete
    c = get_lora_config(**(config or {}))
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=c["r"], 
        lora_alpha=c["alpha"], 
        lora_dropout=c["dropout"],
        target_modules=c["target_modules"],
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    return model
