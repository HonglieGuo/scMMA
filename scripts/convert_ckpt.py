
import torch
import sys
from pathlib import Path
from safetensors.torch import save_file

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.unified_module import MultiModalAdapterModel

def convert_ckpt(ckpt_path, output_dir=None):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} does not exist.")
        return

    if output_dir is None:
        output_dir = ckpt_path.parent / "exported_model"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from {ckpt_path}...")
    # Load model from checkpoint (restores loading logic)
    # strict=False allows ignoring keys if needed, but here we want integrity
    model = MultiModalAdapterModel.load_from_checkpoint(ckpt_path)
    model.eval()
    
    print("Saving model weights...")
    
    # 1. Save state dict as safetensors (Recommended)
    state_dict = model.state_dict()
    save_file(state_dict, output_dir / "model.safetensors")
    print(f"Saved {output_dir / 'model.safetensors'}")
    
    # 2. Save as standard PyTorch .bin (Optional)
    # torch.save(state_dict, output_dir / "pytorch_model.bin")
    
    # 3. Extract and Save Config
    print("Extracting configuration...")
    import json
    
    # Access hyperparameters saved by save_hyperparameters()
    config = model.hparams
    
    # Convert OmegaConf/DictConfig to standard dict if needed
    if hasattr(config, "keys"): # Basic check
        # Convert non-serializable objects to strings/dicts
        def make_serializable(obj):
            if hasattr(obj, "to_container"): # OmegaConf
                from omegaconf import OmegaConf
                return OmegaConf.to_container(obj, resolve=True)
            return obj
            
        serializable_config = make_serializable(config)
        
        # Save as config.json
        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2, default=str)
        print(f"Saved config to {config_path}")
    else:
        print("[WARN] Could not extract hyperparameters to config.json")
    
    print(f"Conversion complete! Files saved in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_ckpt.py <path_to_ckpt>")
    else:
        convert_ckpt(sys.argv[1])
