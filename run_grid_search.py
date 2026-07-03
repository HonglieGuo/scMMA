import subprocess
import time
import sys
import itertools
from datetime import datetime

# ================= 1. Backbone Configuration =================
# Each backbone binds its required extra parameters via a dict.
# Comment out entries to skip specific backbones.
BACKBONE_CONFIGS = {
    "geneformer": ["model/backbone=geneformer"],
    # "scgpt":      ["model/backbone=scgpt",  "data=multimodal_scgpt"],
}

# ================= 2. Base Configuration ====================
# Backbone-related args are in BACKBONE_CONFIGS above; common params stay here.
BASE_OVERRIDES = [
    # -------------------------------------------------------------------------
    # [2] Pooling Mode (Embedding Extraction Strategy)
    # -------------------------------------------------------------------------
    "model.pooling_mode=mean",            # Average pooling over all tokens
    # "model.pooling_mode=cls",           # Use first token (CLS) as feature

    # -------------------------------------------------------------------------
    # [3] Fusion Architecture
    # -------------------------------------------------------------------------
    "model/arch=crossattn",               # [Recommended] Cross-Attention fusion
    # "model/arch=projection",            # Linear projection + concatenation

    # -------------------------------------------------------------------------
    # [4] Dataset & Task
    # -------------------------------------------------------------------------
    # (Fully automated: the script auto-selects config based on ATAC/ADT in path)
    "data.data_dir=datasets/h5ad/BMMC-p10/RNA+ATAC",
    # "data.data_dir=datasets/h5ad/BMMC-p10/RNA+ADT",
    # "data.data_dir=datasets/h5ad/D22/RNA+ADT+ATAC",

    # NOTE: model/atac_encoder is conditionally injected in main() based on dataset type
    # Pure RNA+ADT datasets do not need ATAC encoder; forcing it causes runtime errors

    # -------------------------------------------------------------------------
    # [5] Training Hyperparameters
    # -------------------------------------------------------------------------
    # NOTE: data.batch_size and trainer.accumulate_grad_batches moved to PARAM_GRID for grid search
    "data.batch_size=24",
    "data.max_seq_len=1024",
    "model.learning_rate=0.0001",
    "trainer.devices=1",             # Number of GPUs; use -1 for all available GPUs
    "trainer.max_epochs=30",
    "trainer.accumulate_grad_batches=4",


    # -------------------------------------------------------------------------
    # [6] LoRA Fine-tuning — r=16 is globally optimal, fixed (not searched)
    # -------------------------------------------------------------------------
    "model.lora_config.r=16",

    # -------------------------------------------------------------------------
    # [7] Loss Weights (Multi-objective)
    # -------------------------------------------------------------------------
    "model.mtp_weight=1.0",
    "model.cls_weight=1.0",
    "model.contrastive_weight=1.0",
    "model.supcon_weight=1.0",
    "model.contrastive_temperature=0.07",

    # -------------------------------------------------------------------------
    # [8] Execution Preferences
    # -------------------------------------------------------------------------
    "model.weight_tying=true",
    "model.fusion_module.use_gating=true",

    # -------------------------------------------------------------------------
    # [9] Feature Projection Head — reduces dimensionality to improve clustering
    # -------------------------------------------------------------------------
    "model.projection_head_type=mlp",      # mlp or linear
    "model.projection_dim=128",            # Target dimension (from backbone hidden_dim to 128)
    "model.eval_apply_projection=true"     # Whether to use projected features during evaluation
]

# ================= 2. Search Space =================
# Define all parameters for grid search (Cartesian product).
# Format: "param_name": [list_of_values]
# To explore new dimensions (e.g., LR, Temperature), add entries here.
PARAM_GRID = {
    # "data.batch_size": [12, 24],
    # "trainer.accumulate_grad_batches": [1, 2, 4],
}

# ================= 3. Execution Logic (no modification needed) =================

def generate_experiments(grid):
    """Generate all parameter combinations."""
    keys = grid.keys()
    values = grid.values()
    # Compute Cartesian product
    combinations = list(itertools.product(*values))
    
    experiments = []
    for combo in combinations:
        # Map parameter names to values
        params_dict = dict(zip(keys, combo))
        
        # Auto-generate a readable experiment name
        name_parts = []
        
        # Helper for short keys
        if "data.batch_size" in params_dict:
            name_parts.append(f"BS{params_dict['data.batch_size']}")
        if "trainer.max_epochs" in params_dict:
            name_parts.append(f"E{params_dict['trainer.max_epochs']}")
        if "trainer.accumulate_grad_batches" in params_dict:
            name_parts.append(f"Acc{params_dict['trainer.accumulate_grad_batches']}")
        if "model.learning_rate" in params_dict:
            lr_val = params_dict['model.learning_rate']
            name_parts.append(f"LR{lr_val}")
        if "model.lora_config.r" in params_dict:
            name_parts.append(f"R{params_dict['model.lora_config.r']}")
        if "model.supcon_weight" in params_dict:
            name_parts.append(f"Sup{params_dict['model.supcon_weight']}")
        if "model.contrastive_temperature" in params_dict:
            name_parts.append(f"T{params_dict['model.contrastive_temperature']}")
            
        exp_name = "_".join(name_parts)
        if not exp_name:
            exp_name = "GridSearch_Run"
            
        # Convert to Hydra command-line format
        cmd_args = [f"{k}={v}" for k, v in params_dict.items()]
        
        experiments.append({
            "name": exp_name,
            "args": cmd_args
        })
        
    return experiments

def run_command(command):
    print(f"\n{'='*60}")
    print(f"\U0001f680 Running: {' '.join(command)}")
    print(f"{'='*60}\n")
    try:
        subprocess.run(command, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\u274c Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\U0001f6d1 Interrupted by user.")
        sys.exit(0)

def main():
    experiments = generate_experiments(PARAM_GRID)
    per_backbone = len(experiments)
    total = per_backbone * len(BACKBONE_CONFIGS)

    print(f"\n\U0001f50d Grid Search Initialized \u2014 Multi-Backbone Mode.")
    print(f"\U0001f9ec Backbones : {list(BACKBONE_CONFIGS.keys())}")
    print(f"\U0001f4ca Configs/Backbone: {per_backbone}  |  Total runs: {total}")
    print("\U0001f4cb Scheduled Experiments (per backbone):")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp['name']}")

    print(f"\nEstimated time (assuming 40min/run): {total * 40 / 60:.1f} hours.")
    print("Starting in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)

    run_idx = 0
    for backbone_name, backbone_args in BACKBONE_CONFIGS.items():
        print(f"\n\n{'#'*60}")
        print(f"# Backbone: {backbone_name.upper()}  ({per_backbone} runs)")
        print(f"{'#'*60}")

        for i, exp in enumerate(experiments):
            run_idx += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Prefix experiment name with backbone identifier for CSV differentiation
            full_name = f"{backbone_name}_{exp['name']}"
            print(f"\n\n>>> [{current_time}] Run {run_idx}/{total}: {full_name}")

            # backbone_args injected first; BASE_OVERRIDES no longer contain backbone lines
            # sys.argv[1:] appended last, leveraging Hydra's right-overrides-left mechanism
            all_args = backbone_args + BASE_OVERRIDES + exp['args'] + sys.argv[1:]

            # Auto-detect which base training config to use + conditionally inject ATAC encoder
            config_name = "train_atac"  # Default fallback
            has_atac = False
            for arg in all_args:
                if arg.startswith("data.data_dir="):
                    arg_upper = arg.upper()
                    if "ADT" in arg_upper and "ATAC" in arg_upper:
                        config_name = "train_trimodal"
                        has_atac = True
                    elif "ADT" in arg_upper:
                        config_name = "train_adt"
                        has_atac = False
                    elif "ATAC" in arg_upper:
                        config_name = "train_atac"
                        has_atac = True

            # Only inject ATAC encoder when the dataset contains ATAC modality
            # Pure RNA+ADT datasets do not need ATAC encoder (train_adt.yaml sets atac: null)
            modality_args = ["model/atac_encoder=conv1d"] if has_atac else []

            cmd = [
                sys.executable, "scripts/train.py",
                "--config-name", config_name,
                f"logger.name={full_name}"
            ] + all_args + modality_args

            success = run_command(cmd)

            if not success:
                print(f"\u26a0\ufe0f  {full_name} Failed! Moving to next...")

            time.sleep(3)

if __name__ == "__main__":
    main()
