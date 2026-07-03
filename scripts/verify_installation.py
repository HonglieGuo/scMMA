"""
Verify scMMA installation.

Checks:
1. Core libraries (Torch, Lightning, Hydra).
2. Domain-specific libraries (Scanpy, Muon).
3. Foundation Model Dependencies (Transformers, PEFT).
4. Optional Research Dependencies (scGPT, scmmib).
5. CUDA availability.
"""

import sys
import importlib

def check_import(module_name, display_name=None, optional=False):
    if display_name is None:
        display_name = module_name
    try:
        importlib.import_module(module_name)
        print(f"[OK] {display_name} installed.")
        return True
    except ImportError:
        if optional:
            print(f"[WARN] {display_name} NOT found (Optional, OK if using fallback).")
            return True
        else:
            print(f"[FAIL] {display_name} NOT found.")
            return False

def main():
    print("Verifying scMMA Environment...\n")
    
    all_ok = True
    
    # Core
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("lightning.pytorch", "PyTorch Lightning")
    all_ok &= check_import("hydra", "Hydra (hydra-core)")
    
    print("\n[Data Processing]")
    all_ok &= check_import("scanpy", "Scanpy")
    all_ok &= check_import("anndata", "AnnData")
    all_ok &= check_import("muon", "Muon")
    
    # Geneformer Dependencies
    all_ok &= check_import("transformers", "Transformers (Geneformer)")
    all_ok &= check_import("peft", "PEFT")
    
    # Optional Research
    check_import("scgpt", "scGPT", optional=True)
    check_import("scmmib", "scmmib", optional=True)
    
    print("\nChecking Hardware...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] CUDA NOT available. Training will be slow.")
    except:
        pass

    if all_ok:
        print("\n✅ Critical dependencies are ready for scMMA (Geneformer)!")
    else:
        print("\n❌ Some CRITICAL dependencies are missing. Please install dependencies.")

if __name__ == "__main__":
    main()
