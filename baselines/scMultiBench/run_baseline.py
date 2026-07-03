import os
import sys
import time
import argparse
import subprocess
import datetime
import pandas as pd

# Get the absolute path to the project root (assuming script is in scMMA/baseline/scMultiBench/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCMULTIBENCH_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(SCMULTIBENCH_DIR, "..", "experiment_output"))

# =========================================================================
# 1. Dataset Registry
# =========================================================================
DATASET_REGISTRY = {
    "D1":  {"modality": "RNA+ADT", "task": "vertical", "n_batches": 1, "dir": "datasets/h5/D1"},
    "D15": {"modality": "RNA+ATAC", "task": "vertical", "n_batches": 1, "dir": "datasets/h5/D15"},
    "D18": {"modality": "RNA+ATAC", "task": "vertical", "n_batches": 1, "dir": "datasets/h5/D18"},
    "D22": {"modality": "RNA+ADT+ATAC", "task": "vertical", "n_batches": 1, "dir": "datasets/h5/D22"},
    "D53": {"modality": "RNA+ADT", "task": "cross-batch", "n_batches": 2, "dir": "datasets/h5/D53"},
    "D54": {"modality": "RNA+ADT", "task": "cross-batch", "n_batches": 12, "dir": "datasets/h5/D54"},
    "D56": {"modality": "RNA+ATAC", "task": "cross-batch", "n_batches": 13, "dir": "datasets/h5/D56"},
    "D59": {"modality": "RNA+ADT+ATAC", "task": "cross-batch", "n_batches": 2, "dir": "datasets/h5/D59"},
}

def get_dataset_files(dataset_name):
    ds_info = DATASET_REGISTRY.get(dataset_name)
    if not ds_info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    base_dir = os.path.join(PROJECT_ROOT, ds_info["dir"])
    n = ds_info["n_batches"]
    mod = ds_info["modality"]
    
    files = {"rna": [], "mod2": [], "mod3": [], "cty": []}
    
    is_atac = "ATAC" in mod
    is_adt = "ADT" in mod
    is_trimodal = is_atac and is_adt
    
    # Handle specific naming convention
    # D59 ATAC is named peak{n}.h5
    atac_prefix = "peak" if dataset_name == "D59" else "atac"
    
    if n == 1:
        files["rna"].append(os.path.join(base_dir, "rna.h5"))
        files["cty"].append(os.path.join(base_dir, "cty.csv"))
        if is_trimodal:
            files["mod2"].append(os.path.join(base_dir, "adt.h5"))
            files["mod3"].append(os.path.join(base_dir, f"{atac_prefix}.h5"))
        elif is_adt:
            files["mod2"].append(os.path.join(base_dir, "adt.h5"))
        elif is_atac:
            files["mod2"].append(os.path.join(base_dir, f"{atac_prefix}.h5"))
    else:
        for i in range(1, n + 1):
            files["rna"].append(os.path.join(base_dir, f"rna{i}.h5"))
            files["cty"].append(os.path.join(base_dir, f"cty{i}.csv"))
            if is_trimodal:
                files["mod2"].append(os.path.join(base_dir, f"adt{i}.h5"))
                files["mod3"].append(os.path.join(base_dir, f"{atac_prefix}{i}.h5"))
            elif is_adt:
                files["mod2"].append(os.path.join(base_dir, f"adt{i}.h5"))
            elif is_atac:
                files["mod2"].append(os.path.join(base_dir, f"{atac_prefix}{i}.h5"))
                
    return files, ds_info

# =========================================================================
# 2. Argument Builders
# =========================================================================
def generic_python_builder(script_path, save_dir, files, ds_info):
    """Builds arguments for scripts using --path1, --path2, --save_path convention."""
    args = [sys.executable, os.path.join(SCMULTIBENCH_DIR, script_path)]
    
    args.extend(["--path1"] + files["rna"])
    if len(files["mod2"]) > 0:
        args.extend(["--path2"] + files["mod2"])
    if len(files["mod3"]) > 0:
        args.extend(["--path3"] + files["mod3"])
        
    args.extend(["--save_path", save_dir])
    return args

def r_script_builder(script_path, save_dir, files, ds_info):
    """Builder for R scripts that expect: Rscript script.Rmd rna adt atac save_dir"""
    args = ["Rscript", os.path.join(SCMULTIBENCH_DIR, script_path)]
    
    # RNA path
    args.append(files["rna"][0] if len(files["rna"]) > 0 else "NULL")
    
    # ADT path (mod2 if ADT)
    if "ADT" in ds_info["modality"] and len(files["mod2"]) > 0:
        args.append(files["mod2"][0])
    else:
        args.append("NULL")
        
    # ATAC path (mod2 if ATAC only, mod3 if Trimodal)
    if "ATAC" in ds_info["modality"]:
        if len(files["mod3"]) > 0:
            args.append(files["mod3"][0])
        elif len(files["mod2"]) > 0:
            args.append(files["mod2"][0])
        else:
            args.append("NULL")
    else:
        args.append("NULL")
        
    # Save path (must end with slash for some R scripts)
    save_path_with_slash = save_dir if save_dir.endswith('/') else save_dir + '/'
    args.append(save_path_with_slash)
    
    return args

def seurat_builder(script_path, save_dir, files, ds_info):
    args = ["Rscript", os.path.join(SCMULTIBENCH_DIR, script_path)]
    args.append(",".join(files["rna"]) if len(files["rna"]) > 0 else "NULL")
    
    if "ADT" in ds_info["modality"] and len(files["mod2"]) > 0:
        args.append(",".join(files["mod2"]))
    else:
        args.append("NULL")
        
    if "ATAC" in ds_info["modality"]:
        if len(files["mod3"]) > 0:
            args.append(",".join(files["mod3"]))
        elif len(files["mod2"]) > 0:
            args.append(",".join(files["mod2"]))
        else:
            args.append("NULL")
    else:
        args.append("NULL")
        
    save_path_with_slash = save_dir if save_dir.endswith('/') else save_dir + '/'
    args.append(save_path_with_slash)
    return args

def smile_builder(script_path, save_dir, files, ds_info):
    args = [sys.executable, os.path.join(SCMULTIBENCH_DIR, script_path)]
    
    args.extend(["--ref_path1", files["rna"][0]])
    query_rna = files["rna"][1:] if len(files["rna"]) > 1 else [files["rna"][0]]
    args.extend(["--query_path1"] + query_rna)
    
    if len(files["mod2"]) > 0:
        args.extend(["--ref_path2", files["mod2"][0]])
        query_mod2 = files["mod2"][1:] if len(files["mod2"]) > 1 else [files["mod2"][0]]
        args.extend(["--query_path2"] + query_mod2)
        
    args.extend(["--task", ds_info.get("task", "mosaic")])
    args.extend(["--save_path", save_dir])
    return args

def mofa2_builder(script_path, save_dir, files, ds_info):
    """Builder for MOFA2 Rscript that expects: Rscript script.Rmd [paths...] batch_num num save_dir"""
    args = ["Rscript", os.path.join(SCMULTIBENCH_DIR, script_path)]
    
    modality = ds_info["modality"]
    is_trimodal = modality == "RNA+ADT+ATAC"
    num = 3 if is_trimodal else 2
    
    batch_num = ds_info.get("n_batches", 1)
    
    args.extend(files["rna"])
    args.extend(files["mod2"])
    if is_trimodal:
        args.extend(files["mod3"])
        
    args.append(str(batch_num))
    args.append(str(num))
    
    save_path_with_slash = save_dir if save_dir.endswith('/') else save_dir + '/'
    args.append(save_path_with_slash)
    
    return args

def inmf_builder(script_path, save_dir, files, ds_info):
    args = ["Rscript", os.path.join(SCMULTIBENCH_DIR, script_path)]
    args.extend(files["rna"])
    if len(files["mod2"]) > 0:
        args.extend(files["mod2"])
    if len(files["mod3"]) > 0:
        args.extend(files["mod3"])
    save_path_with_slash = save_dir if save_dir.endswith('/') else save_dir + '/'
    args.append(save_path_with_slash)
    return args

def uinmf_builder(script_path, save_dir, files, ds_info):
    args = ["Rscript", os.path.join(SCMULTIBENCH_DIR, script_path)]
    args.append(files["rna"][0] if len(files["rna"]) > 0 else "NULL")
    args.append(files["mod2"][0] if len(files["mod2"]) > 0 else "NULL")
    save_path_with_slash = save_dir if save_dir.endswith('/') else save_dir + '/'
    args.append(save_path_with_slash)
    return args

# =========================================================================
# 3. Method Registry
# =========================================================================
METHOD_REGISTRY = {
    "MOFA2": {
        "script": "tools_scripts/MOFA2/main_MOFA2.Rmd",
        "language": "R",
        "conda_env": "scmulti",
        "modalities": ["RNA+ADT", "RNA+ATAC"],
        "tasks": ["vertical"],
        "builder": mofa2_builder
    },
    "totalVI": {
        "script": "tools_scripts/totalVI/main_totalVI.py",
        "language": "python",
        "conda_env": "scmulti",
        "modalities": ["RNA+ADT"],
        "tasks": ["vertical", "cross-batch"],
        "builder": generic_python_builder
    },
    "Multigrate": {
        # Note: Multigrate has different scripts for paired/mosaic/etc. We default to paired.
        "script": "tools_scripts/Multigrate/main_Multigrate_paired_integration.py",
        "language": "python",
        "conda_env": "scmulti",
        "modalities": ["RNA+ADT", "RNA+ATAC"],
        "tasks": ["vertical", "cross-batch"],
        "builder": generic_python_builder
    },
    "scMoMaT": {
        "script": "tools_scripts/scMoMaT/main_scMoMaT.py",
        "language": "python",
        "conda_env": "scmulti",
        "modalities": ["RNA+ADT", "RNA+ATAC"],
        "tasks": ["vertical", "cross-batch"],
        "builder": generic_python_builder
    },
    "Seurat_v4": {
        "script": "tools_scripts/Seurat_v4/main_Seurat_v4.R",
        "language": "R",
        "conda_env": "scmulti",
        "modalities": ["RNA+ADT", "RNA+ATAC"],
        "tasks": ["vertical", "cross-batch"],
        "builder": seurat_builder
    },
    "MultiVI": {
        "script": "tools_scripts/MultiVI/main_MultiVI.py",
        "language": "python",
        "conda_env": "scmulti",
        "modalities": ["RNA+ATAC"],
        "tasks": ["vertical", "cross-batch"],
        "builder": generic_python_builder
    },
    "iNMF": {
        "script": "tools_scripts/iNMF/main_iNMF.R",
        "language": "R",
        "conda_env": "scmulti",
        "modalities": ["RNA+ATAC", "RNA+ADT"],
        "tasks": ["vertical", "cross-batch", "diagonal"],
        "builder": inmf_builder
    },
    "UINMF": {
        "script": "tools_scripts/UINMF/main_UINMF.R",
        "language": "R",
        "conda_env": "scmulti",
        "modalities": ["RNA+ADT", "RNA+ATAC"],
        "tasks": ["vertical", "cross-batch"],
        "builder": inmf_builder
    },
    "SMILE": {
        "script": "tools_scripts/SMILE/main_SMILE.py",
        "language": "python",
        "conda_env": "scmulti",
        "modalities": ["RNA+ADT", "RNA+ATAC"],
        "tasks": ["vertical", "cross-batch"],
        "builder": smile_builder
    },
}

# Add default python methods to registry assuming they use the generic format
_generic_methods = ["sciPENN", "scMSI", "Matilda", "MIRA", "GLUE", "uniPort", "MultiMAP"]
for m in _generic_methods:
    if m not in METHOD_REGISTRY:
        METHOD_REGISTRY[m] = {
            "script": f"tools_scripts/{m}/main_{m}.py" if os.path.exists(os.path.join(SCMULTIBENCH_DIR, f"tools_scripts/{m}/main_{m}.py")) else f"tools_scripts/{m}/{m}_human.py",
            "language": "python",
            "conda_env": "scmulti",
            "modalities": ["RNA+ADT", "RNA+ATAC"],
            "tasks": ["vertical", "cross-batch"],
            "builder": generic_python_builder
        }

# =========================================================================
# 4. CSV Logging (Matching scMMA format)
# =========================================================================
def log_result(method_name, dataset_name, ds_info, method_info, env, run_time, status, metrics=None, err_msg=None, embedding_dim=None, embedding_path=None):
    if "Fail" in status and "OOM" not in status:
        print(f"⚠️ Status is {status}. Skipping CSV logging.")
        return
        
    log_file = os.path.join(SCMULTIBENCH_DIR, "..", "baseline_experiment_log.csv")
    
    record = {
        "Times": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Method": method_name,
        "Method_Supported_Modalities": " / ".join(method_info.get("modalities", [])) if method_info else "",
        "Dataset": dataset_name,
        "Dataset_Modalities": ds_info["modality"],
        "Dataset_Task": ds_info["task"].capitalize(),
    }
    
    # Metrics exactly matching logger.py
    metric_keys = ["Leiden_Res", "NMI", "ARI", "ASW_label", "cLISI", "Batch_ASW", "iLISI", "ASW_batch_raw", "Overall", "FOSCTTM", "Match@5", "LTA"]
    if metrics:
        for k in metric_keys:
            record[k] = metrics.get(k, "")
    else:
        for k in metric_keys:
            record[k] = ""
            
    if "OOM" in status:
        for k in metric_keys: record[k] = "OOM"
        record["Run_Time_Sec"] = "OOM"
        record["Peak_Mem_GB"] = "OOM"
        record["Params_M"] = "OOM"
    else:
        record["Run_Time_Sec"] = f"{run_time:.1f}" if run_time else ""
        record["Peak_Mem_GB"] = metrics.get("Peak_Mem_GB", "") if metrics else ""
        record["Params_M"] = metrics.get("Params_M", "") if metrics else ""
        
    record["Embedding_Dim"] = embedding_dim if embedding_dim else ""
    record["Actual_Epochs"] = ""
    record["Conda_Env"] = env
    
    df_new = pd.DataFrame([record])
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        df_old = pd.read_csv(log_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(log_file, index=False)
    else:
        df_new.to_csv(log_file, index=False)
        
    print(f"[INFO] Result logged to {log_file}")

# =========================================================================
# 5. Main Execution
# =========================================================================
def main():
    parser = argparse.ArgumentParser("run_baseline")
    parser.add_argument("--method", nargs="+", help="Methods to run")
    parser.add_argument("--dataset", nargs="+", help="Datasets to run (e.g. D1 D53)")
    parser.add_argument("--conda-env", type=str, help="Override conda environment")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing embeddings")
    parser.add_argument("--list-methods", action="store_true", help="List available methods")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    args = parser.parse_args()
    
    if args.list_methods:
        print("Available Methods:")
        for m, info in METHOD_REGISTRY.items():
            print(f"  - {m} (Lang: {info['language']}, Mods: {info['modalities']})")
        return
        
    if args.list_datasets:
        print("Available Datasets:")
        for d, info in DATASET_REGISTRY.items():
            print(f"  - {d} (Mod: {info['modality']}, Batches: {info['n_batches']})")
        return
        
    if not args.method or not args.dataset:
        parser.error("Please specify --method and --dataset, or use --list-methods/--list-datasets")
        
    for method_name in args.method:
        if method_name not in METHOD_REGISTRY:
            print(f"❌ Unknown method: {method_name}")
            continue
            
        method_info = METHOD_REGISTRY[method_name]
        
        for dataset_name in args.dataset:
            try:
                files, ds_info = get_dataset_files(dataset_name)
            except ValueError as e:
                print(f"❌ {e}")
                continue
                
            # Check modality compatibility
            if not any(mod in ds_info["modality"] for mod in method_info["modalities"]):
                print(f"⚠️ Skipping {method_name} on {dataset_name}: Modality mismatch.")
                continue
                
            print(f"\n{'='*60}")
            print(f"🚀 Running {method_name} on {dataset_name}")
            print(f"{'='*60}")
            
            save_dir = os.path.join(OUTPUT_DIR, method_name, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            
            env = args.conda_env or method_info["conda_env"]
            run_time = None
            embedding_path = os.path.join(save_dir, "embedding.h5")
            
            if not args.eval_only:
                # Build command
                cmd = method_info["builder"](method_info["script"], save_dir, files, ds_info)
                
                # We assume conda environment activation is handled externally (e.g. via run_scmultibench_baslines.sh)
                # or we just run the python executable from the current env.
                # For this script, we'll just run it. The bash script handles `conda activate`.
                
                print(f"Executing: {cmd}")
                start_time = time.time()
                try:
                    if isinstance(cmd, list):
                        # Use time -v for Seurat or R scripts to catch max RSS
                        if cmd[0] == "Rscript":
                            cmd = ["/usr/bin/time", "-v"] + cmd
                        result = subprocess.run(cmd, cwd=SCMULTIBENCH_DIR, capture_output=True, text=True)
                    else:
                        result = subprocess.run(cmd, shell=True, cwd=SCMULTIBENCH_DIR, capture_output=True, text=True)
                        
                    print(result.stdout)
                    
                    if result.returncode != 0:
                        print(result.stderr, file=sys.stderr)
                        stderr_lower = result.stderr.lower()
                        if result.returncode in [137, 139] or "cuda out of memory" in stderr_lower or "std::bad_alloc" in stderr_lower or "killed" in stderr_lower:
                            print(f"❌ {method_name} failed with OOM (Out of Memory).")
                            log_result(method_name, dataset_name, ds_info, method_info, env, None, "OOM", err_msg="Out of Memory")
                            continue
                        else:
                            result.check_returncode()
                            
                    run_time = time.time() - start_time
                    print(f"✅ {method_name} completed in {run_time:.1f} seconds.")
                    
                    # Parse time -v output for Peak Mem if available
                    peak_mem_gb = None
                    if "Maximum resident set size (kbytes):" in result.stderr:
                        for line in result.stderr.splitlines():
                            if "Maximum resident set size" in line:
                                kb = int(line.split(":")[-1].strip())
                                peak_mem_gb = kb / (1024 * 1024)
                                print(f"📊 Extracted Peak Mem from /usr/bin/time: {peak_mem_gb:.2f} GB")
                                break
                                
                    # Write it to a temp metrics_ext.json so eval step can read it, or just keep it
                    import json
                    ext_data = {}
                    if peak_mem_gb:
                        ext_data["Peak_Mem_GB"] = f"{peak_mem_gb:.2f} (CPU)"
                    
                    # Also try to read if the python script wrote its own metrics_ext.json
                    ext_file = os.path.join(save_dir, "metrics_ext.json")
                    if os.path.exists(ext_file):
                        with open(ext_file, 'r') as f:
                            ext_data.update(json.load(f))
                    
                    # Write back so eval step can merge it
                    with open(ext_file, 'w') as f:
                        json.dump(ext_data, f)
                        
                except subprocess.CalledProcessError as e:
                    print(f"❌ {method_name} failed: {e}")
                    log_result(method_name, dataset_name, ds_info, method_info, env, None, "Failed", err_msg=str(e))
                    continue
            else:
                print("Skipping execution (--eval-only).")
                
            if not args.skip_eval:
                if not os.path.exists(embedding_path):
                    print(f"❌ Embedding not found at {embedding_path}. Cannot evaluate.")
                    log_result(method_name, dataset_name, ds_info, method_info, env, run_time, "Failed", err_msg="Embedding not found")
                    continue
                    
                print("Running Evaluation...")
                eval_script = os.path.join(SCMULTIBENCH_DIR, "eval_embedding.py")
                
                # Import dynamically to avoid subprocess overhead if possible, but subprocess is safer for memory
                from eval_embedding import evaluate_embedding
                try:
                    metrics = evaluate_embedding([embedding_path], files["cty"], save_path=save_dir)
                    
                    # Merge with ext_data
                    ext_file = os.path.join(save_dir, "metrics_ext.json")
                    if os.path.exists(ext_file):
                        import json
                        with open(ext_file, 'r') as f:
                            metrics.update(json.load(f))
                            
                    log_result(method_name, dataset_name, ds_info, method_info, env, run_time, "Success", 
                              metrics=metrics, embedding_dim=metrics.get('Embedding_Dim'), embedding_path=embedding_path)
                except Exception as e:
                    print(f"❌ Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    log_result(method_name, dataset_name, ds_info, method_info, env, run_time, "Eval Failed", err_msg=str(e), embedding_path=embedding_path)
            else:
                log_result(method_name, dataset_name, ds_info, method_info, env, run_time, "Success (No Eval)", embedding_path=embedding_path)

if __name__ == "__main__":
    main()
