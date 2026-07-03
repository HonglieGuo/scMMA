import os
import argparse
import datetime
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_file = os.path.join(project_root, "baselines", "baseline_experiment_log.csv")
    
    # Read metrics from eval_dir/metrics.json if it exists
    metrics_file = os.path.join(args.eval_dir, "metrics.json")
    metrics = {}
    status = "Failed"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        status = "Success"

    # Also read metrics from eval_dir/metrics.csv if it exists (from eval_embedding.py)
    csv_metrics_file = os.path.join(args.eval_dir, "metrics.csv")
    if os.path.exists(csv_metrics_file):
        try:
            df_csv = pd.read_csv(csv_metrics_file)
            if not df_csv.empty:
                csv_dict = df_csv.iloc[0].to_dict()
                metrics.update(csv_dict)
                status = "Success"
        except Exception as e:
            print(f"⚠️ Error reading metrics.csv: {e}")
            
    dataset_info = {
        "D1": ("RNA+ADT", "Vertical"),
        "D15": ("RNA+ATAC", "Vertical"),
        "D18": ("RNA+ATAC", "Vertical"),
        "D22": ("RNA+ADT+ATAC", "Vertical"),
        "D53": ("RNA+ADT", "Cross-batch"),
        "D54": ("RNA+ADT", "Cross-batch"),
        "D56": ("RNA+ATAC", "Cross-batch"),
        "D59": ("RNA+ADT+ATAC", "Cross-batch")
    }
    ds_modality, ds_task = dataset_info.get(args.dataset, ("", ""))

    record = {
        "Times": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Method": args.method,
        "Method_Supported_Modalities": args.modality,
        "Dataset": args.dataset,
        "Dataset_Modalities": ds_modality,
        "Dataset_Task": ds_task,
    }
    
    metric_keys = ["Leiden_Res", "NMI", "ARI", "ASW_label", "cLISI", "Batch_ASW", "iLISI", "ASW_batch_raw", "Overall", "FOSCTTM", "Match@5", "LTA"]
    for k in metric_keys:
        record[k] = metrics.get(k, "")
        
    record["Run_Time_Sec"] = metrics.get("time", "")
    record["Embedding_Dim"] = metrics.get("Embedding_Dim", "")
    record["Actual_Epochs"] = metrics.get("Actual_Epochs", "")
    record["Peak_Mem_GB"] = metrics.get("Peak_Mem_GB", "")
    record["Params_M"] = metrics.get("Params_M", "")
    record["Conda_Env"] = "scmulti"
    
    df_new = pd.DataFrame([record])
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        df_old = pd.read_csv(log_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(log_file, index=False)
    else:
        df_new.to_csv(log_file, index=False)
        
    print(f"📝 Result logged to {log_file}")

if __name__ == "__main__":
    main()
