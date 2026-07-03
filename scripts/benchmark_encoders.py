import subprocess
import sys
import os

def run_benchmark():
    print("🚀 Starting ATAC Encoder Benchmark (5 Epochs each)...")
    
    # Common arguments
    # Use sys.executable to ensure we use the same Python interpreter (conda env)
    base_cmd = [sys.executable, "scripts/train.py", "trainer.max_epochs=5"]
    
    experiments = [
        {
            "name": "Chromosomal Encoder (Benchmark)",
            "args": ["model/atac_encoder=chromosomal", "logger.name=bench_chromosomal"]
        },
        {
            "name": "Conv1D Encoder (Benchmark)",
            "args": ["model/atac_encoder=conv1d", "logger.name=bench_conv1d"]
        }
    ]
    
    for exp in experiments:
        print(f"\n==================================================")
        print(f"▶️ Running: {exp['name']}")
        print(f"==================================================")
        
        cmd = base_cmd + exp["args"]
        cmd_str = " ".join(cmd)
        print(f"Command: {cmd_str}")
        
        try:
            # Run process and wait for completion
            # We use shell=False for list of args
            subprocess.run(cmd, check=True)
            print(f"✅ Finished: {exp['name']}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {exp['name']}")
            print(f"Error: {e}")
            # Ask user if they want to continue? No, usually in benchmark we stop or logging might be broken.
            # But let's try to continue to the next one if it's just a runtime error? 
            # Ideally we want to see both results.
            print("Continuing to next experiment...")
            
    print("\n🎉 Benchmark Batch Completed.")

if __name__ == "__main__":
    run_benchmark()
