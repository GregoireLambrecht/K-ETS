import pandas as pd
import subprocess
import sys
import os

def run_batch_from_csv(csv_path):
    # 1. Setup absolute project root
    project_root = os.path.abspath(os.getcwd())
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 2. Setup environment for imports
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = project_root + os.pathsep + my_env.get("PYTHONPATH", "")
    
    # 3. Load the CSV
    df = pd.read_csv(csv_path)
    
    if 'path' not in df.columns:
        print("Error: CSV must have a 'path' column.")
        return

    print(f"Project Root: {project_root}")
    print(f"Running {len(df)} scenarios from {csv_path}")

    for file_path in df['path']:
        # Strip any potential whitespace or hidden characters
        clean_path = str(file_path).strip()
        
        # Verify the file exists relative to project_root before calling subprocess
        if not os.path.exists(clean_path):
            print(f"Skipping: {clean_path} (File not found)")
            continue

        print(f"\nProcessing: {clean_path}")
        
        # 4. Command matches the --scenario_path requirement in main.py
        cmd = [
            sys.executable, "-m", "main", 
            "--scenario_path", clean_path
        ]
        
        try:
            # Use project_root as the execution context
            subprocess.run(cmd, check=True, env=my_env, cwd=project_root)
            print(f"Success: {clean_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error: {clean_path} failed with exit code {e.returncode}")

if __name__ == "__main__":
    run_batch_from_csv("BAU.csv")