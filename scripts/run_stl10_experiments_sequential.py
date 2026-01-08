"""Script to run all STL-10 experiments one by one."""

import subprocess
import sys
from pathlib import Path
import time

def run_experiment(run_num):
    script_name = f"run_grid_exp_{run_num:03d}.py"
    script_path = Path(__file__).parent.parent / "experiments" / script_name
    
    print(f"\n" + "="*80)
    print(f"                    RUNNING EXPERIMENT: {script_name}                     ".center(80))
    print("="*80)
    
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Experiment {script_name} produced stderr:\n{result.stderr}")
        print(f"\n--- Experiment run_{run_num:03d} completed successfully! ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- Experiment run_{run_num:03d} failed with error: ---")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"--- Experiment run_{run_num:03d} failed with unexpected error: {e} ---")
        return False

def main():
    # Define the range of STL-10 experiment numbers
    stl10_run_numbers = list(range(70, 87))  # run_070 to run_086

    print("="*80)
    print("                  STL-10 EXPERIMENTS - SEQUENTIAL EXECUTION                   ".center(80))
    print("="*80)
    print(f"Total experiments to run: {len(stl10_run_numbers)}")
    print(f"Experiments: run_{stl10_run_numbers[0]:03d} through run_{stl10_run_numbers[-1]:03d}")
    print("="*80)

    failed_experiments = []

    for i, run_num in enumerate(stl10_run_numbers):
        print(f"\n[{i+1}/{len(stl10_run_numbers)}] Starting run_{run_num:03d}...")
        success = run_experiment(run_num)
        if not success:
            failed_experiments.append(run_num)
        time.sleep(1)  # Small delay between runs

    print("\n" + "="*80)
    print("                  STL-10 EXPERIMENTS - EXECUTION SUMMARY                    ".center(80))
    print("="*80)
    if not failed_experiments:
        print("All STL-10 experiments completed successfully!")
    else:
        print(f"The following STL-10 experiments failed: {failed_experiments}")
    print("="*80)

if __name__ == "__main__":
    main()


