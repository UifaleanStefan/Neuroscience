"""Script to run all CIFAR-10 experiments sequentially, one by one."""

import subprocess
import sys
from pathlib import Path

# List of CIFAR-10 experiment files (run_053 through run_069)
CIFAR10_EXPERIMENTS = list(range(53, 70))  # 53 to 69 inclusive

def run_experiment(run_num):
    """Run a single experiment file."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    experiment_file = project_root / 'experiments' / f'run_grid_exp_{run_num:03d}.py'
    
    if not experiment_file.exists():
        print(f"ERROR: Experiment file not found: {experiment_file}")
        return False
    
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: run_grid_exp_{run_num:03d}.py".center(80))
    print("="*80)
    
    try:
        # Run the experiment
        result = subprocess.run(
            [sys.executable, str(experiment_file)],
            cwd=str(project_root),
            check=True,
            capture_output=False  # Show output in real-time
        )
        print(f"\n[SUCCESS] Experiment run_{run_num:03d} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] Experiment run_{run_num:03d} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Experiment run_{run_num:03d} interrupted by user")
        return False
    except Exception as e:
        print(f"\n[FAILED] Experiment run_{run_num:03d} failed with error: {e}")
        return False

def main():
    """Run all CIFAR-10 experiments sequentially."""
    print("="*80)
    print("CIFAR-10 EXPERIMENTS - SEQUENTIAL EXECUTION".center(80))
    print("="*80)
    print(f"Total experiments to run: {len(CIFAR10_EXPERIMENTS)}")
    print(f"Experiments: run_053 through run_069")
    print("="*80)
    
    results = []
    
    for i, run_num in enumerate(CIFAR10_EXPERIMENTS, 1):
        print(f"\n[{i}/{len(CIFAR10_EXPERIMENTS)}] Starting run_{run_num:03d}...")
        
        success = run_experiment(run_num)
        results.append((run_num, success))
        
        if not success:
            print(f"\n[WARNING] Experiment run_{run_num:03d} did not complete successfully")
            response = input("Continue with next experiment? (y/n): ")
            if response.lower() != 'y':
                print("\nStopping execution.")
                break
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY".center(80))
    print("="*80)
    
    successful = [r[0] for r in results if r[1]]
    failed = [r[0] for r in results if not r[1]]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n[SUCCESS] Successful experiments: {successful}")
    if failed:
        print(f"\n[FAILED] Failed experiments: {failed}")
    
    print("="*80)

if __name__ == "__main__":
    main()

