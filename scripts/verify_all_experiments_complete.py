"""Verify that all experiment files have corresponding outputs."""

from pathlib import Path
import json

experiments_dir = Path('experiments')
outputs_dir = Path('outputs/grid_experiments')

# Get all experiment files
exp_files = sorted([int(f.stem.split('_')[-1]) for f in experiments_dir.glob('run_grid_exp_*.py')])

# Get all completed outputs
outputs = []
for d in outputs_dir.iterdir():
    if d.is_dir() and d.name.startswith('run_'):
        meta = d / 'metadata.json'
        if meta.exists():
            try:
                run_id = int(d.name.split('_')[1])
                outputs.append(run_id)
            except:
                pass

outputs = sorted(outputs)

# Find missing
missing = [i for i in exp_files if i not in outputs]

print("="*70)
print("EXPERIMENT COMPLETENESS CHECK".center(70))
print("="*70)
print()
print(f"Experiment files found: {len(exp_files)}")
print(f"Completed outputs found: {len(outputs)}")
print()

if missing:
    print(f"WARNING: MISSING OUTPUTS: {len(missing)} experiments")
    print(f"   Run IDs: {missing}")
else:
    print("SUCCESS: ALL EXPERIMENTS COMPLETE!")
    print()

# Check dataset coverage
datasets = {}
for d in outputs_dir.iterdir():
    if d.is_dir() and d.name.startswith('run_'):
        meta = d / 'metadata.json'
        if meta.exists():
            try:
                data = json.load(open(meta))
                dataset = data.get('dataset', 'unknown')
                run_id = int(d.name.split('_')[1])
                if dataset not in datasets:
                    datasets[dataset] = []
                datasets[dataset].append(run_id)
            except:
                pass

print("Dataset coverage:")
for ds, runs in sorted(datasets.items()):
    runs_sorted = sorted(runs)
    print(f"  {ds}: {len(runs_sorted)} experiments (run_{min(runs_sorted):03d} to run_{max(runs_sorted):03d})")

print()
print("="*70)

