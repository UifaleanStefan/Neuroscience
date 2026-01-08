"""Check which experiments have been completed."""

from pathlib import Path
import json
from collections import defaultdict

outputs_dir = Path('outputs/grid_experiments')
datasets = defaultdict(list)

for d in outputs_dir.iterdir():
    if d.is_dir() and d.name.startswith('run_'):
        meta = d / 'metadata.json'
        if meta.exists():
            try:
                data = json.load(open(meta))
                run_id = int(d.name.split('_')[1])
                dataset = data.get('dataset', 'unknown')
                datasets[dataset].append(run_id)
            except Exception as e:
                print(f"Error processing {d.name}: {e}")

print("="*70)
print("EXPERIMENT STATUS SUMMARY".center(70))
print("="*70)
print()

total = 0
for ds, runs in sorted(datasets.items()):
    runs_sorted = sorted(runs)
    print(f"{ds}: {len(runs)} experiments")
    print(f"  Range: run_{min(runs_sorted):03d} to run_{max(runs_sorted):03d}")
    total += len(runs)

print()
print(f"Total experiments completed: {total}")
print("="*70)

