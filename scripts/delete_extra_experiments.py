"""Delete extra/duplicate MNIST experiment results."""

from pathlib import Path
import json
import shutil

outputs_dir = Path('outputs/grid_experiments')

# Find all MNIST experiments
mnist_experiments = []
for d in outputs_dir.iterdir():
    if d.is_dir() and d.name.startswith('run_'):
        meta = d / 'metadata.json'
        if meta.exists():
            try:
                data = json.load(open(meta))
                if data.get('dataset') == 'mnist':
                    run_id = int(d.name.split('_')[1])
                    mnist_experiments.append((run_id, d.name, d))
            except Exception as e:
                print(f"Error processing {d.name}: {e}")

mnist_experiments.sort()

print("="*70)
print("IDENTIFYING EXTRA MNIST EXPERIMENTS".center(70))
print("="*70)
print()

# Standard pattern: 17 experiments (run_018 to run_034)
# Expected: 4 architectures × 4 step counts + 1 backprop = 17
# But we have 18 experiments (run_018 to run_035)

# Find duplicates for run_026-029 (3-layer MLP)
duplicates_to_delete = []
for run_id, name, path in mnist_experiments:
    # Delete the mlp_3layer_128_64_32 versions (keep mlp_3layer_256_128_64)
    if 'mlp_3layer_128_64_32' in name:
        duplicates_to_delete.append((run_id, name, path))
    # Delete run_035 (extra 20000-step conv_mlp)
    elif run_id == 35:
        duplicates_to_delete.append((run_id, name, path))

print(f"Found {len(duplicates_to_delete)} extra experiments to delete:")
print()
for run_id, name, path in duplicates_to_delete:
    print(f"  run_{run_id:03d}: {name}")

print()
print("="*70)
print("Deleting extra experiments...")
print()

for run_id, name, path in duplicates_to_delete:
    print(f"  Deleting {name}...")
    shutil.rmtree(path)

print()
print(f"Successfully deleted {len(duplicates_to_delete)} extra experiments.")
print("="*70)

