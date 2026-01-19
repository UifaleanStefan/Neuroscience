"""
Create a bar chart comparing LPL vs Backpropagation baselines across all datasets.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

# Backprop experiment directories
backprop_experiments = {
    'Synthetic Shapes': Path('outputs/grid_experiments/run_017_synthetic_shapes_10000steps_mlp_1layer_128_tanh_backprop'),
    'MNIST': Path('outputs/grid_experiments/run_034_mnist_10000steps_mlp_1layer_128_tanh_backprop'),
    'Fashion-MNIST': Path('outputs/grid_experiments/run_052_fashion_mnist_10000steps_mlp_1layer_128_tanh_backprop'),
    'CIFAR-10': Path('outputs/grid_experiments/run_069_cifar10_10000steps_mlp_1layer_128_tanh_backprop'),
    'STL-10': Path('outputs/grid_experiments/run_086_stl10_10000steps_mlp_1layer_128_tanh_backprop')
}

def compute_linear_probe(embeddings, labels, n_splits=3, random_seed=42):
    """Compute linear probe accuracy with multiple splits."""
    X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    accuracies = []
    for split in range(n_splits):
        split_seed = random_seed + split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=split_seed, stratify=y
            )
            
            clf = LogisticRegression(
                max_iter=1000, 
                random_state=split_seed, 
                C=1.0, 
                solver='lbfgs', 
                n_jobs=1
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        except Exception as e:
            print(f"  Warning: Linear probe failed for split {split}: {e}")
            continue
    
    if len(accuracies) == 0:
        return None
    
    return np.mean(accuracies) * 100  # Convert to percentage

def load_backprop_results(exp_dir):
    """Load embeddings and compute linear probe accuracy."""
    embeddings_file = exp_dir / 'embeddings_after.pt'
    if not embeddings_file.exists():
        print(f"  Warning: {embeddings_file} not found")
        return None
    
    data = torch.load(embeddings_file, map_location='cpu')
    
    # Handle different formats
    if isinstance(data, dict):
        if 'embeddings' in data:
            embeddings = data['embeddings']
            labels = data['labels']
        else:
            print(f"  Warning: Unknown format. Keys: {list(data.keys())}")
            return None
    else:
        print(f"  Warning: Unexpected format: {type(data)}")
        return None
    
    # Compute linear probe
    accuracy = compute_linear_probe(embeddings, labels)
    return accuracy

# Best LPL accuracy per dataset (from previous analysis)
lpl_best_accuracy = {
    'Synthetic Shapes': 100.0,   # MLP 1-layer, 1k steps
    'MNIST': 69.0,                # MLP 1-layer, 5k steps
    'Fashion-MNIST': 67.0,        # MLP 3-layer, 50k steps (collapsed, but best)
    'CIFAR-10': 25.3,             # MLP 1-layer, 1k steps
    'STL-10': 25.2                # MLP 1-layer, 1k steps
}

# Load backprop results
print("Loading backpropagation baseline results...")
backprop_accuracy = {}
for dataset, exp_dir in backprop_experiments.items():
    print(f"\nProcessing {dataset}...")
    if not exp_dir.exists():
        print(f"  Warning: Experiment directory not found: {exp_dir}")
        backprop_accuracy[dataset] = None
        continue
    
    accuracy = load_backprop_results(exp_dir)
    backprop_accuracy[dataset] = accuracy
    if accuracy is not None:
        print(f"  Backprop accuracy: {accuracy:.2f}%")
    else:
        print(f"  Failed to compute accuracy")

# Prepare data for plotting
datasets_ordered = ['Synthetic\nShapes', 'MNIST', 'Fashion-\nMNIST', 'CIFAR-10', 'STL-10']
dataset_keys = ['Synthetic Shapes', 'MNIST', 'Fashion-MNIST', 'CIFAR-10', 'STL-10']
lpl_values = [lpl_best_accuracy[key] for key in dataset_keys]
backprop_values = [backprop_accuracy.get(key, None) for key in dataset_keys]

# Check if we have all values
missing = [d for d, v in zip(datasets_ordered, backprop_values) if v is None]
if missing:
    print(f"\nWarning: Missing backprop results for: {missing}")

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Set positions for bars
x = np.arange(len(datasets_ordered))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, lpl_values, width, label='LPL (Best)', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, backprop_values, width, label='Backpropagation (10k)', color='#A23B72', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Customize plot
ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
ax.set_ylabel('Linear Probe Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('LPL vs Backpropagation: Best Performance Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets_ordered, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)

# Add note
note_text = 'Note: Backprop uses self-supervised temporal consistency loss (same objective as LPL)'
ax.text(0.5, 0.02, note_text,
        transform=ax.transAxes, fontsize=9, style='italic',
        horizontalalignment='center', verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_dir = Path('analysis/representation_invariance')
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / 'lpl_vs_backprop_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved figure to {output_path}")

# Also save as PDF
pdf_path = output_dir / 'lpl_vs_backprop_comparison.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
print(f"Saved PDF to {pdf_path}")

plt.close()

# Print summary
print("\n" + "="*60)
print("LPL vs Backpropagation Comparison Summary")
print("="*60)
print(f"{'Dataset':<20} {'LPL Best':<12} {'Backprop (10k)':<15} {'Difference':<12}")
print("="*60)
for i, dataset in enumerate(dataset_keys):
    lpl_val = lpl_values[i]
    backprop_val = backprop_values[i] if backprop_values[i] is not None else 0
    diff = lpl_val - backprop_val if backprop_values[i] is not None else None
    diff_str = f"{diff:+.1f}%" if diff is not None else "N/A"
    backprop_str = f"{backprop_val:.1f}%" if backprop_values[i] is not None else "N/A"
    print(f"{dataset:<20} {lpl_val:>6.1f}%      {backprop_str:<15} {diff_str:<12}")
