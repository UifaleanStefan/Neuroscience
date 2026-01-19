"""
Fix participation ratio calculation and swap selectivity normalization issues.
"""

import torch
import numpy as np
from pathlib import Path
import json

# Load swap experiment data
swap_file = Path("outputs/activations/swap_experiment.pt")
if swap_file.exists():
    data = torch.load(swap_file, map_location='cpu')
    activations_before = data['activations_before'].numpy()
    activations_after = data['activations_after'].numpy()
    labels_before = data['labels_before'].numpy()
    labels_after = data['labels_after'].numpy()
    
    print("="*70)
    print("Swap Selectivity Scale Check")
    print("="*70)
    print(f"Before activations shape: {activations_before.shape}")
    print(f"Before activations mean: {activations_before.mean():.6f}")
    print(f"Before activations std: {activations_before.std():.6f}")
    print(f"Before activations min: {activations_before.min():.6f}")
    print(f"Before activations max: {activations_before.max():.6f}")
    print()
    print(f"After activations shape: {activations_after.shape}")
    print(f"After activations mean: {activations_after.mean():.6f}")
    print(f"After activations std: {activations_after.std():.6f}")
    print(f"After activations min: {activations_after.min():.6f}")
    print(f"After activations max: {activations_after.max():.6f}")
    print()
    
    # Check selectivity computation
    unique_labels = np.unique(labels_before)
    print(f"Classes: {unique_labels}")
    print()
    
    for label in sorted(unique_labels[:2]):  # First 2 classes
        mask_before = labels_before == label
        mask_after = labels_after == label
        
        if mask_before.sum() > 0:
            mean_activation_before = activations_before[mask_before].mean(axis=0)
            print(f"Class {label} before: mean={mean_activation_before.mean():.6f}, "
                  f"std={mean_activation_before.std():.6f}, "
                  f"max={mean_activation_before.max():.6f}")
        if mask_after.sum() > 0:
            mean_activation_after = activations_after[mask_after].mean(axis=0)
            print(f"Class {label} after: mean={mean_activation_after.mean():.6f}, "
                  f"std={mean_activation_after.std():.6f}, "
                  f"max={mean_activation_after.max():.6f}")
    print()

# Test participation ratio on sample data
print("="*70)
print("Participation Ratio Test")
print("="*70)

# Create synthetic test data
np.random.seed(42)
test_data_1 = np.random.randn(100, 128)  # High dimensional
test_data_2 = np.random.randn(100, 128) * 0.01 + np.random.randn(100, 1)  # Low dimensional (one dominant dimension)

def compute_participation_ratio_old(activations):
    """Old version - check for bugs."""
    activations_centered = activations - activations.mean(axis=0, keepdims=True)
    C = np.cov(activations_centered.T)
    trace_C = np.trace(C)
    trace_C2 = np.trace(C @ C)
    if trace_C2 > 0:
        return (trace_C ** 2) / trace_C2
    return 0.0

def compute_participation_ratio_new(activations):
    """New version - using eigenvalues directly."""
    activations_centered = activations - activations.mean(axis=0, keepdims=True)
    C = np.cov(activations_centered.T)
    
    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(C)
    eigenvals = np.real(eigenvals)  # Take real part
    eigenvals = eigenvals[eigenvals > 1e-10]  # Filter near-zero eigenvalues
    
    # Participation ratio from eigenvalues
    if len(eigenvals) > 0:
        trace_C = eigenvals.sum()
        trace_C2 = (eigenvals ** 2).sum()
        if trace_C2 > 0:
            return (trace_C ** 2) / trace_C2
    return 0.0

print("Test data 1 (high dimensional):")
pr_old_1 = compute_participation_ratio_old(test_data_1)
pr_new_1 = compute_participation_ratio_new(test_data_1)
print(f"  Old method: {pr_old_1:.6f}")
print(f"  New method: {pr_new_1:.6f}")
print(f"  Expected: ~{min(100, 128)} (should be close to min(n_samples, n_features))")
print()

print("Test data 2 (low dimensional - one dominant):")
pr_old_2 = compute_participation_ratio_old(test_data_2)
pr_new_2 = compute_participation_ratio_new(test_data_2)
print(f"  Old method: {pr_old_2:.6f}")
print(f"  New method: {pr_new_2:.6f}")
print(f"  Expected: ~1.0 (one dominant dimension)")
print()

# Check actual activation data
print("Checking actual activation files...")
activation_files = list(Path("outputs/grid_experiments").glob("**/activations_after.pt"))
if activation_files:
    test_file = activation_files[0]
    print(f"Testing file: {test_file}")
    data = torch.load(test_file, map_location='cpu')
    
    if 'activations' in data:
        acts = data['activations'].numpy()
    elif 'layer1_activations' in data:
        acts = data['layer1_activations'].numpy()
    else:
        acts = None
    
    if acts is not None:
        print(f"  Shape: {acts.shape}")
        print(f"  Mean: {acts.mean():.6f}, Std: {acts.std():.6f}")
        
        # Sample subset for testing
        if len(acts) > 1000:
            indices = np.random.choice(len(acts), 1000, replace=False)
            acts_sub = acts[indices]
        else:
            acts_sub = acts
        
        pr_old = compute_participation_ratio_old(acts_sub)
        pr_new = compute_participation_ratio_new(acts_sub)
        print(f"  Participation ratio (old): {pr_old:.6f}")
        print(f"  Participation ratio (new): {pr_new:.6f}")
        print(f"  Note: Should be between 1 and {min(len(acts_sub), acts_sub.shape[1])}")
