"""
Grid Experiment #012: LPL Baseline Run (3-Layer MLP, Extended Training - Long Regime)

Configuration:
- Dataset: Synthetic Shapes
- Training steps: 50,000 (extended from Run #11)
- Architecture: 3-layer MLP (128 → 64 → 32 units)
- Activation: tanh
- Learning rule: full LPL (Hebbian + Predictive + Stabilization enabled)
- Baseline: none (LPL only)
- Seed: 42 (matches Run #9, #10, and #11)

This run is identical to Run #11 except for training length (50000 vs 10000 steps).
"""

import torch
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lpl_core.hierarchical_lpl_3layer import HierarchicalLPL3Layer
from data.synthetic_shapes import SyntheticShapesDataset, create_temporal_pair_dataset


class LayerConfig:
    """Configuration object for LPL layer."""
    def __init__(self, lr_hebb=0.001, lr_pred=0.001, lr_stab=0.0005,
                 use_hebb=True, use_pred=True, use_stab=True):
        self.lr_hebb = lr_hebb
        self.lr_pred = lr_pred
        self.lr_stab = lr_stab
        self.use_hebb = use_hebb
        self.use_pred = use_pred
        self.use_stab = use_stab


def export_activations(model, dataset, num_samples=1000):
    """
    Export activations for a set of samples.
    For 3-layer model, exports final layer (layer 3) activations.
    
    Args:
        model: HierarchicalLPL3Layer model
        dataset: SyntheticShapesDataset (can be temporal pair or single image mode)
        num_samples: Number of samples to export
        
    Returns:
        Dictionary with 'activations' and 'labels'
    """
    activations = []
    labels = []
    
    for i in range(min(num_samples, len(dataset))):
        # Handle both temporal pair and single image modes
        if hasattr(dataset, 'return_temporal_pair') and dataset.return_temporal_pair:
            x_t, _, label = dataset[i]
            image = x_t
        else:
            image, label = dataset[i]
        
        # Flatten image to 1D tensor
        x = image.flatten()
        
        # Ensure input is float32 and in [0,1] range
        if x.dtype != torch.float32:
            x = x.float()
        x = torch.clamp(x, 0.0, 1.0)
        
        # Forward pass to get activations (returns layer 3 output)
        with torch.no_grad():
            y = model.forward(x)  # Returns layer 3 activations
        
        # Check for NaN in activations
        if torch.isnan(y).any():
            print(f"WARNING: NaN detected in activation at sample {i}")
            continue
        
        activations.append(y)
        labels.append(label)
    
    return {
        'activations': torch.stack(activations),
        'labels': torch.tensor(labels)
    }


def main():
    """
    Run grid experiment #012.
    """
    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Experiment configuration
    EXPERIMENT_CONFIG = {
        'dataset': 'synthetic_shapes',
        'steps': 50000,  # Changed to 50000 steps
        'architecture': 'mlp_3layer_128_64_32',
        'activation': 'tanh',
        'rule': 'full_lpl',
        'baseline': 'none',
        'seed': 42,
        'd_in': 32 * 32,      # 32x32 images flattened = 1024
        'd_hidden1': 128,     # First layer output dimension
        'd_hidden2': 64,      # Second layer output dimension
        'd_out': 32,          # Third layer output dimension (final representation)
        'lr_hebb': 0.001,
        'lr_pred': 0.001,
        'lr_stab': 0.0005,
    }
    
    print("="*70)
    print("GRID EXPERIMENT #012".center(70))
    print("="*70)
    print(f"Dataset: {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps: {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture: {EXPERIMENT_CONFIG['architecture']}")
    print(f"Activation: {EXPERIMENT_CONFIG['activation']}")
    print(f"Rule: {EXPERIMENT_CONFIG['rule']}")
    print(f"Baseline: {EXPERIMENT_CONFIG['baseline']}")
    print(f"Seed: {EXPERIMENT_CONFIG['seed']}")
    print("="*70)
    
    # Create output directory with experiment identifier
    output_base = Path('outputs/grid_experiments')
    output_dir = output_base / f"run_012_{EXPERIMENT_CONFIG['dataset']}_{EXPERIMENT_CONFIG['steps']}steps_{EXPERIMENT_CONFIG['architecture']}_{EXPERIMENT_CONFIG['activation']}_{EXPERIMENT_CONFIG['rule']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(EXPERIMENT_CONFIG, f, indent=2)
    print(f"\nMetadata saved to {metadata_file}")
    
    # Create layer configuration (full LPL: all rules enabled)
    # Same configuration for all three layers
    layer_cfg_layer1 = LayerConfig(
        lr_hebb=EXPERIMENT_CONFIG['lr_hebb'],
        lr_pred=EXPERIMENT_CONFIG['lr_pred'],
        lr_stab=EXPERIMENT_CONFIG['lr_stab'],
        use_hebb=True,   # Full LPL
        use_pred=True,   # Full LPL
        use_stab=True    # Full LPL
    )
    
    layer_cfg_layer2 = LayerConfig(
        lr_hebb=EXPERIMENT_CONFIG['lr_hebb'],
        lr_pred=EXPERIMENT_CONFIG['lr_pred'],
        lr_stab=EXPERIMENT_CONFIG['lr_stab'],
        use_hebb=True,   # Full LPL
        use_pred=True,   # Full LPL
        use_stab=True    # Full LPL
    )
    
    layer_cfg_layer3 = LayerConfig(
        lr_hebb=EXPERIMENT_CONFIG['lr_hebb'],
        lr_pred=EXPERIMENT_CONFIG['lr_pred'],
        lr_stab=EXPERIMENT_CONFIG['lr_stab'],
        use_hebb=True,   # Full LPL
        use_pred=True,   # Full LPL
        use_stab=True    # Full LPL
    )
    
    # Create model (3-layer MLP: 1024 -> 128 -> 64 -> 32)
    model = HierarchicalLPL3Layer(
        d_in=EXPERIMENT_CONFIG['d_in'],
        d_hidden1=EXPERIMENT_CONFIG['d_hidden1'],
        d_hidden2=EXPERIMENT_CONFIG['d_hidden2'],
        d_out=EXPERIMENT_CONFIG['d_out'],
        cfg_layer1=layer_cfg_layer1,
        cfg_layer2=layer_cfg_layer2,
        cfg_layer3=layer_cfg_layer3
    )
    
    # Create datasets
    # For activation export: single images (not temporal pairs)
    export_dataset = SyntheticShapesDataset(
        num_samples=1000,
        seed=42,
        return_temporal_pair=False
    )
    
    # For training: temporal pairs
    train_dataset = create_temporal_pair_dataset(
        num_samples=10000,  # Large enough to sample from during training
        seed=42
    )
    
    # Export activations before training
    print("\nExporting activations before training...")
    activations_before = export_activations(model, export_dataset, num_samples=1000)
    
    # Safety check: no NaN in activations
    assert not torch.isnan(activations_before['activations']).any(), \
        "ERROR: NaN detected in activations before training!"
    
    torch.save(activations_before, output_dir / 'activations_before.pt')
    print(f"Saved activations to {output_dir / 'activations_before.pt'}")
    
    # Initialize training logs
    training_logs = {
        'step': [],
        'weight_norm': [],
        'activation_norm': []
    }
    
    # Training loop
    print(f"\nTraining LPL for {EXPERIMENT_CONFIG['steps']} steps...")
    
    for step in range(1, EXPERIMENT_CONFIG['steps'] + 1):
        # Sample a temporal pair from the dataset
        idx = torch.randint(0, len(train_dataset), (1,)).item()
        x_t, x_t1, _ = train_dataset[idx]
        
        # Flatten images to 1D tensors
        x_t_flat = x_t.flatten()
        x_t1_flat = x_t1.flatten()
        
        # Ensure inputs are floats in [0,1] range
        if x_t_flat.dtype != torch.float32:
            x_t_flat = x_t_flat.float()
        if x_t1_flat.dtype != torch.float32:
            x_t1_flat = x_t1_flat.float()
        x_t_flat = torch.clamp(x_t_flat, 0.0, 1.0)
        x_t1_flat = torch.clamp(x_t1_flat, 0.0, 1.0)
        
        # Update model using local learning rules
        model.update(x_t_flat, x_t1_flat)
        
        # Log metrics every step
        # Use layer 3 (final layer) for consistency with other runs
        weight_norm = torch.norm(model.layer3.W).item()
        y_sample = model.forward(x_t_flat)  # Returns layer 3 output
        activation_norm = torch.norm(y_sample).item()
        
        training_logs['step'].append(step)
        training_logs['weight_norm'].append(weight_norm)
        training_logs['activation_norm'].append(activation_norm)
        
        # Safety check: no NaN in weights (check all three layers)
        if (torch.isnan(model.layer1.W).any() or 
            torch.isnan(model.layer2.W).any() or 
            torch.isnan(model.layer3.W).any()):
            print(f"ERROR: NaN detected in weights at step {step}!")
            assert False, f"NaN detected in weights at step {step}"
        
        # Safety check: no NaN in activations
        if torch.isnan(y_sample).any():
            print(f"ERROR: NaN detected in activations at step {step}!")
            assert False, f"NaN detected in activations at step {step}"
        
        # Print progress every 5000 steps (less frequent for longer run)
        if step % 5000 == 0:
            print(f"Step {step}/{EXPERIMENT_CONFIG['steps']} | "
                  f"||W||={weight_norm:.4f} | ||y||={activation_norm:.4f}")
    
    print("Training completed.")
    
    # Final safety check: no NaN in weights (all three layers)
    assert (not torch.isnan(model.layer1.W).any() and 
            not torch.isnan(model.layer2.W).any() and 
            not torch.isnan(model.layer3.W).any()), \
        "ERROR: Weights contain NaN values after training!"
    
    # Save training logs
    logs_file = output_dir / 'training_logs.json'
    with open(logs_file, 'w') as f:
        json.dump(training_logs, f, indent=2)
    print(f"\nSaved training logs to {logs_file}")
    
    # Export activations after training
    print("\nExporting activations after training...")
    activations_after = export_activations(model, export_dataset, num_samples=1000)
    
    # Safety check: no NaN in activations
    assert not torch.isnan(activations_after['activations']).any(), \
        "ERROR: NaN detected in activations after training!"
    
    torch.save(activations_after, output_dir / 'activations_after.pt')
    print(f"Saved activations to {output_dir / 'activations_after.pt'}")
    
    # Comprehensive safety verification
    print("\n" + "="*70)
    print("SAFETY & VERIFICATION".center(70))
    print("="*70)
    
    # Check for NaN in weights (all three layers)
    weights_layer1_has_nan = torch.isnan(model.layer1.W).any().item()
    weights_layer2_has_nan = torch.isnan(model.layer2.W).any().item()
    weights_layer3_has_nan = torch.isnan(model.layer3.W).any().item()
    weights_has_nan = weights_layer1_has_nan or weights_layer2_has_nan or weights_layer3_has_nan
    weights_check = 'PASS' if not weights_has_nan else 'FAIL'
    print(f"Weights NaN check (Layer1, Layer2, Layer3): {weights_check}")
    if weights_has_nan:
        print("  ⚠️  WARNING: NaN values detected in weights!")
    
    # Check for NaN in activations_before
    act_before_has_nan = torch.isnan(activations_before['activations']).any().item()
    act_before_check = 'PASS' if not act_before_has_nan else 'FAIL'
    print(f"Activations before NaN check: {act_before_check}")
    if act_before_has_nan:
        print("  ⚠️  WARNING: NaN values detected in activations_before!")
    
    # Check for NaN in activations_after
    act_after_has_nan = torch.isnan(activations_after['activations']).any().item()
    act_after_check = 'PASS' if not act_after_has_nan else 'FAIL'
    print(f"Activations after NaN check: {act_after_check}")
    if act_after_has_nan:
        print("  ⚠️  WARNING: NaN values detected in activations_after!")
    
    # Check activation std > 0.1 (non-collapsed)
    activations_final = activations_after['activations']
    activation_std = activations_final.std().item()
    activation_mean = activations_final.mean().item()
    activation_min = activations_final.min().item()
    activation_max = activations_final.max().item()
    non_collapsed = activation_std > 0.1
    collapse_check = 'PASS' if non_collapsed else 'FAIL'
    print(f"Activation std > 0.1 (non-collapsed): {collapse_check} (std={activation_std:.6f})")
    if not non_collapsed:
        print("  ⚠️  WARNING: Activation std <= 0.1 - representation may be collapsed!")
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS".center(70))
    print("="*70)
    print(f"Activation mean: {activation_mean:.6f}")
    print(f"Activation std:  {activation_std:.6f}")
    print(f"Activation min:  {activation_min:.6f}")
    print(f"Activation max:  {activation_max:.6f}")
    final_weight_norm = torch.norm(model.layer3.W).item()
    print(f"Weight norm (Layer3): {final_weight_norm:.6f}")
    print("="*70)
    
    # Final summary output
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY".center(70))
    print("="*70)
    print(f"Dataset:          {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps:            {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture:     {EXPERIMENT_CONFIG['architecture']}")
    print(f"Activation std:   {activation_std:.6f}")
    print(f"Weight norm:      {final_weight_norm:.6f}")
    all_checks_pass = not weights_has_nan and not act_before_has_nan and not act_after_has_nan and non_collapsed
    summary_check = 'PASS' if all_checks_pass else 'FAIL'
    print(f"NaN check:        {summary_check}")
    if not all_checks_pass:
        print("\n  ⚠️  WARNING: One or more safety checks failed!")
        print(f"     - Weights NaN: {weights_check}")
        print(f"     - Activations before NaN: {act_before_check}")
        print(f"     - Activations after NaN: {act_after_check}")
        print(f"     - Non-collapsed (std > 0.1): {collapse_check}")
    print("="*70)
    
    print(f"\nExperiment completed successfully!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()




