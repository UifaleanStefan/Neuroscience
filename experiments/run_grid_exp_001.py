"""
Grid Experiment #001: LPL Baseline Run

Configuration:
- Dataset: Synthetic Shapes
- Training steps: 1,000
- Architecture: 1-layer MLP (128 units)
- Activation: tanh
- Learning rule: full LPL (Hebbian + Predictive + Stabilization enabled)
- Baseline: none (LPL only)

This serves as the reference template for all remaining grid runs.
"""

import torch
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lpl_core.lpl_layer import LPLLayer
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
    
    Args:
        model: LPLLayer model
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
        
        # Forward pass to get activations
        with torch.no_grad():
            y = model.forward(x)
        
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
    Run grid experiment #001.
    """
    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Experiment configuration
    EXPERIMENT_CONFIG = {
        'dataset': 'synthetic_shapes',
        'steps': 1000,
        'architecture': 'mlp_1layer_128',
        'activation': 'tanh',
        'rule': 'full_lpl',
        'baseline': 'none',
        'd_in': 32 * 32,  # 32x32 images flattened
        'd_out': 128,      # 128 units
        'lr_hebb': 0.001,
        'lr_pred': 0.001,
        'lr_stab': 0.0005,
    }
    
    print("="*70)
    print("GRID EXPERIMENT #001".center(70))
    print("="*70)
    print(f"Dataset: {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps: {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture: {EXPERIMENT_CONFIG['architecture']}")
    print(f"Activation: {EXPERIMENT_CONFIG['activation']}")
    print(f"Rule: {EXPERIMENT_CONFIG['rule']}")
    print(f"Baseline: {EXPERIMENT_CONFIG['baseline']}")
    print("="*70)
    
    # Create output directory with experiment identifier
    output_base = Path('outputs/grid_experiments')
    output_dir = output_base / f"run_001_{EXPERIMENT_CONFIG['dataset']}_{EXPERIMENT_CONFIG['steps']}steps_{EXPERIMENT_CONFIG['architecture']}_{EXPERIMENT_CONFIG['activation']}_{EXPERIMENT_CONFIG['rule']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(EXPERIMENT_CONFIG, f, indent=2)
    print(f"\nMetadata saved to {metadata_file}")
    
    # Create layer configuration (full LPL: all rules enabled)
    layer_cfg = LayerConfig(
        lr_hebb=EXPERIMENT_CONFIG['lr_hebb'],
        lr_pred=EXPERIMENT_CONFIG['lr_pred'],
        lr_stab=EXPERIMENT_CONFIG['lr_stab'],
        use_hebb=True,   # Full LPL
        use_pred=True,   # Full LPL
        use_stab=True    # Full LPL
    )
    
    # Create model (1-layer MLP with 128 units)
    model = LPLLayer(
        d_in=EXPERIMENT_CONFIG['d_in'],
        d_out=EXPERIMENT_CONFIG['d_out'],
        cfg=layer_cfg
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
        weight_norm = torch.norm(model.W).item()
        y_sample = model.forward(x_t_flat)
        activation_norm = torch.norm(y_sample).item()
        
        training_logs['step'].append(step)
        training_logs['weight_norm'].append(weight_norm)
        training_logs['activation_norm'].append(activation_norm)
        
        # Safety check: no NaN in weights
        if torch.isnan(model.W).any():
            print(f"ERROR: NaN detected in weights at step {step}!")
            assert False, f"NaN detected in weights at step {step}"
        
        # Safety check: no NaN in activations
        if torch.isnan(y_sample).any():
            print(f"ERROR: NaN detected in activations at step {step}!")
            assert False, f"NaN detected in activations at step {step}"
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}/{EXPERIMENT_CONFIG['steps']} | "
                  f"||W||={weight_norm:.4f} | ||y||={activation_norm:.4f}")
    
    print("Training completed.")
    
    # Final safety check: no NaN in weights
    assert not torch.isnan(model.W).any(), \
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
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS".center(70))
    print("="*70)
    activations_final = activations_after['activations']
    print(f"Activation mean: {activations_final.mean().item():.6f}")
    print(f"Activation std:  {activations_final.std().item():.6f}")
    print(f"Activation min:  {activations_final.min().item():.6f}")
    print(f"Activation max:  {activations_final.max().item():.6f}")
    print(f"Weight norm:     {torch.norm(model.W).item():.6f}")
    print(f"No NaN:          {not torch.isnan(activations_final).any().item()}")
    print("="*70)
    
    print(f"\nExperiment completed successfully!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()




