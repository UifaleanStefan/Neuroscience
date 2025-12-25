"""
Grid Experiment #014: LPL Baseline Run (Conv-MLP Hybrid, 5K steps)

Configuration:
- Dataset: Synthetic Shapes
- Training steps: 5,000
- Architecture: Conv-MLP Hybrid (Conv1 → Linear LPL)
- Activation: tanh (on linear layer)
- Learning rule: Full LPL (on linear layer: Hebbian + Predictive + Stabilization enabled)
- Baseline: none (LPL only)
- Seed: 42

This run tests a Conv-MLP hybrid architecture with a conv layer for feature extraction
followed by a linear LPL layer trained with Full LPL rules, extended to 5,000 steps.
"""

import torch
import json
import sys
import gc
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lpl_core.conv_mlp_hybrid import ConvMLPHybrid
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


def export_activations(model, dataset, num_samples=1000, device='cpu'):
    """
    Export activations for a set of samples.
    Exports final layer (linear LPL layer) activations.
    
    Args:
        model: ConvMLPHybrid model
        dataset: SyntheticShapesDataset (can be temporal pair or single image mode)
        num_samples: Number of samples to export
        device: Device to run on ('cpu' or 'cuda')
        
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
        
        # Image should be 2D (H, W) from synthetic shapes
        # Ensure input is float32 and in [0,1] range
        if image.dtype != torch.float32:
            image = image.float()
        image = torch.clamp(image, 0.0, 1.0)
        
        # Move to device
        image = image.to(device)
        
        # Forward pass to get activations (returns linear layer output)
        with torch.no_grad():
            y = model.forward(image)
        
        # Move back to CPU for storage
        y = y.cpu()
        
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


def move_model_to_device(model, device):
    """Move ConvMLPHybrid model to specified device."""
    # Move conv layer (nn.Module, use .to())
    model.conv = model.conv.to(device)
    if model.pool is not None:
        model.pool = model.pool.to(device)
    # Move linear LPL layer weights (tensors, use .to() and reassign)
    model.linear.W = model.linear.W.to(device)
    model.linear.predictor.P = model.linear.predictor.P.to(device)
    # Store device for future operations
    model.device = device
    return model


def main():
    """
    Run grid experiment #014.
    """
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    
    # Experiment configuration
    EXPERIMENT_CONFIG = {
        'dataset': 'synthetic_shapes',
        'steps': 5000,
        'architecture': 'conv_mlp_hybrid',
        'activation': 'tanh',
        'rule': 'full_lpl',
        'baseline': 'none',
        'seed': 42,
        'input_channels': 1,        # Grayscale
        'input_size': 32,           # 32x32 images
        'conv_out_channels': 8,     # Number of conv output channels (reduced for GPU memory)
        'conv_kernel_size': 3,      # 3x3 conv kernel
        'd_out': 32,                # Linear LPL layer output dimension (reduced for GPU memory)
        'use_pooling': True,        # Use max pooling to reduce spatial dimensions
        'lr_hebb': 0.001,
        'lr_pred': 0.001,
        'lr_stab': 0.0005,
        'device': str(device),
    }
    
    print("="*70)
    print("GRID EXPERIMENT #014".center(70))
    print("="*70)
    print(f"Dataset: {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps: {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture: {EXPERIMENT_CONFIG['architecture']}")
    print(f"  Conv: {EXPERIMENT_CONFIG['conv_out_channels']} channels, "
          f"kernel={EXPERIMENT_CONFIG['conv_kernel_size']}")
    print(f"  Linear: {EXPERIMENT_CONFIG['d_out']} output dims")
    print(f"Activation: {EXPERIMENT_CONFIG['activation']}")
    print(f"Rule: {EXPERIMENT_CONFIG['rule']}")
    print(f"Baseline: {EXPERIMENT_CONFIG['baseline']}")
    print(f"Seed: {EXPERIMENT_CONFIG['seed']}")
    print(f"Device: {EXPERIMENT_CONFIG['device']}")
    print("="*70)
    
    # Create output directory with experiment identifier
    output_base = Path('outputs/grid_experiments')
    output_dir = output_base / f"run_014_{EXPERIMENT_CONFIG['dataset']}_{EXPERIMENT_CONFIG['steps']}steps_{EXPERIMENT_CONFIG['architecture']}_{EXPERIMENT_CONFIG['activation']}_{EXPERIMENT_CONFIG['rule']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(EXPERIMENT_CONFIG, f, indent=2)
    print(f"\nMetadata saved to {metadata_file}")
    
    # Create layer configuration (full LPL: all rules enabled)
    linear_cfg = LayerConfig(
        lr_hebb=EXPERIMENT_CONFIG['lr_hebb'],
        lr_pred=EXPERIMENT_CONFIG['lr_pred'],
        lr_stab=EXPERIMENT_CONFIG['lr_stab'],
        use_hebb=True,   # Full LPL
        use_pred=True,   # Full LPL
        use_stab=True    # Full LPL
    )
    
    # Create model (Conv-MLP Hybrid)
    model = ConvMLPHybrid(
        input_channels=EXPERIMENT_CONFIG['input_channels'],
        input_size=EXPERIMENT_CONFIG['input_size'],
        conv_out_channels=EXPERIMENT_CONFIG['conv_out_channels'],
        conv_kernel_size=EXPERIMENT_CONFIG['conv_kernel_size'],
        d_out=EXPERIMENT_CONFIG['d_out'],
        cfg_linear=linear_cfg,
        use_pooling=EXPERIMENT_CONFIG.get('use_pooling', True)
    )
    
    # Move model to device
    model = move_model_to_device(model, device)
    
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
    activations_before = export_activations(model, export_dataset, num_samples=1000, device=device)
    
    # Safety check: no NaN in activations
    assert not torch.isnan(activations_before['activations']).any(), \
        "ERROR: NaN detected in activations before training!"
    
    torch.save(activations_before, output_dir / 'activations_before.pt')
    print(f"Saved activations to {output_dir / 'activations_before.pt'}")
    
    # Initialize training logs
    # Log every N steps to reduce memory usage
    log_interval = 10  # Log every 10 steps instead of every step
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
        
        # Ensure inputs are floats in [0,1] range (images are already 2D)
        if x_t.dtype != torch.float32:
            x_t = x_t.float()
        if x_t1.dtype != torch.float32:
            x_t1 = x_t1.float()
        x_t = torch.clamp(x_t, 0.0, 1.0)
        x_t1 = torch.clamp(x_t1, 0.0, 1.0)
        
        # Move to device
        x_t = x_t.to(device)
        x_t1 = x_t1.to(device)
        
        # Update model using local learning rules (only linear layer updates)
        model.update(x_t, x_t1)
        
        # Safety check: no NaN in weights (check linear layer) - always check
        if torch.isnan(model.linear.W).any():
            print(f"ERROR: NaN detected in weights at step {step}!")
            assert False, f"NaN detected in weights at step {step}"
        
        # Log metrics periodically to reduce memory usage
        if step % log_interval == 0 or step == 1:
            # Use linear layer weight norm for consistency
            weight_norm = torch.norm(model.linear.W).item()
            y_sample = model.forward(x_t)  # Returns linear layer output
            activation_norm = torch.norm(y_sample).item()
            
            # Safety check: no NaN in activations (only when logging to save memory)
            if torch.isnan(y_sample).any():
                print(f"ERROR: NaN detected in activations at step {step}!")
                assert False, f"NaN detected in activations at step {step}"
            
            training_logs['step'].append(step)
            training_logs['weight_norm'].append(weight_norm)
            training_logs['activation_norm'].append(activation_norm)
            
            # Clear y_sample to save memory
            del y_sample
        
        # Clear intermediate tensors to save memory
        del x_t, x_t1
        
        # Print progress every 500 steps
        if step % 500 == 0:
            # Compute metrics for display
            weight_norm = torch.norm(model.linear.W).item()
            # Sample a random image for activation norm display
            sample_idx = torch.randint(0, len(train_dataset), (1,)).item()
            sample_img, _, _ = train_dataset[sample_idx]
            sample_img = sample_img.to(device).float().clamp(0.0, 1.0)
            y_temp = model.forward(sample_img)
            activation_norm = torch.norm(y_temp).item()
            del y_temp, sample_img
            print(f"Step {step}/{EXPERIMENT_CONFIG['steps']} | "
                  f"||W||={weight_norm:.4f} | ||y||={activation_norm:.4f}")
            # Periodic garbage collection to prevent memory buildup
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    print("Training completed.")
    
    # Final safety check: no NaN in weights
    assert not torch.isnan(model.linear.W).any(), \
        "ERROR: Weights contain NaN values after training!"
    
    # Save training logs
    logs_file = output_dir / 'training_logs.json'
    with open(logs_file, 'w') as f:
        json.dump(training_logs, f, indent=2)
    print(f"\nSaved training logs to {logs_file}")
    
    # Export activations after training
    print("\nExporting activations after training...")
    activations_after = export_activations(model, export_dataset, num_samples=1000, device=device)
    
    # Safety check: no NaN in activations
    assert not torch.isnan(activations_after['activations']).any(), \
        "ERROR: NaN detected in activations after training!"
    
    torch.save(activations_after, output_dir / 'activations_after.pt')
    print(f"Saved activations to {output_dir / 'activations_after.pt'}")
    
    # Comprehensive safety verification
    print("\n" + "="*70)
    print("SAFETY & VERIFICATION".center(70))
    print("="*70)
    
    # Check for NaN in weights
    weights_has_nan = torch.isnan(model.linear.W).any().item()
    weights_check = 'PASS' if not weights_has_nan else 'FAIL'
    print(f"Weights NaN check (Linear layer): {weights_check}")
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
    final_weight_norm = torch.norm(model.linear.W).item()
    print(f"Weight norm (Linear): {final_weight_norm:.6f}")
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
