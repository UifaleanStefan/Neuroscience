"""
Grid Experiment #029: MNIST with 3-layer MLP (50k steps, long training)

Configuration:
- Dataset: MNIST (grayscale, 28x28 images, flattened to 784)
- Training steps: 50,000
- Architecture: 3-layer MLP (784 → 256 → 128 → 64 units)
- Activation: tanh (scaled, same as previous runs)
- Learning rule: full LPL (Hebbian + Predictive + Stabilization enabled) on all layers
- Temporal pairs: Translation + noise transformations
- Seed: 42

This experiment includes:
- Close monitoring of stability during long training
- Per-layer activation variance tracking
- Weight norm tracking
- Update magnitude tracking
- Early abort if all layers collapse to near-zero variance
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
from data.mnist import MNISTTemporalPairDataset, create_mnist_temporal_pair_dataset


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
    Export activations from all three layers.
    
    Args:
        model: HierarchicalLPL3Layer model
        dataset: MNISTTemporalPairDataset (can be temporal pair or single image mode)
        num_samples: Number of samples to export
        
    Returns:
        Dictionary with 'layer1_activations', 'layer2_activations', 'layer3_activations', and 'labels'
    """
    layer1_activations = []
    layer2_activations = []
    layer3_activations = []
    labels = []
    
    for i in range(min(num_samples, len(dataset))):
        # Handle both temporal pair and single image modes
        if hasattr(dataset, 'return_temporal_pair') and dataset.return_temporal_pair:
            x_t, _, label = dataset[i]
            image = x_t
        else:
            image, label = dataset[i]
        
        # Flatten image to 1D tensor (28x28 = 784)
        x = image.flatten()
        
        # Ensure input is float32 and in [0,1] range
        if x.dtype != torch.float32:
            x = x.float()
        x = torch.clamp(x, 0.0, 1.0)
        
        # Forward pass to get activations from all three layers
        with torch.no_grad():
            y1 = model.layer1.forward(x)
            y2 = model.layer2.forward(y1)
            y3 = model.layer3.forward(y2)
        
        # Check for NaN in activations (all layers)
        if torch.isnan(y1).any():
            print(f"WARNING: NaN detected in layer1 activation at sample {i}")
            continue
        if torch.isnan(y2).any():
            print(f"WARNING: NaN detected in layer2 activation at sample {i}")
            continue
        if torch.isnan(y3).any():
            print(f"WARNING: NaN detected in layer3 activation at sample {i}")
            continue
        
        layer1_activations.append(y1)
        layer2_activations.append(y2)
        layer3_activations.append(y3)
        labels.append(label)
    
    return {
        'layer1_activations': torch.stack(layer1_activations),
        'layer2_activations': torch.stack(layer2_activations),
        'layer3_activations': torch.stack(layer3_activations),
        'labels': torch.tensor(labels)
    }


def main():
    """
    Run grid experiment #029.
    """
    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Experiment configuration
    EXPERIMENT_CONFIG = {
        'dataset': 'mnist',
        'steps': 50000,
        'architecture': 'mlp_3layer_256_128_64',
        'activation': 'tanh',
        'rule': 'full_lpl',
        'baseline': 'none',
        'd_in': 28 * 28,  # 28x28 images flattened to 784
        'd_hidden1': 256,  # First layer output dimension
        'd_hidden2': 128,  # Second layer output dimension
        'd_out': 64,       # Third layer output dimension (final representation)
        'lr_hebb': 0.001,
        'lr_pred': 0.001,
        'lr_stab': 0.0005,
        'seed': 42,
        'translate_range': 2,
        'noise_std': 0.05
    }
    
    print("="*70)
    print("GRID EXPERIMENT #029".center(70))
    print("="*70)
    print(f"Dataset: {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps: {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture: {EXPERIMENT_CONFIG['architecture']}")
    print(f"Activation: {EXPERIMENT_CONFIG['activation']}")
    print(f"Rule: {EXPERIMENT_CONFIG['rule']}")
    print(f"Baseline: {EXPERIMENT_CONFIG['baseline']}")
    print(f"Input dimension: {EXPERIMENT_CONFIG['d_in']} (28x28 flattened)")
    print(f"Layer 1 output: {EXPERIMENT_CONFIG['d_hidden1']}")
    print(f"Layer 2 output: {EXPERIMENT_CONFIG['d_hidden2']}")
    print(f"Layer 3 output: {EXPERIMENT_CONFIG['d_out']}")
    print("="*70)
    
    # Create output directory with experiment identifier
    output_base = Path('outputs/grid_experiments')
    output_dir = output_base / f"run_029_{EXPERIMENT_CONFIG['dataset']}_{EXPERIMENT_CONFIG['steps']}steps_{EXPERIMENT_CONFIG['architecture']}_{EXPERIMENT_CONFIG['activation']}_{EXPERIMENT_CONFIG['rule']}"
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
    
    # Create model (3-layer MLP: 784 → 256 → 128 → 64)
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
    export_dataset = MNISTTemporalPairDataset(
        train=True,
        return_temporal_pair=False,
        translate_range=EXPERIMENT_CONFIG['translate_range'],
        noise_std=EXPERIMENT_CONFIG['noise_std'],
        seed=EXPERIMENT_CONFIG['seed']
    )
    
    # For training: temporal pairs
    train_dataset = create_mnist_temporal_pair_dataset(
        train=True,
        translate_range=EXPERIMENT_CONFIG['translate_range'],
        noise_std=EXPERIMENT_CONFIG['noise_std'],
        seed=EXPERIMENT_CONFIG['seed']
    )
    
    print(f"\nDataset sizes:")
    print(f"  Export dataset: {len(export_dataset)} samples")
    print(f"  Training dataset: {len(train_dataset)} samples")
    
    # Export activations before training (initialization)
    print("\nExporting activations at initialization (before training)...")
    activations_before = export_activations(model, export_dataset, num_samples=1000)
    
    # Safety check: no NaN in activations (all three layers)
    assert not torch.isnan(activations_before['layer1_activations']).any(), \
        "ERROR: NaN detected in layer1 activations before training!"
    assert not torch.isnan(activations_before['layer2_activations']).any(), \
        "ERROR: NaN detected in layer2 activations before training!"
    assert not torch.isnan(activations_before['layer3_activations']).any(), \
        "ERROR: NaN detected in layer3 activations before training!"
    
    # Safety check: activation std > 0.1 (all three layers)
    layer1_std_before = activations_before['layer1_activations'].std().item()
    layer2_std_before = activations_before['layer2_activations'].std().item()
    layer3_std_before = activations_before['layer3_activations'].std().item()
    print(f"Layer1 activation std before training: {layer1_std_before:.6f}")
    print(f"Layer2 activation std before training: {layer2_std_before:.6f}")
    print(f"Layer3 activation std before training: {layer3_std_before:.6f}")
    if layer1_std_before < 0.1:
        print(f"WARNING: Layer1 activation std ({layer1_std_before:.6f}) is below 0.1 threshold!")
    if layer2_std_before < 0.1:
        print(f"WARNING: Layer2 activation std ({layer2_std_before:.6f}) is below 0.1 threshold!")
    if layer3_std_before < 0.1:
        print(f"WARNING: Layer3 activation std ({layer3_std_before:.6f}) is below 0.1 threshold!")
    
    torch.save(activations_before, output_dir / 'activations_before.pt')
    print(f"Saved activations to {output_dir / 'activations_before.pt'}")
    
    # Initialize training logs with per-layer statistics and update magnitudes
    training_logs = {
        'step': [],
        'weight_norm_layer1': [],
        'weight_norm_layer2': [],
        'weight_norm_layer3': [],
        'activation_norm_layer1': [],
        'activation_norm_layer2': [],
        'activation_norm_layer3': [],
        'activation_mean_layer1': [],
        'activation_mean_layer2': [],
        'activation_mean_layer3': [],
        'activation_std_layer1': [],  # Activation variance (std)
        'activation_std_layer2': [],
        'activation_std_layer3': [],
        'update_magnitude_layer1': [],  # Weight update magnitude
        'update_magnitude_layer2': [],
        'update_magnitude_layer3': []
    }
    
    # Collapse tracking
    collapse_info = {
        'layer1_collapsed': False,
        'layer2_collapsed': False,
        'layer3_collapsed': False,
        'all_layers_collapsed': False,
        'collapse_step_layer1': None,
        'collapse_step_layer2': None,
        'collapse_step_layer3': None,
        'collapse_step_all': None
    }
    
    # Training loop
    print(f"\nTraining LPL for {EXPERIMENT_CONFIG['steps']} steps...")
    print("Note: This is a long training run. Progress will be reported every 500 steps.")
    print("Training will abort if all layers collapse to near-zero variance (std < 0.1).")
    
    for step in range(1, EXPERIMENT_CONFIG['steps'] + 1):
        # Sample a temporal pair from the dataset
        idx = torch.randint(0, len(train_dataset), (1,)).item()
        x_t, x_t1, _ = train_dataset[idx]
        
        # Flatten images to 1D tensors (28x28 = 784)
        x_t_flat = x_t.flatten()
        x_t1_flat = x_t1.flatten()
        
        # Ensure inputs are floats in [0,1] range
        if x_t_flat.dtype != torch.float32:
            x_t_flat = x_t_flat.float()
        if x_t1_flat.dtype != torch.float32:
            x_t1_flat = x_t1_flat.float()
        x_t_flat = torch.clamp(x_t_flat, 0.0, 1.0)
        x_t1_flat = torch.clamp(x_t1_flat, 0.0, 1.0)
        
        # Store weights before update to compute update magnitude
        W1_before = model.layer1.W.clone()
        W2_before = model.layer2.W.clone()
        W3_before = model.layer3.W.clone()
        
        # Update model using local learning rules
        model.update(x_t_flat, x_t1_flat)
        
        # Compute update magnitudes (norm of weight change)
        update_magnitude_layer1 = torch.norm(model.layer1.W - W1_before).item()
        update_magnitude_layer2 = torch.norm(model.layer2.W - W2_before).item()
        update_magnitude_layer3 = torch.norm(model.layer3.W - W3_before).item()
        
        # Log metrics every step (all three layers)
        weight_norm_layer1 = torch.norm(model.layer1.W).item()
        weight_norm_layer2 = torch.norm(model.layer2.W).item()
        weight_norm_layer3 = torch.norm(model.layer3.W).item()
        y1_sample = model.layer1.forward(x_t_flat)
        y2_sample = model.layer2.forward(y1_sample)
        y3_sample = model.layer3.forward(y2_sample)
        activation_norm_layer1 = torch.norm(y1_sample).item()
        activation_norm_layer2 = torch.norm(y2_sample).item()
        activation_norm_layer3 = torch.norm(y3_sample).item()
        
        # Per-layer activation statistics (variance = std^2, but we track std)
        activation_mean_layer1 = y1_sample.mean().item()
        activation_mean_layer2 = y2_sample.mean().item()
        activation_mean_layer3 = y3_sample.mean().item()
        activation_std_layer1 = y1_sample.std().item()  # Activation variance proxy
        activation_std_layer2 = y2_sample.std().item()
        activation_std_layer3 = y3_sample.std().item()
        
        # Check for collapse (std < 0.1 threshold)
        if not collapse_info['layer1_collapsed'] and activation_std_layer1 < 0.1:
            collapse_info['layer1_collapsed'] = True
            collapse_info['collapse_step_layer1'] = step
            print(f"\n*** COLLAPSE DETECTED: Layer1 collapsed at step {step} (std={activation_std_layer1:.6f}) ***")
        
        if not collapse_info['layer2_collapsed'] and activation_std_layer2 < 0.1:
            collapse_info['layer2_collapsed'] = True
            collapse_info['collapse_step_layer2'] = step
            print(f"\n*** COLLAPSE DETECTED: Layer2 collapsed at step {step} (std={activation_std_layer2:.6f}) ***")
        
        if not collapse_info['layer3_collapsed'] and activation_std_layer3 < 0.1:
            collapse_info['layer3_collapsed'] = True
            collapse_info['collapse_step_layer3'] = step
            print(f"\n*** COLLAPSE DETECTED: Layer3 collapsed at step {step} (std={activation_std_layer3:.6f}) ***")
        
        # Check if all layers collapsed
        if (collapse_info['layer1_collapsed'] and 
            collapse_info['layer2_collapsed'] and 
            collapse_info['layer3_collapsed'] and
            not collapse_info['all_layers_collapsed']):
            collapse_info['all_layers_collapsed'] = True
            collapse_info['collapse_step_all'] = step
            print(f"\n" + "="*70)
            print("*** ALL LAYERS COLLAPSED - ABORTING TRAINING ***".center(70))
            print("="*70)
            print(f"All layers collapsed at step {step}")
            print(f"Layer1 collapsed at step: {collapse_info['collapse_step_layer1']}")
            print(f"Layer2 collapsed at step: {collapse_info['collapse_step_layer2']}")
            print(f"Layer3 collapsed at step: {collapse_info['collapse_step_layer3']}")
            print(f"Final activation stds:")
            print(f"  Layer1: {activation_std_layer1:.6f}")
            print(f"  Layer2: {activation_std_layer2:.6f}")
            print(f"  Layer3: {activation_std_layer3:.6f}")
            print("="*70)
            break  # Abort training
        
        training_logs['step'].append(step)
        training_logs['weight_norm_layer1'].append(weight_norm_layer1)
        training_logs['weight_norm_layer2'].append(weight_norm_layer2)
        training_logs['weight_norm_layer3'].append(weight_norm_layer3)
        training_logs['activation_norm_layer1'].append(activation_norm_layer1)
        training_logs['activation_norm_layer2'].append(activation_norm_layer2)
        training_logs['activation_norm_layer3'].append(activation_norm_layer3)
        training_logs['activation_mean_layer1'].append(activation_mean_layer1)
        training_logs['activation_mean_layer2'].append(activation_mean_layer2)
        training_logs['activation_mean_layer3'].append(activation_mean_layer3)
        training_logs['activation_std_layer1'].append(activation_std_layer1)
        training_logs['activation_std_layer2'].append(activation_std_layer2)
        training_logs['activation_std_layer3'].append(activation_std_layer3)
        training_logs['update_magnitude_layer1'].append(update_magnitude_layer1)
        training_logs['update_magnitude_layer2'].append(update_magnitude_layer2)
        training_logs['update_magnitude_layer3'].append(update_magnitude_layer3)
        
        # Safety check: no NaN in weights (all three layers)
        if (torch.isnan(model.layer1.W).any() or 
            torch.isnan(model.layer2.W).any() or 
            torch.isnan(model.layer3.W).any()):
            print(f"ERROR: NaN detected in weights at step {step}!")
            assert False, f"NaN detected in weights at step {step}"
        
        # Safety check: no NaN in activations (all three layers)
        if (torch.isnan(y1_sample).any() or 
            torch.isnan(y2_sample).any() or 
            torch.isnan(y3_sample).any()):
            print(f"ERROR: NaN detected in activations at step {step}!")
            assert False, f"NaN detected in activations at step {step}"
        
        # Print progress every 500 steps
        if step % 500 == 0:
            print(f"Step {step}/{EXPERIMENT_CONFIG['steps']} | "
                  f"||W1||={weight_norm_layer1:.4f} | ||W2||={weight_norm_layer2:.4f} | ||W3||={weight_norm_layer3:.4f} | "
                  f"y1_std={activation_std_layer1:.4f} | y2_std={activation_std_layer2:.4f} | y3_std={activation_std_layer3:.4f} | "
                  f"dW1={update_magnitude_layer1:.6f} | dW2={update_magnitude_layer2:.6f} | dW3={update_magnitude_layer3:.6f}")
    
    final_step = step
    print(f"\nTraining completed at step {final_step}.")
    
    # Save collapse information
    collapse_file = output_dir / 'collapse_info.json'
    with open(collapse_file, 'w') as f:
        json.dump(collapse_info, f, indent=2)
    print(f"Saved collapse information to {collapse_file}")
    
    # Final safety check: no NaN in weights (all three layers)
    assert (not torch.isnan(model.layer1.W).any() and 
            not torch.isnan(model.layer2.W).any() and 
            not torch.isnan(model.layer3.W).any()), \
        "ERROR: Weights contain NaN values after training!"
    
    # Save training logs
    logs_file = output_dir / 'training_logs.json'
    with open(logs_file, 'w') as f:
        json.dump(training_logs, f, indent=2)
    print(f"Saved training logs to {logs_file}")
    
    # Export activations after training (final step)
    print("\nExporting activations at final step (after training)...")
    activations_after = export_activations(model, export_dataset, num_samples=1000)
    
    # Safety check: no NaN in activations (all three layers)
    assert not torch.isnan(activations_after['layer1_activations']).any(), \
        "ERROR: NaN detected in layer1 activations after training!"
    assert not torch.isnan(activations_after['layer2_activations']).any(), \
        "ERROR: NaN detected in layer2 activations after training!"
    assert not torch.isnan(activations_after['layer3_activations']).any(), \
        "ERROR: NaN detected in layer3 activations after training!"
    
    # Safety check: activation std > 0.1 (all three layers, non-collapsed)
    layer1_std_after = activations_after['layer1_activations'].std().item()
    layer2_std_after = activations_after['layer2_activations'].std().item()
    layer3_std_after = activations_after['layer3_activations'].std().item()
    print(f"\nLayer1 activation std after training: {layer1_std_after:.6f}")
    print(f"Layer2 activation std after training: {layer2_std_after:.6f}")
    print(f"Layer3 activation std after training: {layer3_std_after:.6f}")
    
    # Explicit collapse reporting for each layer
    layer1_collapsed = layer1_std_after < 0.1
    layer2_collapsed = layer2_std_after < 0.1
    layer3_collapsed = layer3_std_after < 0.1
    
    print("\n" + "="*70)
    print("COLLAPSE STATUS".center(70))
    print("="*70)
    
    if layer1_collapsed:
        print(f"*** LAYER 1 COLLAPSED: Activation std ({layer1_std_after:.6f}) is below 0.1 threshold - REPRESENTATION COLLAPSED! ***")
        if collapse_info['collapse_step_layer1']:
            print(f"    Collapse detected during training at step: {collapse_info['collapse_step_layer1']}")
    else:
        print(f"OK: Layer1 activation std ({layer1_std_after:.6f}) is above 0.1 threshold - representation is healthy")
    
    if layer2_collapsed:
        print(f"*** LAYER 2 COLLAPSED: Activation std ({layer2_std_after:.6f}) is below 0.1 threshold - REPRESENTATION COLLAPSED! ***")
        if collapse_info['collapse_step_layer2']:
            print(f"    Collapse detected during training at step: {collapse_info['collapse_step_layer2']}")
    else:
        print(f"OK: Layer2 activation std ({layer2_std_after:.6f}) is above 0.1 threshold - representation is healthy")
    
    if layer3_collapsed:
        print(f"*** LAYER 3 COLLAPSED: Activation std ({layer3_std_after:.6f}) is below 0.1 threshold - REPRESENTATION COLLAPSED! ***")
        if collapse_info['collapse_step_layer3']:
            print(f"    Collapse detected during training at step: {collapse_info['collapse_step_layer3']}")
    else:
        print(f"OK: Layer3 activation std ({layer3_std_after:.6f}) is above 0.1 threshold - representation is healthy")
    
    if collapse_info['all_layers_collapsed']:
        print(f"\n*** ALL LAYERS COLLAPSED: Training was aborted at step {collapse_info['collapse_step_all']} ***")
    else:
        print(f"\nNo complete collapse: Training completed successfully through all {final_step} steps")
    print("="*70)
    
    torch.save(activations_after, output_dir / 'activations_after.pt')
    print(f"\nSaved activations to {output_dir / 'activations_after.pt'}")
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS".center(70))
    print("="*70)
    layer1_final = activations_after['layer1_activations']
    layer2_final = activations_after['layer2_activations']
    layer3_final = activations_after['layer3_activations']
    
    print("Layer 1 (256 units):")
    print(f"  Activation mean: {layer1_final.mean().item():.6f}")
    print(f"  Activation std:  {layer1_final.std().item():.6f}")
    print(f"  Activation min:  {layer1_final.min().item():.6f}")
    print(f"  Activation max:  {layer1_final.max().item():.6f}")
    print(f"  Weight norm:     {torch.norm(model.layer1.W).item():.6f}")
    print(f"  No NaN:          {not torch.isnan(layer1_final).any().item()}")
    print(f"  Std > 0.1:       {layer1_std_after > 0.1}")
    
    print("\nLayer 2 (128 units):")
    print(f"  Activation mean: {layer2_final.mean().item():.6f}")
    print(f"  Activation std:  {layer2_final.std().item():.6f}")
    print(f"  Activation min:  {layer2_final.min().item():.6f}")
    print(f"  Activation max:  {layer2_final.max().item():.6f}")
    print(f"  Weight norm:     {torch.norm(model.layer2.W).item():.6f}")
    print(f"  No NaN:          {not torch.isnan(layer2_final).any().item()}")
    print(f"  Std > 0.1:       {layer2_std_after > 0.1}")
    
    print("\nLayer 3 (64 units):")
    print(f"  Activation mean: {layer3_final.mean().item():.6f}")
    print(f"  Activation std:  {layer3_final.std().item():.6f}")
    print(f"  Activation min:  {layer3_final.min().item():.6f}")
    print(f"  Activation max:  {layer3_final.max().item():.6f}")
    print(f"  Weight norm:     {torch.norm(model.layer3.W).item():.6f}")
    print(f"  No NaN:          {not torch.isnan(layer3_final).any().item()}")
    print(f"  Std > 0.1:       {layer3_std_after > 0.1}")
    print("="*70)
    
    # Verify all files were created
    print("\nVerifying exported files...")
    required_files = [
        'metadata.json',
        'training_logs.json',
        'collapse_info.json',
        'activations_before.pt',
        'activations_after.pt'
    ]
    
    all_files_exist = True
    for filename in required_files:
        filepath = output_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size
            print(f"  OK: {filename} ({file_size:,} bytes)")
        else:
            print(f"  ERROR: {filename} not found!")
            all_files_exist = False
    
    if not all_files_exist:
        assert False, "Not all required files were exported!"
    
    print(f"\nExperiment completed successfully!")
    print(f"All outputs saved to: {output_dir}")
    print(f"\nGenerated files:")
    for filename in required_files:
        print(f"  - {output_dir / filename}")


if __name__ == "__main__":
    main()


