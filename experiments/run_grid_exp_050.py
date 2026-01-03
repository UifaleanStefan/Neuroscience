"""
Grid Experiment #050: Fashion-MNIST with Conv-MLP Hybrid (10k steps)

Configuration:
- Dataset: Fashion-MNIST (grayscale, 28x28 images)
- Training steps: 10,000
- Architecture: Conv-MLP Hybrid
  - Conv layer: 16 channels, kernel 5, stride 1, padding 2
  - Flatten
  - MLP head: 128 hidden units → 64 output units
- Activation: tanh (scaled) throughout
- Learning rule: Full LPL (Hebbian + Predictive + Stabilization) on MLP layers
- Temporal pairs: Translation + noise transformations
- Seed: 42
- Intermediate exports at: initialization, 5k steps, final step
"""

import torch
import torch.nn as nn
import json
import sys
import gc
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lpl_core.hierarchical_lpl import HierarchicalLPL
from data.fashion_mnist import FashionMNISTTemporalPairDataset, create_fashion_mnist_temporal_pair_dataset


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


class ConvMLPHybrid2Layer:
    """
    Conv-MLP Hybrid with 2-layer MLP head.
    
    Architecture: Conv layer → Flatten → MLP (128 → 64)
    The convolutional layer extracts spatial features from 2D images.
    The MLP head processes the flattened conv features using LPL.
    
    Design choice: The conv layer acts as a fixed feature extractor (not trained).
    This tests whether LPL can learn useful representations from random conv features.
    The conv layer uses torch.no_grad() during forward pass to prevent gradient tracking
    and save memory, but parameters remain trainable if you want to add backprop later.
    """
    
    def __init__(self, input_channels: int, input_size: int, 
                 conv_out_channels: int, conv_kernel_size: int,
                 conv_stride: int, conv_padding: int,
                 mlp_hidden: int, mlp_out: int,
                 cfg_layer1, cfg_layer2):
        """
        Initialize Conv-MLP Hybrid model with 2-layer MLP.
        
        Args:
            input_channels: Number of input channels (1 for grayscale)
            input_size: Input image size (28 for 28x28)
            conv_out_channels: Number of output channels from conv layer
            conv_kernel_size: Kernel size for conv layer
            conv_stride: Stride for conv layer
            conv_padding: Padding for conv layer
            mlp_hidden: Hidden layer dimension (128)
            mlp_out: Output dimension (64)
            cfg_layer1: Configuration for first MLP layer
            cfg_layer2: Configuration for second MLP layer
        """
        self.input_channels = input_channels
        self.input_size = input_size
        self.conv_out_channels = conv_out_channels
        
        # Create convolutional layer
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding
        )
        
        # Note: Conv layer is not trained with LPL rules (acts as fixed feature extractor)
        # We use torch.no_grad() during forward pass to prevent gradient tracking
        # but keep requires_grad=True in case you want to train it with backprop later
        
        # Calculate flattened conv output size
        # With padding=2, kernel=5, stride=1: output size = input_size (28x28)
        conv_output_size = conv_out_channels * input_size * input_size
        
        # Create 2-layer MLP using HierarchicalLPL
        self.mlp = HierarchicalLPL(
            d_in=conv_output_size,
            d_hidden=mlp_hidden,
            d_out=mlp_out,
            cfg_layer1=cfg_layer1,
            cfg_layer2=cfg_layer2
        )
        
        self.d_out = mlp_out
    
    def forward(self, x: torch.Tensor, return_conv_features=False):
        """
        Forward pass through conv and MLP layers.
        
        Args:
            x: Input tensor of shape (H, W) for grayscale image, or (C, H, W)
            return_conv_features: If True, also return conv feature maps
            
        Returns:
            Output tensor from MLP of shape (d_out,)
            If return_conv_features=True, also returns conv feature maps
        """
        # Ensure x is 3D: (C, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add channel dimension: (1, H, W)
        elif x.dim() == 1:
            # If flattened, reshape to 2D then add channel
            size = int(x.shape[0] ** 0.5)
            x = x.reshape(1, size, size)
        
        # Conv forward
        conv_out = self.conv(x)  # (C_out, H, W)
        
        # Apply tanh activation to conv output (as specified)
        conv_out = torch.tanh(conv_out)
        
        # Flatten conv output
        conv_flat = conv_out.flatten()  # (C_out * H * W,)
        
        # MLP forward (includes tanh activation in LPL layers)
        y1, y2 = self.mlp.get_activations(conv_flat)
        
        if return_conv_features:
            return y2, conv_out  # Return final output and conv features
        return y2
    
    def update(self, x_t: torch.Tensor, x_t1: torch.Tensor):
        """
        Update model using local learning rules.
        
        Updates the MLP layers using Full LPL rules.
        The conv layer acts as feature extraction (not updated with LPL rules).
        
        Args:
            x_t: Input tensor at time t (2D image of shape (H, W) or flattened)
            x_t1: Input tensor at time t+1 (2D image of shape (H, W) or flattened)
        """
        # Process inputs through conv layer to get features
        # Handle different input formats
        if x_t.dim() == 1:
            # Flattened input - reshape to 2D
            size = int(x_t.shape[0] ** 0.5)
            x_t = x_t.reshape(size, size)
            x_t1 = x_t1.reshape(size, size)
        
        # Ensure x_t and x_t1 are 3D: (C, H, W)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)  # (1, H, W)
            x_t1 = x_t1.unsqueeze(0)  # (1, H, W)
        
        # Forward through conv (with tanh) - use no_grad to prevent gradient tracking
        # This saves memory by not building computation graph for conv operations
        with torch.no_grad():
            conv_t = torch.tanh(self.conv(x_t))
            conv_t1 = torch.tanh(self.conv(x_t1))
            
            # Flatten conv outputs immediately to reduce memory
            conv_flat_t = conv_t.flatten().detach()  # detach() ensures no gradient connection
            conv_flat_t1 = conv_t1.flatten().detach()
        
        # Clear intermediate conv tensors to save memory
        del conv_t, conv_t1
        
        # Update MLP layers using local learning rules (Full LPL)
        self.mlp.update(conv_flat_t, conv_flat_t1)
        
        # Clear flattened tensors
        del conv_flat_t, conv_flat_t1
        
        # Explicitly clear any cached intermediate values
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def export_activations(model, dataset, num_samples=1000):
    """
    Export activations from conv features and MLP layers.
    
    Args:
        model: ConvMLPHybrid2Layer model
        dataset: FashionMNISTTemporalPairDataset (can be temporal pair or single image mode)
        num_samples: Number of samples to export
        
    Returns:
        Dictionary with 'conv_features', 'mlp_layer1_activations', 'mlp_layer2_activations', and 'labels'
    """
    conv_features_list = []
    mlp_layer1_activations = []
    mlp_layer2_activations = []
    labels = []
    
    for i in range(min(num_samples, len(dataset))):
        # Handle both temporal pair and single image modes
        if hasattr(dataset, 'return_temporal_pair') and dataset.return_temporal_pair:
            x_t, _, label = dataset[i]
            image = x_t
        else:
            image, label = dataset[i]
        
        # Image should be 2D (H, W) from Fashion-MNIST
        # Ensure input is float32 and in [0,1] range
        if image.dtype != torch.float32:
            image = image.float()
        image = torch.clamp(image, 0.0, 1.0)
        
        # Forward pass to get activations
        # Ensure image is 3D: (C, H, W)
        image_3d = image.unsqueeze(0) if image.dim() == 2 else image
        
        with torch.no_grad():
            # Get conv features
            conv_feat = torch.tanh(model.conv(image_3d))
            conv_flat = conv_feat.flatten()
            
            # Get MLP activations
            y1, y2 = model.mlp.get_activations(conv_flat)
        
        # Check for NaN in activations
        if torch.isnan(conv_feat).any():
            print(f"WARNING: NaN detected in conv features at sample {i}")
            continue
        if torch.isnan(y1).any():
            print(f"WARNING: NaN detected in MLP layer1 activation at sample {i}")
            continue
        if torch.isnan(y2).any():
            print(f"WARNING: NaN detected in MLP layer2 activation at sample {i}")
            continue
        
        conv_features_list.append(conv_feat)
        mlp_layer1_activations.append(y1)
        mlp_layer2_activations.append(y2)
        labels.append(label)
    
    return {
        'conv_features': torch.stack(conv_features_list),
        'mlp_layer1_activations': torch.stack(mlp_layer1_activations),
        'mlp_layer2_activations': torch.stack(mlp_layer2_activations),
        'labels': torch.tensor(labels)
    }


def main():
    """
    Run grid experiment #050.
    """
    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Experiment configuration
    EXPERIMENT_CONFIG = {
        'dataset': 'fashion_mnist',
        'steps': 10000,
        'architecture': 'conv_mlp_hybrid',
        'activation': 'tanh',
        'rule': 'full_lpl',
        'baseline': 'none',
        'input_channels': 1,        # Grayscale
        'input_size': 28,           # 28x28 images
        'conv_out_channels': 16,   # Number of conv output channels
        'conv_kernel_size': 5,      # 5x5 conv kernel
        'conv_stride': 1,           # Stride 1
        'conv_padding': 2,          # Padding 2
        'mlp_hidden': 128,          # MLP hidden layer dimension
        'mlp_out': 64,              # MLP output dimension
        'lr_hebb': 0.001,
        'lr_pred': 0.001,
        'lr_stab': 0.0005,
        'seed': 42,
        'translate_range': 2,
        'noise_std': 0.05
    }
    
    print("="*70)
    print("GRID EXPERIMENT #050".center(70))
    print("="*70)
    print(f"Dataset: {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps: {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture: {EXPERIMENT_CONFIG['architecture']}")
    print(f"  Conv: {EXPERIMENT_CONFIG['conv_out_channels']} channels, "
          f"kernel={EXPERIMENT_CONFIG['conv_kernel_size']}, "
          f"stride={EXPERIMENT_CONFIG['conv_stride']}, "
          f"padding={EXPERIMENT_CONFIG['conv_padding']}")
    print(f"  MLP: {EXPERIMENT_CONFIG['mlp_hidden']} -> {EXPERIMENT_CONFIG['mlp_out']} units")
    print(f"Activation: {EXPERIMENT_CONFIG['activation']}")
    print(f"Rule: {EXPERIMENT_CONFIG['rule']}")
    print(f"Baseline: {EXPERIMENT_CONFIG['baseline']}")
    print("="*70)
    
    # Create output directory with experiment identifier
    output_base = Path('outputs/grid_experiments')
    output_dir = output_base / "run_050_fashion_mnist_10000steps_conv_mlp_tanh_full_lpl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(EXPERIMENT_CONFIG, f, indent=2)
    print(f"\nMetadata saved to {metadata_file}")
    
    # Create layer configuration (full LPL: all rules enabled)
    # Same configuration for both MLP layers
    mlp_cfg_layer1 = LayerConfig(
        lr_hebb=EXPERIMENT_CONFIG['lr_hebb'],
        lr_pred=EXPERIMENT_CONFIG['lr_pred'],
        lr_stab=EXPERIMENT_CONFIG['lr_stab'],
        use_hebb=True,   # Full LPL
        use_pred=True,   # Full LPL
        use_stab=True    # Full LPL
    )
    
    mlp_cfg_layer2 = LayerConfig(
        lr_hebb=EXPERIMENT_CONFIG['lr_hebb'],
        lr_pred=EXPERIMENT_CONFIG['lr_pred'],
        lr_stab=EXPERIMENT_CONFIG['lr_stab'],
        use_hebb=True,   # Full LPL
        use_pred=True,   # Full LPL
        use_stab=True    # Full LPL
    )
    
    # Create model (Conv-MLP Hybrid with 2-layer MLP)
    model = ConvMLPHybrid2Layer(
        input_channels=EXPERIMENT_CONFIG['input_channels'],
        input_size=EXPERIMENT_CONFIG['input_size'],
        conv_out_channels=EXPERIMENT_CONFIG['conv_out_channels'],
        conv_kernel_size=EXPERIMENT_CONFIG['conv_kernel_size'],
        conv_stride=EXPERIMENT_CONFIG['conv_stride'],
        conv_padding=EXPERIMENT_CONFIG['conv_padding'],
        mlp_hidden=EXPERIMENT_CONFIG['mlp_hidden'],
        mlp_out=EXPERIMENT_CONFIG['mlp_out'],
        cfg_layer1=mlp_cfg_layer1,
        cfg_layer2=mlp_cfg_layer2
    )
    
    # Create datasets
    # For activation export: single images (not temporal pairs)
    export_dataset = FashionMNISTTemporalPairDataset(
        train=True,
        return_temporal_pair=False,
        translate_range=EXPERIMENT_CONFIG['translate_range'],
        noise_std=EXPERIMENT_CONFIG['noise_std'],
        seed=EXPERIMENT_CONFIG['seed']
    )
    
    # For training: temporal pairs
    train_dataset = create_fashion_mnist_temporal_pair_dataset(
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
    
    # Safety check: no NaN in activations
    assert not torch.isnan(activations_before['conv_features']).any(), \
        "ERROR: NaN detected in conv features before training!"
    assert not torch.isnan(activations_before['mlp_layer1_activations']).any(), \
        "ERROR: NaN detected in MLP layer1 activations before training!"
    assert not torch.isnan(activations_before['mlp_layer2_activations']).any(), \
        "ERROR: NaN detected in MLP layer2 activations before training!"
    
    # Safety check: activation std > 0.1 (variance check)
    conv_std_before = activations_before['conv_features'].std().item()
    mlp1_std_before = activations_before['mlp_layer1_activations'].std().item()
    mlp2_std_before = activations_before['mlp_layer2_activations'].std().item()
    print(f"Conv features std before training: {conv_std_before:.6f}")
    print(f"MLP Layer1 activation std before training: {mlp1_std_before:.6f}")
    print(f"MLP Layer2 activation std before training: {mlp2_std_before:.6f}")
    if conv_std_before < 0.1:
        print(f"WARNING: Conv features std ({conv_std_before:.6f}) is below 0.1 threshold!")
    if mlp1_std_before < 0.1:
        print(f"WARNING: MLP Layer1 activation std ({mlp1_std_before:.6f}) is below 0.1 threshold!")
    if mlp2_std_before < 0.1:
        print(f"WARNING: MLP Layer2 activation std ({mlp2_std_before:.6f}) is below 0.1 threshold!")
    
    torch.save(activations_before, output_dir / 'activations_before.pt')
    print(f"Saved activations to {output_dir / 'activations_before.pt'}")
    
    # Initialize training logs with per-layer activation statistics and weight norms
    training_logs = {
        'step': [],
        'weight_norm_conv': [],  # Conv layer weight norm (for tracking, even though not trained)
        'weight_norm_mlp_layer1': [],
        'weight_norm_mlp_layer2': [],
        'activation_norm_mlp_layer1': [],
        'activation_norm_mlp_layer2': [],
        'activation_mean_mlp_layer1': [],
        'activation_mean_mlp_layer2': [],
        'activation_std_mlp_layer1': [],  # Activation variance (std)
        'activation_std_mlp_layer2': []
    }
    
    # Log interval to reduce memory usage (log every 10 steps instead of every step)
    log_interval = 10
    
    # Training loop
    print(f"\nTraining LPL for {EXPERIMENT_CONFIG['steps']} steps...")
    
    try:
        for step in range(1, EXPERIMENT_CONFIG['steps'] + 1):
            # Sample a temporal pair from the dataset
            idx = torch.randint(0, len(train_dataset), (1,)).item()
            x_t, x_t1, _ = train_dataset[idx]
            
            # Ensure inputs are floats in [0,1] range (images are 2D: H, W)
            if x_t.dtype != torch.float32:
                x_t = x_t.float()
            if x_t1.dtype != torch.float32:
                x_t1 = x_t1.float()
            x_t = torch.clamp(x_t, 0.0, 1.0)
            x_t1 = torch.clamp(x_t1, 0.0, 1.0)
            
            # Update model using local learning rules (MLP layers update)
            try:
                model.update(x_t, x_t1)
            except Exception as e:
                print(f"\nERROR: Exception during model.update() at step {step}:")
                print(f"  {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Safety check: no NaN in weights (MLP layers) - always check
            if (torch.isnan(model.mlp.layer1.W).any() or 
                torch.isnan(model.mlp.layer2.W).any()):
                print(f"ERROR: NaN detected in weights at step {step}!")
                assert False, f"NaN detected in weights at step {step}"
            
            # Log metrics periodically to reduce memory usage
            if step % log_interval == 0 or step == 1:
                # Compute weight norms separately for conv and MLP layers
                weight_norm_conv = torch.norm(model.conv.weight).item()
                weight_norm_mlp_layer1 = torch.norm(model.mlp.layer1.W).item()
                weight_norm_mlp_layer2 = torch.norm(model.mlp.layer2.W).item()
                
                # Get activations for logging
                x_t_3d = x_t.unsqueeze(0) if x_t.dim() == 2 else x_t
                with torch.no_grad():
                    conv_out = torch.tanh(model.conv(x_t_3d))
                    conv_flat = conv_out.flatten()
                    y1_sample, y2_sample = model.mlp.get_activations(conv_flat)
                    
                    activation_norm_mlp_layer1 = torch.norm(y1_sample).item()
                    activation_norm_mlp_layer2 = torch.norm(y2_sample).item()
                    
                    # Per-layer activation statistics
                    activation_mean_mlp_layer1 = y1_sample.mean().item()
                    activation_mean_mlp_layer2 = y2_sample.mean().item()
                    activation_std_mlp_layer1 = y1_sample.std().item()
                    activation_std_mlp_layer2 = y2_sample.std().item()
                    
                    training_logs['step'].append(step)
                    training_logs['weight_norm_conv'].append(weight_norm_conv)
                    training_logs['weight_norm_mlp_layer1'].append(weight_norm_mlp_layer1)
                    training_logs['weight_norm_mlp_layer2'].append(weight_norm_mlp_layer2)
                    training_logs['activation_norm_mlp_layer1'].append(activation_norm_mlp_layer1)
                    training_logs['activation_norm_mlp_layer2'].append(activation_norm_mlp_layer2)
                    training_logs['activation_mean_mlp_layer1'].append(activation_mean_mlp_layer1)
                    training_logs['activation_mean_mlp_layer2'].append(activation_mean_mlp_layer2)
                    training_logs['activation_std_mlp_layer1'].append(activation_std_mlp_layer1)
                    training_logs['activation_std_mlp_layer2'].append(activation_std_mlp_layer2)
                    
                    # Safety check: no NaN in activations (only when logging to save memory)
                    if (torch.isnan(y1_sample).any() or 
                        torch.isnan(y2_sample).any()):
                        print(f"ERROR: NaN detected in activations at step {step}!")
                        assert False, f"NaN detected in activations at step {step}"
                    
                    # Clear intermediate tensors to save memory
                    del conv_out, conv_flat, y1_sample, y2_sample, x_t_3d
            
            # Clear input tensors to save memory
            del x_t, x_t1
            
            # Export intermediate activations at 5k steps
            if step == 5000:
                print(f"\nExporting activations at {step} steps (midpoint checkpoint)...")
                activations_midpoint = export_activations(model, export_dataset, num_samples=1000)
                
                # Safety check: no NaN in activations
                assert not torch.isnan(activations_midpoint['conv_features']).any(), \
                    f"ERROR: NaN detected in conv features at step {step}!"
                assert not torch.isnan(activations_midpoint['mlp_layer1_activations']).any(), \
                    f"ERROR: NaN detected in MLP layer1 activations at step {step}!"
                assert not torch.isnan(activations_midpoint['mlp_layer2_activations']).any(), \
                    f"ERROR: NaN detected in MLP layer2 activations at step {step}!"
                
                # Safety check: activation std > 0.1 (variance check)
                conv_std_mid = activations_midpoint['conv_features'].std().item()
                mlp1_std_mid = activations_midpoint['mlp_layer1_activations'].std().item()
                mlp2_std_mid = activations_midpoint['mlp_layer2_activations'].std().item()
                print(f"Conv features std at {step} steps: {conv_std_mid:.6f}")
                print(f"MLP Layer1 activation std at {step} steps: {mlp1_std_mid:.6f}")
                print(f"MLP Layer2 activation std at {step} steps: {mlp2_std_mid:.6f}")
                
                torch.save(activations_midpoint, output_dir / 'activations_5000steps.pt')
                print(f"Saved activations to {output_dir / 'activations_5000steps.pt'}")
            
            # Print progress every 100 steps
            if step % 100 == 0:
                # Compute metrics for display
                weight_norm_conv = torch.norm(model.conv.weight).item()
                weight_norm_mlp_layer1 = torch.norm(model.mlp.layer1.W).item()
                weight_norm_mlp_layer2 = torch.norm(model.mlp.layer2.W).item()
                
                # Sample a random image for activation norm display
                sample_idx = torch.randint(0, len(train_dataset), (1,)).item()
                sample_img, _, _ = train_dataset[sample_idx]
                if sample_img.dtype != torch.float32:
                    sample_img = sample_img.float()
                sample_img = torch.clamp(sample_img, 0.0, 1.0)
                sample_img_3d = sample_img.unsqueeze(0) if sample_img.dim() == 2 else sample_img
                with torch.no_grad():
                    conv_temp = torch.tanh(model.conv(sample_img_3d))
                    conv_flat_temp = conv_temp.flatten()
                    y1_temp, y2_temp = model.mlp.get_activations(conv_flat_temp)
                activation_norm_mlp_layer1 = torch.norm(y1_temp).item()
                activation_norm_mlp_layer2 = torch.norm(y2_temp).item()
                activation_std_mlp_layer1 = y1_temp.std().item()
                activation_std_mlp_layer2 = y2_temp.std().item()
                
                print(f"Step {step}/{EXPERIMENT_CONFIG['steps']} | "
                      f"||W_conv||={weight_norm_conv:.4f} | "
                      f"||W1||={weight_norm_mlp_layer1:.4f} | ||W2||={weight_norm_mlp_layer2:.4f} | "
                      f"||y1||={activation_norm_mlp_layer1:.4f} | ||y2||={activation_norm_mlp_layer2:.4f} | "
                      f"y1_std={activation_std_mlp_layer1:.4f} | y2_std={activation_std_mlp_layer2:.4f}")
                
                # Clear temporary tensors
                del sample_img, sample_img_3d, conv_temp, conv_flat_temp, y1_temp, y2_temp
                
                # Periodic garbage collection to prevent memory buildup
                gc.collect()
    
    except Exception as e:
        print(f"\nFATAL ERROR: Training crashed at step {step if 'step' in locals() else 'unknown'}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial training logs
        if 'training_logs' in locals() and len(training_logs['step']) > 0:
            logs_file = output_dir / 'training_logs_partial.json'
            with open(logs_file, 'w') as f:
                json.dump(training_logs, f, indent=2)
            print(f"\nSaved partial training logs to {logs_file}")
        
        raise
    
    print("Training completed.")
    
    # Final safety check: no NaN in weights (MLP layers)
    assert (not torch.isnan(model.mlp.layer1.W).any() and 
            not torch.isnan(model.mlp.layer2.W).any()), \
        "ERROR: Weights contain NaN values after training!"
    
    # Save training logs
    logs_file = output_dir / 'training_logs.json'
    with open(logs_file, 'w') as f:
        json.dump(training_logs, f, indent=2)
    print(f"\nSaved training logs to {logs_file}")
    
    # Export activations after training (final step)
    print("\nExporting activations after training (final step)...")
    activations_after = export_activations(model, export_dataset, num_samples=1000)
    
    # Safety check: no NaN in activations
    assert not torch.isnan(activations_after['conv_features']).any(), \
        "ERROR: NaN detected in conv features after training!"
    assert not torch.isnan(activations_after['mlp_layer1_activations']).any(), \
        "ERROR: NaN detected in MLP layer1 activations after training!"
    assert not torch.isnan(activations_after['mlp_layer2_activations']).any(), \
        "ERROR: NaN detected in MLP layer2 activations after training!"
    
    # Safety check: activation std > 0.1 (variance check, non-collapsed)
    conv_std_after = activations_after['conv_features'].std().item()
    mlp1_std_after = activations_after['mlp_layer1_activations'].std().item()
    mlp2_std_after = activations_after['mlp_layer2_activations'].std().item()
    print(f"\nConv features std after training: {conv_std_after:.6f}")
    print(f"MLP Layer1 activation std after training: {mlp1_std_after:.6f}")
    print(f"MLP Layer2 activation std after training: {mlp2_std_after:.6f}")
    
    # Explicit collapse reporting
    conv_collapsed = conv_std_after < 0.1
    mlp1_collapsed = mlp1_std_after < 0.1
    mlp2_collapsed = mlp2_std_after < 0.1
    
    if conv_collapsed:
        print(f"*** CONV FEATURES COLLAPSED: Std ({conv_std_after:.6f}) is below 0.1 threshold! ***")
    else:
        print(f"OK: Conv features std ({conv_std_after:.6f}) is above 0.1 threshold - healthy")
    
    if mlp1_collapsed:
        print(f"*** MLP LAYER 1 COLLAPSED: Activation std ({mlp1_std_after:.6f}) is below 0.1 threshold - REPRESENTATION COLLAPSED! ***")
    else:
        print(f"OK: MLP Layer1 activation std ({mlp1_std_after:.6f}) is above 0.1 threshold - representation is healthy")
    
    if mlp2_collapsed:
        print(f"*** MLP LAYER 2 COLLAPSED: Activation std ({mlp2_std_after:.6f}) is below 0.1 threshold - REPRESENTATION COLLAPSED! ***")
    else:
        print(f"OK: MLP Layer2 activation std ({mlp2_std_after:.6f}) is above 0.1 threshold - representation is healthy")
    
    torch.save(activations_after, output_dir / 'activations_after.pt')
    print(f"Saved activations to {output_dir / 'activations_after.pt'}")
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS".center(70))
    print("="*70)
    conv_final = activations_after['conv_features']
    mlp1_final = activations_after['mlp_layer1_activations']
    mlp2_final = activations_after['mlp_layer2_activations']
    
    # Compute final weight norms
    weight_norm_conv_final = torch.norm(model.conv.weight).item()
    weight_norm_mlp1_final = torch.norm(model.mlp.layer1.W).item()
    weight_norm_mlp2_final = torch.norm(model.mlp.layer2.W).item()
    
    print("Conv Layer:")
    print(f"  Weight norm:     {weight_norm_conv_final:.6f}")
    print(f"  Activation mean: {conv_final.mean().item():.6f}")
    print(f"  Activation std:  {conv_final.std().item():.6f}")
    print(f"  Activation min:  {conv_final.min().item():.6f}")
    print(f"  Activation max:  {conv_final.max().item():.6f}")
    print(f"  Shape:           {conv_final.shape}")
    print(f"  Std > 0.1:       {conv_std_after > 0.1}")
    
    print("\nMLP Layer 1 (128 units):")
    print(f"  Weight norm:     {weight_norm_mlp1_final:.6f}")
    print(f"  Activation mean: {mlp1_final.mean().item():.6f}")
    print(f"  Activation std:  {mlp1_final.std().item():.6f}")
    print(f"  Activation min:  {mlp1_final.min().item():.6f}")
    print(f"  Activation max:  {mlp1_final.max().item():.6f}")
    print(f"  No NaN:          {not torch.isnan(mlp1_final).any().item()}")
    print(f"  Std > 0.1:       {mlp1_std_after > 0.1}")
    
    print("\nMLP Layer 2 (64 units):")
    print(f"  Weight norm:     {weight_norm_mlp2_final:.6f}")
    print(f"  Activation mean: {mlp2_final.mean().item():.6f}")
    print(f"  Activation std:  {mlp2_final.std().item():.6f}")
    print(f"  Activation min:  {mlp2_final.min().item():.6f}")
    print(f"  Activation max:  {mlp2_final.max().item():.6f}")
    print(f"  No NaN:          {not torch.isnan(mlp2_final).any().item()}")
    print(f"  Std > 0.1:       {mlp2_std_after > 0.1}")
    print("="*70)
    
    # Verify all files were created
    print("\nVerifying exported files...")
    required_files = [
        'metadata.json',
        'training_logs.json',
        'activations_before.pt',
        'activations_5000steps.pt',
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

