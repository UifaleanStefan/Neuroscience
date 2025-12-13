"""Ablation study experiments for LPL."""

import torch
import torchvision
import torchvision.transforms as transforms
import sys
from pathlib import Path

# Add project root to path to import lpl_core
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from lpl_core.lpl_layer import LPLLayer


class MinimalConfig:
    """Minimal configuration object for LPL layer."""
    def __init__(self):
        # Reduced learning rates to prevent numerical instability
        self.lr_hebb = 0.001
        self.lr_pred = 0.001
        # Further reduced lr_stab for stabilization ablation stability
        self.lr_stab = 0.0005
        self.use_hebb = True
        self.use_pred = True
        self.use_stab = True


def weak_augmentation():
    """
    Create weak augmentation function for temporal correlation.
    
    Applies:
    - Random crop with padding
    - Small random horizontal translation
    - Additive Gaussian noise
    """
    def augment(image):
        # Random crop with padding (padding=4, crop back to 32x32)
        padded = transforms.functional.pad(image, padding=4, padding_mode='reflect')
        x = torch.randint(0, 9, (1,)).item()
        y = torch.randint(0, 9, (1,)).item()
        cropped = transforms.functional.crop(padded, x, y, 32, 32)
        
        # Small random horizontal translation (±2 pixels)
        translate_x = torch.randint(-2, 3, (1,)).item()
        translated = transforms.functional.affine(
            cropped, angle=0, translate=(translate_x, 0), scale=1.0, shear=0
        )
        
        # Additive Gaussian noise
        noise = torch.randn_like(translated) * 0.05
        noisy = translated + noise
        noisy = torch.clamp(noisy, 0.0, 1.0)
        
        return noisy
    
    return augment


def export_activations(layer, dataset, num_samples=1000):
    """
    Export activations and labels for a set of samples.
    
    Args:
        layer: LPLLayer instance
        dataset: CIFAR-10 dataset
        num_samples: Number of samples to export
        
    Returns:
        Dictionary with 'activations' and 'labels' tensors
    """
    activations = []
    labels = []
    
    for i in range(num_samples):
        image, label = dataset[i]
        # Convert to tensor if needed (dataset returns PIL or tensor)
        if not isinstance(image, torch.Tensor):
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        # Flatten image to 1D tensor
        x = image.flatten()
        
        # Ensure input is float32 and in [0,1] range
        if x.dtype != torch.float32:
            x = x.float()
        x = torch.clamp(x, 0.0, 1.0)
        
        # Forward pass to get representation
        y = layer.forward(x)
        
        # Check for NaN in activations
        if torch.isnan(y).any():
            print(f"WARNING: NaN detected in activation at sample {i}")
        
        activations.append(y)
        labels.append(label)
    
    return {
        'activations': torch.stack(activations),
        'labels': torch.tensor(labels)
    }


def run_ablation_condition(name, cfg, trainset, augment, num_steps=5000, shuffle_temporal=False):
    """
    Run a single ablation condition: train and export activations.
    
    Args:
        name: Name of the ablation condition (for saving)
        cfg: Configuration object
        trainset: CIFAR-10 training dataset
        augment: Augmentation function
        num_steps: Number of training steps
        shuffle_temporal: If True, break temporal correlation by shuffling x_t1
    """
    print(f"\n{'='*60}")
    print(f"Running ablation: {name}")
    print(f"{'='*60}")
    
    # CIFAR-10 dimensions: 32x32x3 = 3072
    d_in = 32 * 32 * 3
    d_out = 128  # Representation dimension
    
    # Create new LPL layer instance for this condition
    layer = LPLLayer(d_in=d_in, d_out=d_out, cfg=cfg)
    
    # Training loop
    print(f"Training for {num_steps} steps...")
    dataset_size = len(trainset)
    
    for step in range(1, num_steps + 1):
        # Sample a random image for x_t
        idx_t = torch.randint(0, dataset_size, (1,)).item()
        image_t, _ = trainset[idx_t]
        x_t = augment(image_t).flatten()
        
        if shuffle_temporal:
            # Break temporal correlation: use a different random image for x_t1
            idx_t1 = torch.randint(0, dataset_size, (1,)).item()
            image_t1, _ = trainset[idx_t1]
            x_t1 = augment(image_t1).flatten()
        else:
            # Normal temporal correlation: use augmented view of same image
            x_t1 = augment(image_t).flatten()
        
        # Ensure inputs are floats in [0,1] range
        assert x_t.dtype == torch.float32, f"x_t dtype is {x_t.dtype}, expected float32"
        assert x_t1.dtype == torch.float32, f"x_t1 dtype is {x_t1.dtype}, expected float32"
        assert (x_t >= 0.0).all() and (x_t <= 1.0).all(), "x_t values must be in [0,1]"
        assert (x_t1 >= 0.0).all() and (x_t1 <= 1.0).all(), "x_t1 values must be in [0,1]"
        
        # Update layer weights using local learning rules
        layer.update(x_t, x_t1)
        
        # Check for NaN values in weights (early detection of instability)
        if step % 100 == 0:
            if torch.isnan(layer.W).any():
                print(f"WARNING: NaN detected in weights at step {step}!")
                break
            if torch.isnan(layer.predictor.P).any():
                print(f"WARNING: NaN detected in predictor at step {step}!")
                break
        
        # Print progress every 1000 steps
        if step % 1000 == 0:
            # Print diagnostic info
            w_norm = torch.norm(layer.W).item()
            y_sample = layer.forward(x_t)
            y_norm = torch.norm(y_sample).item()
            print(f"Step {step}/{num_steps} | ||W||={w_norm:.4f} | ||y||={y_norm:.4f}")
    
    print("Training completed.")
    
    # Check for NaN before exporting
    if torch.isnan(layer.W).any():
        print(f"ERROR: Weights contain NaN values! Cannot export activations for {name}.")
        return
    
    # Export activations after training
    print("Exporting activations...")
    activations = export_activations(layer, trainset, num_samples=1000)
    
    # Check exported activations for NaN
    if torch.isnan(activations['activations']).any():
        print(f"ERROR: Exported activations contain NaN values for {name}!")
        return
    
    # Save to outputs/activations/
    output_dir = Path('outputs/activations')
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'activations_ablation_{name}.pt'
    torch.save(activations, output_dir / filename)
    print(f"Saved activations to {output_dir / filename}")
    print(f"Activation stats: mean={activations['activations'].mean().item():.6f}, "
          f"std={activations['activations'].std().item():.6f}")


def main(num_steps=5000):
    """
    Run all ablation experiments on CIFAR-10.
    
    Args:
        num_steps: Number of training steps per condition (default: 5000)
    """
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Load CIFAR-10 training set
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Create augmentation function
    augment = weak_augmentation()
    
    # Ablation 1: Hebbian term disabled
    cfg_hebb = MinimalConfig()
    cfg_hebb.use_hebb = False
    run_ablation_condition('hebb', cfg_hebb, trainset, augment, num_steps)
    
    # Ablation 2: Predictive term disabled
    cfg_pred = MinimalConfig()
    cfg_pred.use_pred = False
    run_ablation_condition('pred', cfg_pred, trainset, augment, num_steps)
    
    # Ablation 3: Stabilization term disabled
    cfg_stab = MinimalConfig()
    cfg_stab.use_stab = False
    run_ablation_condition('stab', cfg_stab, trainset, augment, num_steps)
    
    # Ablation 4: Random temporal pairing (shuffled x_t1)
    cfg_shuffle = MinimalConfig()  # All terms enabled, but temporal correlation broken
    run_ablation_condition('shuffle', cfg_shuffle, trainset, augment, num_steps, shuffle_temporal=True)
    
    print(f"\n{'='*60}")
    print("All ablation experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
