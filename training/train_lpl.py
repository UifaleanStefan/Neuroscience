"""Training script for Latent Predictive Learning."""

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
        # Reduced lr_stab for better stability
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


def export_activations(layer, dataset, num_samples=1000, device='cpu'):
    """
    Export activations and labels for a set of samples.
    
    Args:
        layer: LPLLayer instance
        dataset: CIFAR-10 dataset
        num_samples: Number of samples to export
        device: Device to use
        
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


def main(num_steps=1000):
    """
    Train LPL layer on CIFAR-10 with temporally correlated views.
    
    Args:
        num_steps: Number of training steps (default: 10000)
    """
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Create configuration
    cfg = MinimalConfig()
    
    # CIFAR-10 dimensions: 32x32x3 = 3072
    d_in = 32 * 32 * 3
    d_out = 128  # Representation dimension
    
    # Create LPL layer
    layer = LPLLayer(d_in=d_in, d_out=d_out, cfg=cfg)
    
    # Load CIFAR-10 training set
    transform = transforms.ToTensor()  # Convert PIL to tensor
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Create augmentation function
    augment = weak_augmentation()
    
    # Export activations before training
    print("Exporting activations before training...")
    activations_before = export_activations(layer, trainset, num_samples=1000)
    output_dir = Path('outputs/activations')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(activations_before, output_dir / 'activations_before.pt')
    print(f"Saved activations to {output_dir / 'activations_before.pt'}")
    
    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    dataset_size = len(trainset)
    
    for step in range(1, num_steps + 1):
        # Sample a random image
        idx = torch.randint(0, dataset_size, (1,)).item()
        image, _ = trainset[idx]
        
        # Create two augmented views: x_t and x_t1
        x_t = augment(image).flatten()
        x_t1 = augment(image).flatten()
        
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
        print("ERROR: Weights contain NaN values! Cannot export activations.")
        return
    
    # Export activations after training
    print("\nExporting activations after training...")
    activations_after = export_activations(layer, trainset, num_samples=1000)
    
    # Check exported activations for NaN
    if torch.isnan(activations_after['activations']).any():
        print("ERROR: Exported activations contain NaN values!")
        return
    
    torch.save(activations_after, output_dir / 'activations_after.pt')
    print(f"Saved activations to {output_dir / 'activations_after.pt'}")
    print(f"Activation stats: mean={activations_after['activations'].mean().item():.6f}, "
          f"std={activations_after['activations'].std().item():.6f}")


if __name__ == "__main__":
    main()
