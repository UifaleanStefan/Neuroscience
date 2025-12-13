"""Swap experiment runner for LPL."""

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
        self.lr_hebb = 0.01
        self.lr_pred = 0.01
        self.lr_stab = 0.01
        self.use_hebb = True
        self.use_pred = True
        self.use_stab = True


def weak_augmentation_with_swap_info():
    """
    Create augmentation function that returns both augmented image and swap info.
    
    Returns:
        Function that returns (augmented_image, translate_x) tuple
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
        
        return noisy, translate_x
    
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
        
        # Forward pass to get representation
        y = layer.forward(x)
        
        activations.append(y)
        labels.append(label)
    
    return {
        'activations': torch.stack(activations),
        'labels': torch.tensor(labels)
    }


def main(num_steps=5000):
    """
    Run swap exposure experiment (Li & DiCarlo 2008 replication).
    
    Args:
        num_steps: Number of training steps (default: 5000)
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
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Define two objects: A and B (using two CIFAR-10 classes)
    # Object A: class 0 (airplane), Object B: class 1 (automobile)
    class_a = 0
    class_b = 1
    
    # Filter dataset to get only objects A and B
    indices_a = [i for i in range(len(trainset)) if trainset[i][1] == class_a]
    indices_b = [i for i in range(len(trainset)) if trainset[i][1] == class_b]
    
    # Create augmentation function
    augment = weak_augmentation_with_swap_info()
    
    # Export activations before training
    print("Exporting activations before training...")
    activations_before = export_activations(layer, trainset, num_samples=1000)
    
    # Training loop
    print(f"\nTraining for {num_steps} steps with swap exposure...")
    print("Swap condition: horizontal translation > 1 pixel")
    
    swap_count = 0
    
    for step in range(1, num_steps + 1):
        # Sample x_t (object A)
        idx_a = indices_a[torch.randint(0, len(indices_a), (1,)).item()]
        image_a, _ = trainset[idx_a]
        
        # Apply augmentation to object A
        x_t_aug, translate_x = augment(image_a)
        x_t = x_t_aug.flatten()
        
        # Swap logic: if horizontal shift > 1 pixel, replace x_t with object B
        if abs(translate_x) > 1:
            # Swap: use object B instead
            idx_b = indices_b[torch.randint(0, len(indices_b), (1,)).item()]
            image_b, _ = trainset[idx_b]
            x_t_swapped, _ = augment(image_b)
            x_t = x_t_swapped.flatten()
            swap_count += 1
        
        # Generate x_t1 as augmented view (of swapped or original object)
        # Use the same object type that x_t came from
        if abs(translate_x) > 1:
            # x_t is from object B, so x_t1 should also be from object B
            idx_t1 = indices_b[torch.randint(0, len(indices_b), (1,)).item()]
            image_t1, _ = trainset[idx_t1]
        else:
            # x_t is from object A, so x_t1 should also be from object A
            idx_t1 = indices_a[torch.randint(0, len(indices_a), (1,)).item()]
            image_t1, _ = trainset[idx_t1]
        
        x_t1, _ = augment(image_t1)
        x_t1 = x_t1.flatten()
        
        # Update layer weights using local learning rules
        layer.update(x_t, x_t1)
        
        # Print progress every 1000 steps
        if step % 1000 == 0:
            swap_rate = swap_count / step * 100
            print(f"Step {step}/{num_steps} | Swaps: {swap_count} ({swap_rate:.1f}%)")
    
    print("Training completed.")
    
    # Export activations after training
    print("\nExporting activations after training...")
    activations_after = export_activations(layer, trainset, num_samples=1000)
    
    # Save results to single .pt file
    output_dir = Path('outputs/activations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'activations_before': activations_before['activations'],
        'activations_after': activations_after['activations'],
        'labels_before': activations_before['labels'],
        'labels_after': activations_after['labels']
    }
    
    torch.save(results, output_dir / 'swap_experiment.pt')
    print(f"Saved swap experiment results to {output_dir / 'swap_experiment.pt'}")


if __name__ == "__main__":
    main()
