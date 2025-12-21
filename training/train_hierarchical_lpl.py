"""Training script for 2-layer hierarchical LPL on synthetic shapes."""

import torch
import sys
from pathlib import Path

# Add project root to path to import lpl_core
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from lpl_core.hierarchical_lpl import HierarchicalLPL
from data.synthetic_shapes import create_temporal_pair_dataset


class LayerConfig:
    """Configuration object for a single LPL layer."""
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
    Export activations from both layers.
    
    Args:
        model: HierarchicalLPL model
        dataset: SyntheticShapesDataset (should return single images, not pairs)
        num_samples: Number of samples to export
        
    Returns:
        Dictionary with 'layer1_activations', 'layer2_activations', and 'labels'
    """
    layer1_activations = []
    layer2_activations = []
    labels = []
    
    for i in range(min(num_samples, len(dataset))):
        # Get single image (not temporal pair)
        if hasattr(dataset, 'return_temporal_pair') and dataset.return_temporal_pair:
            # If dataset returns pairs, just use the first element
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
        
        # Forward pass to get activations from both layers
        y1, y2 = model.get_activations(x)
        
        # Check for NaN
        if torch.isnan(y1).any():
            print(f"WARNING: NaN detected in layer1 activation at sample {i}")
        if torch.isnan(y2).any():
            print(f"WARNING: NaN detected in layer2 activation at sample {i}")
        
        layer1_activations.append(y1)
        layer2_activations.append(y2)
        labels.append(label)
    
    return {
        'layer1_activations': torch.stack(layer1_activations),
        'layer2_activations': torch.stack(layer2_activations),
        'labels': torch.tensor(labels)
    }


def main(num_steps=10000):
    """
    Train 2-layer hierarchical LPL on synthetic shapes.
    
    Args:
        num_steps: Number of training steps (default: 5000)
    """
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration for layer 1 (input -> hidden)
    cfg_layer1 = LayerConfig(
        lr_hebb=0.001,
        lr_pred=0.001,
        lr_stab=0.0005,  # Reduced for stability
        use_hebb=True,
        use_pred=True,
        use_stab=True  # Can be disabled via config
    )
    
    # Configuration for layer 2 (hidden -> output)
    cfg_layer2 = LayerConfig(
        lr_hebb=0.001,
        lr_pred=0.001,
        lr_stab=0.0005,  # Reduced for stability
        use_hebb=True,
        use_pred=True,
        use_stab=True  # Can be disabled via config
    )
    
    # Dimensions: 32x32 images flattened = 1024
    d_in = 32 * 32
    d_hidden = 128  # First layer output dimension
    d_out = 64      # Second layer output dimension
    
    # Create hierarchical model
    model = HierarchicalLPL(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        cfg_layer1=cfg_layer1,
        cfg_layer2=cfg_layer2
    )
    
    # Load synthetic shapes dataset (for single images, not temporal pairs)
    # We'll create temporal pairs manually during training
    from data.synthetic_shapes import SyntheticShapesDataset
    dataset = SyntheticShapesDataset(num_samples=1000, seed=42, return_temporal_pair=False)
    
    # Also create a temporal pair dataset for training
    temporal_dataset = create_temporal_pair_dataset(num_samples=10000, seed=42)
    
    # Export activations before training
    print("Exporting activations before training...")
    activations_before = export_activations(model, dataset, num_samples=1000)
    output_dir = Path('outputs/activations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(activations_before, output_dir / 'hierarchical_activations_before.pt')
    print(f"Saved activations to {output_dir / 'hierarchical_activations_before.pt'}")
    
    # Training loop
    print(f"\nTraining hierarchical LPL for {num_steps} steps...")
    
    for step in range(1, num_steps + 1):
        # Sample a temporal pair from the dataset
        idx = torch.randint(0, len(temporal_dataset), (1,)).item()
        x_t, x_t1, _ = temporal_dataset[idx]
        
        # Flatten images to 1D tensors
        x_t_flat = x_t.flatten()
        x_t1_flat = x_t1.flatten()
        
        # Ensure inputs are floats in [0,1] range
        assert x_t_flat.dtype == torch.float32, f"x_t dtype is {x_t_flat.dtype}, expected float32"
        assert x_t1_flat.dtype == torch.float32, f"x_t1 dtype is {x_t1_flat.dtype}, expected float32"
        assert (x_t_flat >= 0.0).all() and (x_t_flat <= 1.0).all(), "x_t values must be in [0,1]"
        assert (x_t1_flat >= 0.0).all() and (x_t1_flat <= 1.0).all(), "x_t1 values must be in [0,1]"
        
        # Update both layers using local learning rules
        model.update(x_t_flat, x_t1_flat)
        
        # Check for NaN values (early detection of instability)
        if step % 500 == 0:
            if torch.isnan(model.layer1.W).any():
                print(f"WARNING: NaN detected in layer1 weights at step {step}!")
                break
            if torch.isnan(model.layer2.W).any():
                print(f"WARNING: NaN detected in layer2 weights at step {step}!")
                break
            if torch.isnan(model.layer1.predictor.P).any():
                print(f"WARNING: NaN detected in layer1 predictor at step {step}!")
                break
            if torch.isnan(model.layer2.predictor.P).any():
                print(f"WARNING: NaN detected in layer2 predictor at step {step}!")
                break
        
        # Print progress every 1000 steps
        if step % 1000 == 0:
            # Print diagnostic info
            w1_norm = torch.norm(model.layer1.W).item()
            w2_norm = torch.norm(model.layer2.W).item()
            y1_sample, y2_sample = model.get_activations(x_t_flat)
            y1_norm = torch.norm(y1_sample).item()
            y2_norm = torch.norm(y2_sample).item()
            print(f"Step {step}/{num_steps} | "
                  f"Layer1: ||W||={w1_norm:.4f} ||y||={y1_norm:.4f} | "
                  f"Layer2: ||W||={w2_norm:.4f} ||y||={y2_norm:.4f}")
    
    print("Training completed.")
    
    # Check for NaN before exporting
    if torch.isnan(model.layer1.W).any() or torch.isnan(model.layer2.W).any():
        print("ERROR: Weights contain NaN values! Cannot export activations.")
        return
    
    # Export activations after training
    print("\nExporting activations after training...")
    activations_after = export_activations(model, dataset, num_samples=1000)
    
    # Check exported activations for NaN
    if torch.isnan(activations_after['layer1_activations']).any():
        print("ERROR: Layer1 activations contain NaN values!")
        return
    if torch.isnan(activations_after['layer2_activations']).any():
        print("ERROR: Layer2 activations contain NaN values!")
        return
    
    torch.save(activations_after, output_dir / 'hierarchical_activations_after.pt')
    print(f"Saved activations to {output_dir / 'hierarchical_activations_after.pt'}")
    
    # Print statistics
    print(f"\nLayer1 activation stats: "
          f"mean={activations_after['layer1_activations'].mean().item():.6f}, "
          f"std={activations_after['layer1_activations'].std().item():.6f}")
    print(f"Layer2 activation stats: "
          f"mean={activations_after['layer2_activations'].mean().item():.6f}, "
          f"std={activations_after['layer2_activations'].std().item():.6f}")


if __name__ == "__main__":
    main()

