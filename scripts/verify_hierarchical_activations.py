"""Verify hierarchical activations have proper variance and structure."""

import torch
from pathlib import Path


def verify_activations(filepath, stage_name):
    """Verify activations are non-collapsed and have variance."""
    print(f"\n{stage_name}")
    print("=" * 60)
    
    data = torch.load(filepath, map_location='cpu')
    
    # Check Layer 1 activations
    layer1 = data['layer1_activations']
    print(f"Layer 1 Activations:")
    print(f"  Shape: {layer1.shape}")
    print(f"  Mean: {layer1.mean().item():.6f}")
    print(f"  Std: {layer1.std().item():.6f}")
    print(f"  Min: {layer1.min().item():.6f}")
    print(f"  Max: {layer1.max().item():.6f}")
    print(f"  Has NaN: {torch.isnan(layer1).any().item()}")
    print(f"  Non-collapsed (std > 0.1): {layer1.std().item() > 0.1}")
    
    # Check Layer 2 activations
    layer2 = data['layer2_activations']
    print(f"\nLayer 2 Activations:")
    print(f"  Shape: {layer2.shape}")
    print(f"  Mean: {layer2.mean().item():.6f}")
    print(f"  Std: {layer2.std().item():.6f}")
    print(f"  Min: {layer2.min().item():.6f}")
    print(f"  Max: {layer2.max().item():.6f}")
    print(f"  Has NaN: {torch.isnan(layer2).any().item()}")
    print(f"  Non-collapsed (std > 0.1): {layer2.std().item() > 0.1}")
    
    # Check labels
    labels = data['labels']
    print(f"\nLabels:")
    print(f"  Shape: {labels.shape}")
    print(f"  Unique values: {torch.unique(labels).numel()}")
    
    return {
        'layer1_std': layer1.std().item(),
        'layer2_std': layer2.std().item(),
        'layer1_has_nan': torch.isnan(layer1).any().item(),
        'layer2_has_nan': torch.isnan(layer2).any().item(),
    }


def main():
    """Main verification function."""
    output_dir = Path('outputs/activations')
    
    print("VERIFYING HIERARCHICAL LPL ACTIVATIONS")
    print("=" * 60)
    
    # Verify before training
    before_file = output_dir / 'hierarchical_activations_before.pt'
    if before_file.exists():
        before_stats = verify_activations(before_file, "BEFORE TRAINING")
    else:
        print(f"\nWarning: {before_file} not found!")
        return
    
    # Verify after training
    after_file = output_dir / 'hierarchical_activations_after.pt'
    if after_file.exists():
        after_stats = verify_activations(after_file, "AFTER TRAINING")
    else:
        print(f"\nWarning: {after_file} not found!")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    layer1_ok = after_stats['layer1_std'] > 0.1 and not after_stats['layer1_has_nan']
    layer2_ok = after_stats['layer2_std'] > 0.1 and not after_stats['layer2_has_nan']
    
    print(f"Layer 1 (non-collapsed): {'PASS' if layer1_ok else 'FAIL'}")
    print(f"  Std: {after_stats['layer1_std']:.6f} ({'good' if after_stats['layer1_std'] > 0.1 else 'collapsed'})")
    print(f"  No NaN: {'Yes' if not after_stats['layer1_has_nan'] else 'No'}")
    
    print(f"\nLayer 2 (non-collapsed): {'PASS' if layer2_ok else 'FAIL'}")
    print(f"  Std: {after_stats['layer2_std']:.6f} ({'good' if after_stats['layer2_std'] > 0.1 else 'collapsed'})")
    print(f"  No NaN: {'Yes' if not after_stats['layer2_has_nan'] else 'No'}")
    
    if layer1_ok and layer2_ok:
        print("\nAll verifications passed!")
    else:
        print("\nSome verifications failed!")


if __name__ == "__main__":
    main()

