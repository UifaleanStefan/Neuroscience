"""Verify BYOL embeddings for NaN, collapse, and variance."""

import torch
from pathlib import Path
import json


def verify_embeddings(filepath, stage_name):
    """Verify embeddings are non-collapsed and have proper variance."""
    print(f"\n{stage_name}")
    print("=" * 70)
    
    data = torch.load(filepath, map_location='cpu')
    
    embeddings = data['embeddings']
    labels = data['labels']
    
    print(f"Embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean().item():.6f}")
    print(f"  Std: {embeddings.std().item():.6f}")
    print(f"  Min: {embeddings.min().item():.6f}")
    print(f"  Max: {embeddings.max().item():.6f}")
    print(f"  Has NaN: {torch.isnan(embeddings).any().item()}")
    print(f"  Non-collapsed (std > 0.1): {embeddings.std().item() > 0.1}")
    
    # Check L2 norms (should be ~1.0 since embeddings are L2 normalized)
    norms = torch.norm(embeddings, dim=1)
    print(f"\n  L2 Norms:")
    print(f"    Mean: {norms.mean().item():.6f}")
    print(f"    Std: {norms.std().item():.6f}")
    print(f"    Min: {norms.min().item():.6f}")
    print(f"    Max: {norms.max().item():.6f}")
    
    print(f"\nLabels:")
    print(f"  Shape: {labels.shape}")
    print(f"  Unique values: {torch.unique(labels).numel()}")
    
    return {
        'std': embeddings.std().item(),
        'has_nan': torch.isnan(embeddings).any().item(),
        'mean_norm': norms.mean().item(),
    }


def main():
    """Main verification function."""
    output_dir = Path('outputs/activations')
    
    print("VERIFYING BYOL EMBEDDINGS")
    print("=" * 70)
    
    # Check which BYOL files exist
    files_to_check = [
        ('byol_embeddings_before.pt', 'byol_embeddings_after.pt', 'CIFAR-10'),
        ('byol_shapes_embeddings_before.pt', 'byol_shapes_embeddings_after.pt', 'Synthetic Shapes'),
    ]
    
    for before_file, after_file, dataset_name in files_to_check:
        before_path = output_dir / before_file
        after_path = output_dir / after_file
        
        if not before_path.exists() and not after_path.exists():
            continue
        
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        if before_path.exists():
            before_stats = verify_embeddings(before_path, "BEFORE TRAINING")
        else:
            print(f"\nWarning: {before_file} not found!")
            continue
        
        if after_path.exists():
            after_stats = verify_embeddings(after_path, "AFTER TRAINING")
        else:
            print(f"\nWarning: {after_file} not found!")
            continue
        
        # Summary
        print(f"\n{'='*70}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*70}")
        
        before_ok = after_stats['std'] > 0.1 and not after_stats['has_nan']
        
        print(f"Before training:")
        print(f"  Std: {before_stats['std']:.6f} ({'good' if before_stats['std'] > 0.1 else 'collapsed'})")
        print(f"  Has NaN: {before_stats['has_nan']}")
        print(f"  Mean L2 norm: {before_stats['mean_norm']:.6f}")
        
        print(f"\nAfter training:")
        print(f"  Std: {after_stats['std']:.6f} ({'good' if after_stats['std'] > 0.1 else 'collapsed'})")
        print(f"  Has NaN: {after_stats['has_nan']}")
        print(f"  Mean L2 norm: {after_stats['mean_norm']:.6f}")
        
        if before_ok and not after_stats['has_nan']:
            print(f"\nStatus: PASS - Embeddings are valid")
        else:
            print(f"\nStatus: FAIL - Issues detected")
    
    # Check training logs if they exist
    print(f"\n{'='*70}")
    print("TRAINING LOGS")
    print(f"{'='*70}")
    
    log_files = ['byol_training_logs.json', 'byol_shapes_training_logs.json']
    for log_file in log_files:
        log_path = output_dir / log_file
        if log_path.exists():
            with open(log_path, 'r') as f:
                logs = json.load(f)
            
            losses = logs['loss']
            print(f"\n{log_file}:")
            print(f"  Total steps: {len(losses)}")
            print(f"  Final loss: {losses[-1]:.6f}")
            print(f"  Mean loss: {sum(losses) / len(losses):.6f}")
            print(f"  Min loss: {min(losses):.6f}")
            print(f"  Max loss: {max(losses):.6f}")
            print(f"  Has NaN in losses: {any(torch.isnan(torch.tensor(l)) for l in losses)}")


if __name__ == "__main__":
    main()

