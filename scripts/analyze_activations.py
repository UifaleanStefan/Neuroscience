"""Script to analyze LPL activations from .pt files."""

import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_activation_stats(activations):
    """
    Compute statistics for activation tensor.
    
    Args:
        activations: Tensor of shape (num_samples, dim)
        
    Returns:
        Dictionary with mean, std, min, max, and L2 norm
    """
    stats = {
        'mean': activations.mean().item(),
        'std': activations.std().item(),
        'min': activations.min().item(),
        'max': activations.max().item(),
    }
    
    # Compute L2 norm: average norm across all samples
    l2_norms = torch.norm(activations, dim=1)
    stats['avg_l2_norm'] = l2_norms.mean().item()
    stats['std_l2_norm'] = l2_norms.std().item()
    
    return stats


def compute_cosine_similarity(activations_before, activations_after):
    """
    Compute average cosine similarity per sample between before and after activations.
    
    Args:
        activations_before: Tensor of shape (num_samples, dim)
        activations_after: Tensor of shape (num_samples, dim)
        
    Returns:
        Average cosine similarity per sample
    """
    # Convert to numpy for sklearn
    before_np = activations_before.detach().cpu().numpy()
    after_np = activations_after.detach().cpu().numpy()
    
    # Compute cosine similarity for each pair of samples
    similarities = []
    for i in range(len(activations_before)):
        # Reshape for sklearn (needs 2D arrays)
        before_sample = before_np[i:i+1, :]
        after_sample = after_np[i:i+1, :]
        sim = cosine_similarity(before_sample, after_sample)[0, 0]
        similarities.append(sim)
    
    return np.mean(similarities), np.std(similarities)


def analyze_pt_file(file_path):
    """
    Analyze a single .pt file containing activation data.
    
    Args:
        file_path: Path to the .pt file
    """
    print(f"\n{'='*70}")
    print(f"File: {file_path.name}")
    print(f"{'='*70}")
    
    # Load the file
    data = torch.load(file_path, map_location='cpu')
    
    if isinstance(data, dict):
        # Check if it's a swap experiment format (with activations_before/after)
        if 'activations_before' in data and 'activations_after' in data:
            print("\n[Before Training]")
            stats_before = compute_activation_stats(data['activations_before'])
            print(f"  Mean:        {stats_before['mean']:12.6f}")
            print(f"  Std:         {stats_before['std']:12.6f}")
            print(f"  Min:         {stats_before['min']:12.6f}")
            print(f"  Max:         {stats_before['max']:12.6f}")
            print(f"  Avg L2 norm: {stats_before['avg_l2_norm']:12.6f} ± {stats_before['std_l2_norm']:8.6f}")
            
            print("\n[After Training]")
            stats_after = compute_activation_stats(data['activations_after'])
            print(f"  Mean:        {stats_after['mean']:12.6f}")
            print(f"  Std:         {stats_after['std']:12.6f}")
            print(f"  Min:         {stats_after['min']:12.6f}")
            print(f"  Max:         {stats_after['max']:12.6f}")
            print(f"  Avg L2 norm: {stats_after['avg_l2_norm']:12.6f} ± {stats_after['std_l2_norm']:8.6f}")
            
            # Compute cosine similarity
            print("\n[Comparison]")
            mean_sim, std_sim = compute_cosine_similarity(
                data['activations_before'], 
                data['activations_after']
            )
            print(f"  Avg cosine similarity: {mean_sim:8.6f} ± {std_sim:8.6f}")
            
        elif 'activations' in data:
            # Standard format with single activations tensor
            print("\n[Activations]")
            stats = compute_activation_stats(data['activations'])
            print(f"  Mean:        {stats['mean']:12.6f}")
            print(f"  Std:         {stats['std']:12.6f}")
            print(f"  Min:         {stats['min']:12.6f}")
            print(f"  Max:         {stats['max']:12.6f}")
            print(f"  Avg L2 norm: {stats['avg_l2_norm']:12.6f} ± {stats['std_l2_norm']:8.6f}")
        else:
            print("  Unknown dictionary format. Keys:", list(data.keys()))
    
    elif isinstance(data, torch.Tensor):
        # Single tensor format
        print("\n[Activations]")
        stats = compute_activation_stats(data)
        print(f"  Mean:        {stats['mean']:12.6f}")
        print(f"  Std:         {stats['std']:12.6f}")
        print(f"  Min:         {stats['min']:12.6f}")
        print(f"  Max:         {stats['max']:12.6f}")
        print(f"  Avg L2 norm: {stats['avg_l2_norm']:12.6f} ± {stats['std_l2_norm']:8.6f}")
    else:
        print(f"  Unsupported data type: {type(data)}")


def main():
    """Main function to analyze all .pt files in outputs/activations/."""
    
    activations_dir = Path('outputs/activations')
    
    if not activations_dir.exists():
        print(f"Directory {activations_dir} does not exist!")
        return
    
    # Find all .pt files
    pt_files = list(activations_dir.glob('*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in {activations_dir}")
        return
    
    print(f"Found {len(pt_files)} .pt file(s) in {activations_dir}\n")
    
    # Process each file
    for pt_file in sorted(pt_files):
        try:
            analyze_pt_file(pt_file)
        except Exception as e:
            print(f"\nError analyzing {pt_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()






