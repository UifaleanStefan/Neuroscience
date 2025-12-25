"""Analyze identity preservation in swap experiment."""

import torch
import numpy as np
from pathlib import Path


def normalize_to_unit_length(activations):
    """
    Normalize activations to unit length (L2 norm = 1).
    
    Args:
        activations: Tensor of shape (N, D)
        
    Returns:
        Normalized tensor of shape (N, D)
    """
    norms = torch.norm(activations, dim=1, keepdim=True)
    # Avoid division by zero
    norms = torch.clamp(norms, min=1e-8)
    return activations / norms


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two tensors.
    
    Assumes inputs are already normalized to unit length.
    
    Args:
        a: Tensor of shape (..., D) - should be normalized
        b: Tensor of shape (..., D) - should be normalized
        
    Returns:
        Cosine similarity (dot product of normalized vectors)
    """
    return torch.sum(a * b, dim=-1)


def compute_same_sample_similarity(activations_before, activations_after):
    """
    Compute cosine similarity between same samples before and after training.
    
    This measures how much individual samples change their representations
    after swap training. High similarity indicates identity preservation.
    
    Args:
        activations_before: Tensor of shape (N, D)
        activations_after: Tensor of shape (N, D)
        
    Returns:
        Tensor of shape (N,) with similarities for each sample
    """
    # Normalize to unit length
    before_norm = normalize_to_unit_length(activations_before)
    after_norm = normalize_to_unit_length(activations_after)
    
    # Compute cosine similarity for each sample pair (same index)
    similarities = cosine_similarity(before_norm, after_norm)
    
    return similarities


def compute_same_label_similarity(activations, labels):
    """
    Compute average cosine similarity between different samples with the same label.
    
    This measures how similar representations are within the same class.
    High similarity indicates good class coherence.
    
    Args:
        activations: Tensor of shape (N, D)
        labels: Tensor of shape (N,)
        
    Returns:
        Average cosine similarity for same-label pairs
    """
    activations_norm = normalize_to_unit_length(activations)
    labels_np = labels.cpu().numpy()
    unique_labels = np.unique(labels_np)
    
    all_similarities = []
    
    for label in unique_labels:
        # Get indices of samples with this label
        label_mask = labels == label
        label_indices = torch.where(label_mask)[0]
        
        if len(label_indices) < 2:
            continue
        
        # Compute pairwise similarities within this class
        class_activations = activations_norm[label_indices]
        n_samples = len(class_activations)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sim = cosine_similarity(
                    class_activations[i:i+1], 
                    class_activations[j:j+1]
                ).item()
                all_similarities.append(sim)
    
    return torch.tensor(all_similarities) if all_similarities else torch.tensor([0.0])


def compute_different_label_similarity(activations, labels):
    """
    Compute average cosine similarity between samples with different labels.
    
    This measures how similar representations are across different classes.
    Low similarity indicates good class separation.
    
    Args:
        activations: Tensor of shape (N, D)
        labels: Tensor of shape (N,)
        
    Returns:
        Average cosine similarity for different-label pairs
    """
    activations_norm = normalize_to_unit_length(activations)
    labels_np = labels.cpu().numpy()
    unique_labels = np.unique(labels_np)
    
    all_similarities = []
    
    # Sample a subset of pairs to avoid O(N^2) computation for large datasets
    # For 1000 samples, we'll compute all pairs (reasonable)
    max_pairs = 50000  # Limit to avoid excessive computation
    
    pair_count = 0
    for i, label_i in enumerate(unique_labels):
        label_i_mask = labels == label_i
        label_i_indices = torch.where(label_i_mask)[0]
        
        for j, label_j in enumerate(unique_labels):
            if i >= j:  # Only compute once per pair of classes
                continue
            
            label_j_mask = labels == label_j
            label_j_indices = torch.where(label_j_mask)[0]
            
            # Sample pairs between these two classes
            for idx_i in label_i_indices:
                for idx_j in label_j_indices:
                    if pair_count >= max_pairs:
                        break
                    sim = cosine_similarity(
                        activations_norm[idx_i:idx_i+1],
                        activations_norm[idx_j:idx_j+1]
                    ).item()
                    all_similarities.append(sim)
                    pair_count += 1
                if pair_count >= max_pairs:
                    break
            if pair_count >= max_pairs:
                break
        if pair_count >= max_pairs:
            break
    
    return torch.tensor(all_similarities) if all_similarities else torch.tensor([0.0])


def compute_identity_preservation_score(same_sample_sim, different_label_sim):
    """
    Compute identity preservation score.
    
    Identity Preservation = Same-Sample Similarity - Different-Label Similarity
    
    This measures how well individual sample identities are preserved relative
    to how different they are from samples of other classes. Higher scores
    indicate better identity preservation.
    
    Args:
        same_sample_sim: Tensor of same-sample similarities
        different_label_sim: Tensor of different-label similarities
        
    Returns:
        Identity preservation score (scalar)
    """
    mean_same_sample = same_sample_sim.mean().item()
    mean_different_label = different_label_sim.mean().item()
    return mean_same_sample - mean_different_label


def main():
    """Main function to analyze swap experiment identity preservation."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load swap experiment data
    swap_file = Path('outputs/activations/swap_experiment.pt')
    
    if not swap_file.exists():
        print(f"Error: {swap_file} does not exist!")
        return
    
    print("Loading swap experiment data...")
    data = torch.load(swap_file, map_location='cpu')
    
    activations_before = data['activations_before']
    activations_after = data['activations_after']
    labels_before = data['labels_before']
    labels_after = data['labels_after']
    
    print(f"Loaded activations: before shape {activations_before.shape}, "
          f"after shape {activations_after.shape}\n")
    
    # Compute same-sample similarity (before vs after, same index)
    print("Computing same-sample similarities...")
    same_sample_sim = compute_same_sample_similarity(
        activations_before, activations_after
    )
    
    # Compute same-label similarity (before training)
    print("Computing same-label similarities (before training)...")
    same_label_before = compute_same_label_similarity(
        activations_before, labels_before
    )
    
    # Compute same-label similarity (after training)
    print("Computing same-label similarities (after training)...")
    same_label_after = compute_same_label_similarity(
        activations_after, labels_after
    )
    
    # Compute different-label similarity (before training)
    print("Computing different-label similarities (before training)...")
    different_label_before = compute_different_label_similarity(
        activations_before, labels_before
    )
    
    # Compute different-label similarity (after training)
    print("Computing different-label similarities (after training)...")
    different_label_after = compute_different_label_similarity(
        activations_after, labels_after
    )
    
    # Compute identity preservation score
    identity_preservation_before = compute_identity_preservation_score(
        same_sample_sim, different_label_before
    )
    identity_preservation_after = compute_identity_preservation_score(
        same_sample_sim, different_label_after
    )
    
    # Print results
    print("\n" + "="*80)
    print("SWAP EXPERIMENT: IDENTITY PRESERVATION ANALYSIS")
    print("="*80)
    
    print("\n[Same-Sample Similarity]")
    print("  Cosine similarity between activations before and after training")
    print("  (same sample index, measures individual identity preservation)")
    print(f"  Mean: {same_sample_sim.mean().item():.4f} ± {same_sample_sim.std().item():.4f}")
    
    print("\n[Same-Label Similarity]")
    print("  Cosine similarity between different samples with the same label")
    print("  (measures class coherence)")
    print(f"  Before training: {same_label_before.mean().item():.4f} ± {same_label_before.std().item():.4f}")
    print(f"  After training:  {same_label_after.mean().item():.4f} ± {same_label_after.std().item():.4f}")
    
    print("\n[Different-Label Similarity]")
    print("  Cosine similarity between samples with different labels")
    print("  (measures class separation, lower is better)")
    print(f"  Before training: {different_label_before.mean().item():.4f} ± {different_label_before.std().item():.4f}")
    print(f"  After training:  {different_label_after.mean().item():.4f} ± {different_label_after.std().item():.4f}")
    
    print("\n[Identity Preservation Score]")
    print("  Score = Same-Sample Similarity - Different-Label Similarity")
    print("  (higher = better identity preservation relative to class separation)")
    print(f"  Before training: {identity_preservation_before:.4f}")
    print(f"  After training:  {identity_preservation_after:.4f}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Metric':<35} {'Before':<20} {'After':<20}")
    print("-"*80)
    print(f"{'Same-Sample Similarity':<35} "
          f"{same_sample_sim.mean().item():>8.4f} ± {same_sample_sim.std().item():>6.4f}  "
          f"{same_sample_sim.mean().item():>8.4f} ± {same_sample_sim.std().item():>6.4f}")
    print(f"{'Same-Label Similarity':<35} "
          f"{same_label_before.mean().item():>8.4f} ± {same_label_before.std().item():>6.4f}  "
          f"{same_label_after.mean().item():>8.4f} ± {same_label_after.std().item():>6.4f}")
    print(f"{'Different-Label Similarity':<35} "
          f"{different_label_before.mean().item():>8.4f} ± {different_label_before.std().item():>6.4f}  "
          f"{different_label_after.mean().item():>8.4f} ± {different_label_after.std().item():>6.4f}")
    print(f"{'Identity Preservation Score':<35} "
          f"{identity_preservation_before:>20.4f}  {identity_preservation_after:>20.4f}")
    print("="*80)
    
    print("\nInterpretation:")
    print("- High same-sample similarity: Individual samples retain their identity")
    print("- High same-label similarity: Good class coherence")
    print("- Low different-label similarity: Good class separation")
    print("- High identity preservation score: Better identity preservation")
    print("  (relative to how different samples are from other classes)")


if __name__ == "__main__":
    main()





