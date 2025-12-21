"""Swap-style analysis for hierarchical LPL activations."""

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
    
    # Limit to avoid excessive computation
    max_pairs = 10000
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
    
    Args:
        same_sample_sim: Tensor of same-sample similarities
        different_label_sim: Tensor of different-label similarities
        
    Returns:
        Identity preservation score (scalar)
    """
    mean_same_sample = same_sample_sim.mean().item()
    mean_different_label = different_label_sim.mean().item()
    return mean_same_sample - mean_different_label


def analyze_layer(activations_before, activations_after, labels, layer_name):
    """
    Analyze a single layer's activations.
    
    Args:
        activations_before: Tensor of shape (N, D)
        activations_after: Tensor of shape (N, D)
        labels: Tensor of shape (N,)
        layer_name: Name of the layer (e.g., 'Layer1')
        
    Returns:
        Dictionary with analysis results
    """
    # Check for NaN
    if torch.isnan(activations_before).any() or torch.isnan(activations_after).any():
        print(f"WARNING: {layer_name} contains NaN values!")
        return None
    
    # Check for collapsed representations
    std_before = activations_before.std().item()
    std_after = activations_after.std().item()
    
    if std_before < 0.1:
        print(f"WARNING: {layer_name} before training appears collapsed (std={std_before:.6f})")
    if std_after < 0.1:
        print(f"WARNING: {layer_name} after training appears collapsed (std={std_after:.6f})")
    
    # Compute same-sample similarity
    same_sample_sim = compute_same_sample_similarity(activations_before, activations_after)
    
    # Compute same-label similarity (after training)
    same_label_after = compute_same_label_similarity(activations_after, labels)
    
    # Compute different-label similarity (after training)
    different_label_after = compute_different_label_similarity(activations_after, labels)
    
    # Compute identity preservation score
    identity_score = compute_identity_preservation_score(same_sample_sim, different_label_after)
    
    return {
        'layer': layer_name,
        'same_sample_mean': same_sample_sim.mean().item(),
        'same_sample_std': same_sample_sim.std().item(),
        'same_label_mean': same_label_after.mean().item(),
        'same_label_std': same_label_after.std().item(),
        'different_label_mean': different_label_after.mean().item(),
        'different_label_std': different_label_after.std().item(),
        'identity_score': identity_score,
        'std_before': std_before,
        'std_after': std_after,
    }


def main():
    """Main function to analyze hierarchical LPL activations."""
    
    activations_dir = Path('outputs/activations')
    
    # Load files
    before_file = activations_dir / 'hierarchical_activations_before.pt'
    after_file = activations_dir / 'hierarchical_activations_after.pt'
    
    if not before_file.exists():
        print(f"Error: {before_file} does not exist!")
        return
    
    if not after_file.exists():
        print(f"Error: {after_file} does not exist!")
        return
    
    print("Loading hierarchical activation files...")
    data_before = torch.load(before_file, map_location='cpu')
    data_after = torch.load(after_file, map_location='cpu')
    
    print("Analyzing activations...\n")
    
    # Analyze both layers
    results = []
    
    # Analyze Layer1
    result1 = analyze_layer(
        data_before['layer1_activations'],
        data_after['layer1_activations'],
        data_after['labels'],
        'Layer1'
    )
    if result1:
        results.append(result1)
    
    # Analyze Layer2
    result2 = analyze_layer(
        data_before['layer2_activations'],
        data_after['layer2_activations'],
        data_after['labels'],
        'Layer2'
    )
    if result2:
        results.append(result2)
    
    # Print results table
    print("\n" + "="*100)
    print("HIERARCHICAL LPL SWAP-STYLE ANALYSIS")
    print("="*100)
    print(f"{'Layer':<10} {'Same-Sample Sim':<20} {'Same-Label Sim':<20} {'Diff-Label Sim':<20} {'Identity Score':<15} {'Std (before)':<15} {'Std (after)':<15}")
    print("-"*100)
    
    for r in results:
        print(f"{r['layer']:<10} "
              f"{r['same_sample_mean']:>7.4f} +/- {r['same_sample_std']:<7.4f}  "
              f"{r['same_label_mean']:>7.4f} +/- {r['same_label_std']:<7.4f}  "
              f"{r['different_label_mean']:>7.4f} +/- {r['different_label_std']:<7.4f}  "
              f"{r['identity_score']:>15.4f}  "
              f"{r['std_before']:>15.4f}  "
              f"{r['std_after']:>15.4f}")
    
    print("="*100)
    
    # Layer comparison
    if len(results) == 2:
        layer1 = results[0]
        layer2 = results[1]
        
        print("\n" + "="*100)
        print("LAYER1 vs LAYER2 COMPARISON")
        print("="*100)
        
        print("\n[Class Separation]")
        print(f"  Different-Label Similarity (lower = better separation):")
        print(f"    Layer1: {layer1['different_label_mean']:.4f}")
        print(f"    Layer2: {layer2['different_label_mean']:.4f}")
        better_separation = "Layer2" if layer2['different_label_mean'] < layer1['different_label_mean'] else "Layer1"
        print(f"  -> {better_separation} has better class separation")
        
        print(f"\n  Same-Label Similarity (higher = better coherence):")
        print(f"    Layer1: {layer1['same_label_mean']:.4f}")
        print(f"    Layer2: {layer2['same_label_mean']:.4f}")
        better_coherence = "Layer2" if layer2['same_label_mean'] > layer1['same_label_mean'] else "Layer1"
        print(f"  -> {better_coherence} has better class coherence")
        
        print("\n[Identity Preservation]")
        print(f"  Same-Sample Similarity (higher = better identity preservation):")
        print(f"    Layer1: {layer1['same_sample_mean']:.4f} +/- {layer1['same_sample_std']:.4f}")
        print(f"    Layer2: {layer2['same_sample_mean']:.4f} +/- {layer2['same_sample_std']:.4f}")
        
        print(f"\n  Identity Preservation Score (higher = better):")
        print(f"    Layer1: {layer1['identity_score']:.4f}")
        print(f"    Layer2: {layer2['identity_score']:.4f}")
        better_identity = "Layer2" if layer2['identity_score'] > layer1['identity_score'] else "Layer1"
        print(f"  -> {better_identity} has better identity preservation")
        
        print("\n[Representation Statistics]")
        print(f"  Standard Deviation (should be > 0.1 to avoid collapse):")
        print(f"    Layer1: {layer1['std_before']:.4f} -> {layer1['std_after']:.4f}")
        print(f"    Layer2: {layer2['std_before']:.4f} -> {layer2['std_after']:.4f}")
        layer1_collapsed = layer1['std_after'] < 0.1
        layer2_collapsed = layer2['std_after'] < 0.1
        if layer1_collapsed:
            print(f"    WARNING: Layer1 appears collapsed!")
        if layer2_collapsed:
            print(f"    WARNING: Layer2 appears collapsed!")
        if not layer1_collapsed and not layer2_collapsed:
            print(f"    Both layers show good variance (non-collapsed)")
    
    print("\n" + "="*100)
    print("INTERPRETATION NOTES")
    print("="*100)
    print("1. Same-Sample Similarity: Measures how much individual samples change their")
    print("   representations after training. High similarity indicates identity preservation.")
    print()
    print("2. Same-Label Similarity: Measures how similar representations are within the")
    print("   same class. High similarity indicates good class coherence.")
    print()
    print("3. Different-Label Similarity: Measures how similar representations are across")
    print("   different classes. Low similarity indicates good class separation.")
    print()
    print("4. Identity Preservation Score: Same-Sample - Different-Label. Higher scores")
    print("   indicate better identity preservation relative to class separation.")
    print()
    print("5. Standard Deviation: Should be > 0.1 to avoid representational collapse.")
    print("   Values < 0.1 indicate collapsed representations.")
    print("="*100)


if __name__ == "__main__":
    main()

