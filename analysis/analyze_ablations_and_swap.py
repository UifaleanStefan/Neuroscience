"""
Analyze ablation and swap experiment results.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def linear_probe(activations, labels, n_splits=3, random_seed=42):
    """Compute linear probe accuracy with multiple splits."""
    X = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    accuracies = []
    for split in range(n_splits):
        split_seed = random_seed + split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=split_seed, stratify=y
        )
        
        clf = LogisticRegression(max_iter=1000, random_state=split_seed, C=1.0, solver='lbfgs', n_jobs=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies) if len(accuracies) > 1 else 0.0,
        'all': accuracies
    }

def compute_activation_stats(activations):
    """Compute activation statistics."""
    if isinstance(activations, torch.Tensor):
        activations = activations.numpy()
    
    return {
        'mean': float(np.mean(activations)),
        'std': float(np.std(activations)),
        'min': float(np.min(activations)),
        'max': float(np.max(activations)),
        'median': float(np.median(activations))
    }

def compute_separation_metrics(activations, labels):
    """Compute intra-class variance and inter-class distance."""
    if isinstance(activations, torch.Tensor):
        activations = activations.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    unique_labels = np.unique(labels)
    
    # Intra-class variance
    intra_variances = []
    for label in unique_labels:
        class_mask = labels == label
        class_activations = activations[class_mask]
        if len(class_activations) > 1:
            per_dim_variance = np.var(class_activations, axis=0)
            intra_var = np.mean(per_dim_variance)
            intra_variances.append(intra_var)
    
    mean_intra_variance = np.mean(intra_variances) if intra_variances else 0.0
    
    # Inter-class distance
    centroids = []
    for label in unique_labels:
        class_mask = labels == label
        class_activations = activations[class_mask]
        centroid = class_activations.mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    inter_distances = []
    n_classes = len(centroids)
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_distances.append(dist)
    
    mean_inter_distance = np.mean(inter_distances) if inter_distances else 0.0
    separation_ratio = mean_intra_variance / mean_inter_distance if mean_inter_distance > 0 else np.inf
    
    return {
        'intra_variance': float(mean_intra_variance),
        'inter_distance': float(mean_inter_distance),
        'separation_ratio': float(separation_ratio)
    }

def analyze_swap_experiment(swap_data):
    """Analyze swap experiment identity preservation."""
    activations_before = swap_data['activations_before'].numpy()
    activations_after = swap_data['activations_after'].numpy()
    labels_before = swap_data['labels_before'].numpy()
    labels_after = swap_data['labels_after'].numpy()
    
    # Compute cosine similarity between before and after for same samples
    n_samples = len(activations_before)
    similarities = []
    
    for i in range(n_samples):
        sim = 1 - cosine(activations_before[i], activations_after[i])
        similarities.append(sim)
    
    mean_similarity = np.mean(similarities)
    
    # Check if samples maintain their relative positions (identity preservation)
    # For same-class samples, similarity should be higher
    same_class_sims = []
    diff_class_sims = []
    
    for i in range(min(100, n_samples)):  # Sample subset for efficiency
        for j in range(i + 1, min(100, n_samples)):
            before_sim = 1 - cosine(activations_before[i], activations_before[j])
            after_sim = 1 - cosine(activations_after[i], activations_after[j])
            
            if labels_before[i] == labels_before[j]:
                same_class_sims.append(after_sim)
            else:
                diff_class_sims.append(after_sim)
    
    # Compute correlation between before and after similarities
    # This measures identity preservation: if high correlation, samples maintain relative positions
    sample_pairs_sims_before = []
    sample_pairs_sims_after = []
    
    for i in range(min(200, n_samples)):
        for j in range(i + 1, min(200, n_samples)):
            sim_before = 1 - cosine(activations_before[i], activations_before[j])
            sim_after = 1 - cosine(activations_after[i], activations_after[j])
            sample_pairs_sims_before.append(sim_before)
            sample_pairs_sims_after.append(sim_after)
    
    if len(sample_pairs_sims_before) > 10:
        correlation, p_value = pearsonr(sample_pairs_sims_before, sample_pairs_sims_after)
    else:
        correlation, p_value = 0.0, 1.0
    
    return {
        'mean_self_similarity': float(mean_similarity),
        'same_class_mean_sim': float(np.mean(same_class_sims)) if same_class_sims else None,
        'diff_class_mean_sim': float(np.mean(diff_class_sims)) if diff_class_sims else None,
        'identity_preservation_correlation': float(correlation),
        'correlation_p_value': float(p_value)
    }

def main():
    """Analyze all ablation and swap experiments."""
    
    output_dir = Path('outputs/activations')
    results = {}
    
    # Analyze ablations
    print("Analyzing ablation experiments...")
    ablation_names = ['hebb', 'pred', 'stab', 'shuffle']
    
    for name in ablation_names:
        filepath = output_dir / f'activations_ablation_{name}.pt'
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue
        
        data = torch.load(filepath, map_location='cpu')
        activations = data['activations']
        labels = data['labels']
        
        print(f"\nAnalyzing {name}...")
        
        # Compute metrics
        stats = compute_activation_stats(activations)
        linear_acc = linear_probe(activations, labels)
        separation = compute_separation_metrics(activations, labels)
        
        results[f'ablation_{name}'] = {
            'activation_stats': stats,
            'linear_probe_accuracy': linear_acc,
            'separation_metrics': separation
        }
        
        print(f"  Linear probe: {linear_acc['mean']:.4f} ± {linear_acc['std']:.4f}")
        print(f"  Activation mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
        print(f"  Separation ratio: {separation['separation_ratio']:.6f}")
    
    # Analyze swap experiment
    print("\nAnalyzing swap experiment...")
    swap_filepath = output_dir / 'swap_experiment.pt'
    
    if swap_filepath.exists():
        swap_data = torch.load(swap_filepath, map_location='cpu')
        
        # Analyze before and after separately
        before_activations = swap_data['activations_before']
        after_activations = swap_data['activations_after']
        before_labels = swap_data['labels_before']
        after_labels = swap_data['labels_after']
        
        before_stats = compute_activation_stats(before_activations)
        after_stats = compute_activation_stats(after_activations)
        before_acc = linear_probe(before_activations, before_labels)
        after_acc = linear_probe(after_activations, after_labels)
        before_sep = compute_separation_metrics(before_activations, before_labels)
        after_sep = compute_separation_metrics(after_activations, after_labels)
        
        identity_preservation = analyze_swap_experiment(swap_data)
        
        results['swap'] = {
            'before_training': {
                'activation_stats': before_stats,
                'linear_probe_accuracy': before_acc,
                'separation_metrics': before_sep
            },
            'after_swap_exposure': {
                'activation_stats': after_stats,
                'linear_probe_accuracy': after_acc,
                'separation_metrics': after_sep
            },
            'identity_preservation': identity_preservation
        }
        
        print(f"  Before - Linear probe: {before_acc['mean']:.4f} ± {before_acc['std']:.4f}")
        print(f"  After - Linear probe: {after_acc['mean']:.4f} ± {after_acc['std']:.4f}")
        print(f"  Identity preservation correlation: {identity_preservation['identity_preservation_correlation']:.4f}")
    else:
        print(f"Warning: {swap_filepath} not found")
    
    # Save results
    import json
    output_file = Path('analysis/ablations_and_swap_results.json')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy types to native Python types for JSON
    def convert_to_json(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_to_json(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
