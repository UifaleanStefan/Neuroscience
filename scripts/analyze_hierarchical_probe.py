"""Linear probe analysis for hierarchical LPL activations."""

import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def compute_distances(activations, labels):
    """
    Compute average intra-class and inter-class L2 distances.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        
    Returns:
        Tuple of (intra_class_dist, inter_class_dist, ratio)
    """
    activations_np = activations.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    unique_labels = np.unique(labels_np)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        return 0.0, 0.0, 0.0
    
    # Compute intra-class distances (within same class)
    intra_distances = []
    for label in unique_labels:
        class_mask = labels_np == label
        class_activations = activations_np[class_mask]
        
        if len(class_activations) < 2:
            continue
        
        # Compute pairwise distances within this class
        for i in range(len(class_activations)):
            for j in range(i + 1, len(class_activations)):
                dist = np.linalg.norm(class_activations[i] - class_activations[j])
                intra_distances.append(dist)
    
    # Compute inter-class distances (between different classes)
    inter_distances = []
    max_pairs = 10000  # Limit to avoid excessive computation
    
    pair_count = 0
    for i, label_i in enumerate(unique_labels):
        class_i_mask = labels_np == label_i
        class_i_activations = activations_np[class_i_mask]
        
        for j, label_j in enumerate(unique_labels):
            if i >= j:  # Only compute once per pair of classes
                continue
            
            class_j_mask = labels_np == label_j
            class_j_activations = activations_np[class_j_mask]
            
            # Sample pairs between these two classes
            for act_i in class_i_activations:
                for act_j in class_j_activations:
                    if pair_count >= max_pairs:
                        break
                    dist = np.linalg.norm(act_i - act_j)
                    inter_distances.append(dist)
                    pair_count += 1
                if pair_count >= max_pairs:
                    break
            if pair_count >= max_pairs:
                break
        if pair_count >= max_pairs:
            break
    
    avg_intra = np.mean(intra_distances) if intra_distances else 0.0
    avg_inter = np.mean(inter_distances) if inter_distances else 0.0
    ratio = avg_inter / avg_intra if avg_intra > 0 else 0.0
    
    return avg_intra, avg_inter, ratio


def linear_probe(activations, labels, random_seed=42):
    """
    Train a linear classifier on frozen activations.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        random_seed: Random seed for reproducibility
        
    Returns:
        Test accuracy as float
    """
    # Convert to numpy
    X = activations.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    # Train linear classifier (logistic regression)
    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        C=1.0,
        solver='lbfgs'
    )
    
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Compute test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def analyze_layer(activations, labels, layer_name, stage, random_seed=42):
    """
    Analyze a single layer's activations.
    
    Args:
        activations: Tensor of shape (N, D)
        labels: Tensor of shape (N,)
        layer_name: Name of the layer (e.g., 'Layer1', 'Layer2')
        stage: Stage name (e.g., 'Before', 'After')
        random_seed: Random seed
        
    Returns:
        Dictionary with analysis results
    """
    # Check for NaN
    has_nan = torch.isnan(activations).any().item()
    if has_nan:
        print(f"WARNING: {layer_name} {stage} contains NaN values!")
        return {
            'layer': layer_name,
            'stage': stage,
            'accuracy': np.nan,
            'intra_dist': np.nan,
            'inter_dist': np.nan,
            'ratio': np.nan,
            'std': np.nan,
            'collapsed': True,
            'has_nan': True
        }
    
    # Check for collapsed representations
    std = activations.std().item()
    collapsed = std < 0.1
    
    if collapsed:
        print(f"WARNING: {layer_name} {stage} appears collapsed (std={std:.6f} < 0.1)")
    
    # Compute linear probe accuracy
    accuracy = linear_probe(activations, labels, random_seed=random_seed)
    
    # Compute distances
    intra_dist, inter_dist, ratio = compute_distances(activations, labels)
    
    return {
        'layer': layer_name,
        'stage': stage,
        'accuracy': accuracy,
        'intra_dist': intra_dist,
        'inter_dist': inter_dist,
        'ratio': ratio,
        'std': std,
        'collapsed': collapsed,
        'has_nan': False
    }


def main():
    """Main function to analyze hierarchical LPL activations."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
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
    
    # Analyze all layers and stages
    results = []
    
    # Layer 1 Before
    results.append(analyze_layer(
        data_before['layer1_activations'],
        data_before['labels'],
        'Layer1',
        'Before'
    ))
    
    # Layer 1 After
    results.append(analyze_layer(
        data_after['layer1_activations'],
        data_after['labels'],
        'Layer1',
        'After'
    ))
    
    # Layer 2 Before
    results.append(analyze_layer(
        data_before['layer2_activations'],
        data_before['labels'],
        'Layer2',
        'Before'
    ))
    
    # Layer 2 After
    results.append(analyze_layer(
        data_after['layer2_activations'],
        data_after['labels'],
        'Layer2',
        'After'
    ))
    
    # Print results table
    print("\n" + "="*100)
    print("HIERARCHICAL LPL LINEAR PROBE ANALYSIS")
    print("="*100)
    print(f"{'Layer':<10} {'Stage':<10} {'Accuracy':<12} {'Intra-Dist':<12} {'Inter-Dist':<12} {'Ratio':<10} {'Std':<10} {'Status':<15}")
    print("-"*100)
    
    for r in results:
        status = []
        if r['collapsed']:
            status.append('COLLAPSED')
        if r['has_nan']:
            status.append('NaN')
        if not status:
            status.append('OK')
        
        status_str = ', '.join(status)
        
        if r['has_nan']:
            print(f"{r['layer']:<10} {r['stage']:<10} {'NaN':<12} {'NaN':<12} {'NaN':<12} {'NaN':<10} {'NaN':<10} {status_str:<15}")
        else:
            print(f"{r['layer']:<10} {r['stage']:<10} "
                  f"{r['accuracy']:<12.4f} "
                  f"{r['intra_dist']:<12.4f} "
                  f"{r['inter_dist']:<12.4f} "
                  f"{r['ratio']:<10.4f} "
                  f"{r['std']:<10.4f} "
                  f"{status_str:<15}")
    
    print("="*100)
    
    # Comparison summary
    print("\n" + "="*100)
    print("BEFORE vs AFTER COMPARISON")
    print("="*100)
    
    for layer_name in ['Layer1', 'Layer2']:
        before_r = next(r for r in results if r['layer'] == layer_name and r['stage'] == 'Before')
        after_r = next(r for r in results if r['layer'] == layer_name and r['stage'] == 'After')
        
        print(f"\n{layer_name}:")
        if before_r['has_nan'] or after_r['has_nan']:
            print("  Cannot compare: NaN values present")
            continue
        
        acc_change = after_r['accuracy'] - before_r['accuracy']
        ratio_change = after_r['ratio'] - before_r['ratio']
        std_change = after_r['std'] - before_r['std']
        
        print(f"  Accuracy:  {before_r['accuracy']:.4f} -> {after_r['accuracy']:.4f} (change={acc_change:+.4f})")
        print(f"  Ratio:     {before_r['ratio']:.4f} -> {after_r['ratio']:.4f} (change={ratio_change:+.4f})")
        print(f"  Std:       {before_r['std']:.4f} -> {after_r['std']:.4f} (change={std_change:+.4f})")
    
    print("\n" + "="*100)
    print("Notes:")
    print("- Accuracy: Linear classifier test accuracy (higher is better)")
    print("- Intra-Dist: Average L2 distance within same class (lower = more coherent classes)")
    print("- Inter-Dist: Average L2 distance between different classes (higher = better separation)")
    print("- Ratio: Inter/Intra (higher = better class separation)")
    print("- Std: Standard deviation of activations (should be > 0.1 to avoid collapse)")
    print("="*100)


if __name__ == "__main__":
    main()

