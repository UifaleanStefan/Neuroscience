"""
Metrics computation for activation analysis.

Implements:
- Linear probe (with multiple random splits)
- Intra/inter-class distance analysis
- Variance & collapse diagnostics
- Before/after deltas
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def normalize_activation_format(data: Dict) -> Dict:
    """
    Normalize activation data to standard format.
    
    Handles legacy formats:
    - {'activations': Tensor, 'labels': Tensor}
    - {'layer1_activations': Tensor, 'layer2_activations': Tensor, 'labels': Tensor}
    
    Converts to:
    - {'layer1': Tensor, 'layer2': Tensor (optional), 'labels': Tensor}
    """
    normalized = {}
    
    # Handle labels (should always be present)
    if 'labels' in data:
        normalized['labels'] = data['labels']
    else:
        raise ValueError("Missing 'labels' key in activation data")
    
    # Handle layer1 (single layer format)
    if 'activations' in data:
        normalized['layer1'] = data['activations']
    elif 'layer1' in data:
        normalized['layer1'] = data['layer1']
    elif 'layer1_activations' in data:
        normalized['layer1'] = data['layer1_activations']
    else:
        raise ValueError("Missing layer1 activations in activation data")
    
    # Handle layer2 (optional, multi-layer format)
    if 'layer2' in data:
        normalized['layer2'] = data['layer2']
    elif 'layer2_activations' in data:
        normalized['layer2'] = data['layer2_activations']
    # If not present, that's fine (single layer model)
    
    return normalized


def linear_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    n_splits: int = 3,
    random_seed: int = 42,
    test_size: float = 0.2
) -> Dict[str, float]:
    """
    Train linear classifier on frozen activations with multiple random splits.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        n_splits: Number of random train/test splits to average over
        random_seed: Random seed for reproducibility
        test_size: Fraction of data to use for testing
        
    Returns:
        Dictionary with 'accuracy_mean' and 'accuracy_std'
    """
    X = activations.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    accuracies = []
    
    for split in range(n_splits):
        # Use different random state for each split
        split_seed = random_seed + split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed, stratify=y
        )
        
        # Train linear classifier
        clf = LogisticRegression(
            max_iter=1000,
            random_state=split_seed,
            C=1.0,
            solver='lbfgs',
            n_jobs=1
        )
        
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        except Exception as e:
            warnings.warn(f"Linear probe failed for split {split}: {e}")
            continue
    
    if len(accuracies) == 0:
        return {'accuracy_mean': np.nan, 'accuracy_std': np.nan}
    
    return {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies) if len(accuracies) > 1 else 0.0
    }


def compute_class_distances(
    activations: torch.Tensor,
    labels: torch.Tensor,
    max_pairs: int = 10000
) -> Dict[str, float]:
    """
    Compute intra-class and inter-class L2 distances.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        max_pairs: Maximum number of pairs to compute (for efficiency)
        
    Returns:
        Dictionary with 'intra_mean', 'inter_mean', 'ratio', and per-class breakdowns
    """
    activations_np = activations.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    unique_labels = np.unique(labels_np)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        return {
            'intra_mean': 0.0,
            'inter_mean': 0.0,
            'ratio': 0.0,
            'intra_std': 0.0,
            'inter_std': 0.0
        }
    
    # Compute intra-class distances (within same class)
    intra_distances = []
    per_class_intra = {}
    
    for label in unique_labels:
        class_mask = labels_np == label
        class_activations = activations_np[class_mask]
        
        if len(class_activations) < 2:
            continue
        
        class_intra = []
        # Compute pairwise distances within this class
        for i in range(len(class_activations)):
            for j in range(i + 1, len(class_activations)):
                dist = np.linalg.norm(class_activations[i] - class_activations[j])
                intra_distances.append(dist)
                class_intra.append(dist)
        
        per_class_intra[int(label)] = {
            'mean': np.mean(class_intra) if class_intra else 0.0,
            'std': np.std(class_intra) if len(class_intra) > 1 else 0.0,
            'count': len(class_intra)
        }
    
    # Compute inter-class distances (between different classes)
    inter_distances = []
    pair_count = 0
    
    for i, label_i in enumerate(unique_labels):
        if pair_count >= max_pairs:
            break
            
        class_i_mask = labels_np == label_i
        class_i_activations = activations_np[class_i_mask]
        
        for j, label_j in enumerate(unique_labels):
            if i >= j:  # Avoid double counting
                continue
            
            if pair_count >= max_pairs:
                break
            
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
    std_intra = np.std(intra_distances) if len(intra_distances) > 1 else 0.0
    avg_inter = np.mean(inter_distances) if inter_distances else 0.0
    std_inter = np.std(inter_distances) if len(inter_distances) > 1 else 0.0
    ratio = avg_inter / avg_intra if avg_intra > 0 else 0.0
    
    return {
        'intra_mean': float(avg_intra),
        'intra_std': float(std_intra),
        'inter_mean': float(avg_inter),
        'inter_std': float(std_inter),
        'ratio': float(ratio),
        'per_class_intra': per_class_intra
    }


def compute_variance_diagnostics(
    activations: torch.Tensor,
    collapse_threshold: float = 0.1,
    saturation_threshold: float = 0.99
) -> Dict:
    """
    Compute variance and collapse diagnostics.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        collapse_threshold: Std threshold below which representation is considered collapsed
        saturation_threshold: Threshold for checking tanh saturation (if applicable)
        
    Returns:
        Dictionary with variance statistics and collapse flags
    """
    activations_np = activations.detach().cpu().numpy()
    
    # Global statistics
    mean = activations_np.mean()
    std = activations_np.std()
    
    # Per-dimension statistics
    per_dim_std = activations_np.std(axis=0)
    per_dim_mean = activations_np.mean(axis=0)
    
    # Percentage of dimensions with near-zero variance
    near_zero_threshold = 0.01
    near_zero_variance_dims = (per_dim_std < near_zero_threshold).sum()
    pct_near_zero = (near_zero_variance_dims / activations_np.shape[1]) * 100.0
    
    # Check for collapse
    collapsed = std < collapse_threshold
    
    # Check for saturation (if using tanh, values should be in [-1, 1])
    # Check if many values are at boundaries
    min_val = activations_np.min()
    max_val = activations_np.max()
    
    # Count values near boundaries (within 0.01 of -1 or 1, if in tanh range)
    if min_val >= -1.1 and max_val <= 1.1:
        at_lower_bound = (activations_np < -0.99).sum()
        at_upper_bound = (activations_np > 0.99).sum()
        pct_saturated = ((at_lower_bound + at_upper_bound) / activations_np.size) * 100.0
    else:
        pct_saturated = 0.0
    
    saturated = pct_saturated > (saturation_threshold * 100.0)
    
    # Determine status flag
    if collapsed:
        status = 'COLLAPSED'
    elif saturated:
        status = 'SATURATED'
    else:
        status = 'HEALTHY'
    
    # Norm distribution (L2 norm per sample)
    norms = np.linalg.norm(activations_np, axis=1)
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())
    
    return {
        'mean': float(mean),
        'std': float(std),
        'std_per_dim_mean': float(per_dim_std.mean()),
        'std_per_dim_std': float(per_dim_std.std()),
        'pct_near_zero_variance': float(pct_near_zero),
        'min': float(min_val),
        'max': float(max_val),
        'norm_mean': norm_mean,
        'norm_std': norm_std,
        'collapse_flag': collapsed,
        'saturation_flag': saturated,
        'status': status,
        'pct_saturated': float(pct_saturated)
    }


def analyze_single_file(
    filepath: str,
    layer_name: str = 'layer1',
    stage: str = 'after',
    random_seed: int = 42
) -> Dict:
    """
    Analyze a single activation file.
    
    Args:
        filepath: Path to .pt file
        layer_name: Which layer to analyze ('layer1' or 'layer2')
        stage: Stage identifier ('before' or 'after')
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with all computed metrics
    """
    import torch
    from pathlib import Path
    
    # Load data
    data = torch.load(filepath, map_location='cpu')
    
    # Normalize format
    normalized = normalize_activation_format(data)
    
    # Extract layer activations
    if layer_name not in normalized:
        raise ValueError(f"Layer {layer_name} not found in activation file")
    
    activations = normalized[layer_name]
    labels = normalized['labels']
    
    # Check for NaN
    has_nan = torch.isnan(activations).any().item()
    if has_nan:
        return {
            'layer': layer_name,
            'stage': stage,
            'has_nan': True,
            'error': 'NaN values detected in activations'
        }
    
    # Compute all metrics
    probe_results = linear_probe(activations, labels, random_seed=random_seed)
    distance_results = compute_class_distances(activations, labels)
    variance_results = compute_variance_diagnostics(activations)
    
    # Combine results
    results = {
        'layer': layer_name,
        'stage': stage,
        'has_nan': False,
        **probe_results,
        **distance_results,
        **variance_results
    }
    
    return results


def compute_before_after_deltas(before_results: Dict, after_results: Dict) -> Dict:
    """
    Compute deltas between before and after training.
    
    Args:
        before_results: Results dictionary from before training
        after_results: Results dictionary from after training
        
    Returns:
        Dictionary with delta metrics
    """
    deltas = {}
    
    # Accuracy delta
    if 'accuracy_mean' in before_results and 'accuracy_mean' in after_results:
        deltas['delta_accuracy'] = (
            after_results['accuracy_mean'] - before_results['accuracy_mean']
        )
    
    # Distance deltas
    if 'intra_mean' in before_results and 'intra_mean' in after_results:
        deltas['delta_intra'] = (
            after_results['intra_mean'] - before_results['intra_mean']
        )
    
    if 'inter_mean' in before_results and 'inter_mean' in after_results:
        deltas['delta_inter'] = (
            after_results['inter_mean'] - before_results['inter_mean']
        )
    
    # Ratio delta
    if 'ratio' in before_results and 'ratio' in after_results:
        deltas['delta_ratio'] = (
            after_results['ratio'] - before_results['ratio']
        )
    
    # Variance delta
    if 'std' in before_results and 'std' in after_results:
        deltas['delta_std'] = (
            after_results['std'] - before_results['std']
        )
    
    return deltas


def compare_layers(
    layer1_results: Dict,
    layer2_results: Dict
) -> Dict:
    """
    Compare two layers from the same model/stage.
    
    Args:
        layer1_results: Results for layer 1
        layer2_results: Results for layer 2
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    
    # Separation ratio gap
    if 'ratio' in layer1_results and 'ratio' in layer2_results:
        comparison['ratio_gap'] = (
            layer2_results['ratio'] - layer1_results['ratio']
        )
    
    # Linear probe gap
    if 'accuracy_mean' in layer1_results and 'accuracy_mean' in layer2_results:
        comparison['accuracy_gap'] = (
            layer2_results['accuracy_mean'] - layer1_results['accuracy_mean']
        )
    
    # Variance gap
    if 'std' in layer1_results and 'std' in layer2_results:
        comparison['variance_gap'] = (
            layer2_results['std'] - layer1_results['std']
        )
    
    return comparison

