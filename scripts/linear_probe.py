"""Linear probe script to evaluate LPL activations using linear classification."""

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
    Compute average intra-class and inter-class distances.
    
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
    for i, label_i in enumerate(unique_labels):
        class_i_mask = labels_np == label_i
        class_i_activations = activations_np[class_i_mask]
        
        for j, label_j in enumerate(unique_labels):
            if i >= j:  # Avoid double counting
                continue
            
            class_j_mask = labels_np == label_j
            class_j_activations = activations_np[class_j_mask]
            
            # Compute pairwise distances between classes
            for act_i in class_i_activations:
                for act_j in class_j_activations:
                    dist = np.linalg.norm(act_i - act_j)
                    inter_distances.append(dist)
    
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
    # No regularization needed for probing, but use small C to avoid warnings
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


def main():
    """Main function to run linear probes on all activation files."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    activations_dir = Path('outputs/activations')
    
    if not activations_dir.exists():
        print(f"Directory {activations_dir} does not exist!")
        return
    
    # Find all .pt files except swap_experiment.pt
    all_pt_files = list(activations_dir.glob('*.pt'))
    pt_files = [f for f in all_pt_files if f.name != 'swap_experiment.pt']
    
    if not pt_files:
        print(f"No activation files found in {activations_dir} (excluding swap_experiment.pt)")
        return
    
    print(f"Found {len(pt_files)} activation file(s) to probe\n")
    
    # Store results
    results = []
    
    # Process each file
    for pt_file in sorted(pt_files):
        try:
            # Load the file
            data = torch.load(pt_file, map_location='cpu')
            
            # Extract activations and labels
            if isinstance(data, dict):
                if 'activations' in data and 'labels' in data:
                    activations = data['activations']
                    labels = data['labels']
                else:
                    print(f"Skipping {pt_file.name}: missing 'activations' or 'labels' keys")
                    continue
            else:
                print(f"Skipping {pt_file.name}: unexpected format")
                continue
            
            # Check for NaN values
            if torch.isnan(activations).any():
                print(f"WARNING: {pt_file.name} contains NaN values. Skipping.")
                continue
            
            # Train linear probe
            test_accuracy = linear_probe(activations, labels)
            
            # Compute distances
            intra_dist, inter_dist, ratio = compute_distances(activations, labels)
            
            results.append({
                'filename': pt_file.name,
                'accuracy': test_accuracy,
                'intra_dist': intra_dist,
                'inter_dist': inter_dist,
                'ratio': ratio
            })
            
        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results table
    print("\n" + "="*90)
    print("LINEAR PROBE RESULTS")
    print("="*90)
    print(f"{'Filename':<40} {'Test Accuracy':<15} {'Intra-Class Dist':<18} {'Inter-Class Dist':<18} {'Ratio':<10}")
    print("-"*90)
    
    for result in results:
        print(f"{result['filename']:<40} "
              f"{result['accuracy']:<15.4f} "
              f"{result['intra_dist']:<18.4f} "
              f"{result['inter_dist']:<18.4f} "
              f"{result['ratio']:<10.4f}")
    
    print("="*90)
    print("\nNotes:")
    print("- Test Accuracy: Linear classifier accuracy on 20% held-out test set")
    print("- Intra-Class Dist: Average L2 distance between samples within same class")
    print("- Inter-Class Dist: Average L2 distance between samples from different classes")
    print("- Ratio: Inter / Intra (higher ratio = better class separation)")


if __name__ == "__main__":
    main()


