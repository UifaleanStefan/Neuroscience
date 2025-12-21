"""Compare BYOL and LPL embeddings on CIFAR-10."""

import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def compute_distances(embeddings, labels):
    """Compute intra-class and inter-class L2 distances."""
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    unique_labels = np.unique(labels_np)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        return 0.0, 0.0, 0.0
    
    # Intra-class distances
    intra_distances = []
    for label in unique_labels:
        class_mask = labels_np == label
        class_embeddings = embeddings_np[class_mask]
        
        if len(class_embeddings) < 2:
            continue
        
        for i in range(len(class_embeddings)):
            for j in range(i + 1, len(class_embeddings)):
                dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                intra_distances.append(dist)
    
    # Inter-class distances (sample to avoid excessive computation)
    inter_distances = []
    max_pairs = 10000
    pair_count = 0
    
    for i, label_i in enumerate(unique_labels):
        class_i_mask = labels_np == label_i
        class_i_embeddings = embeddings_np[class_i_mask]
        
        for j, label_j in enumerate(unique_labels):
            if i >= j:
                continue
            
            class_j_mask = labels_np == label_j
            class_j_embeddings = embeddings_np[class_j_mask]
            
            for emb_i in class_i_embeddings:
                for emb_j in class_j_embeddings:
                    if pair_count >= max_pairs:
                        break
                    dist = np.linalg.norm(emb_i - emb_j)
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


def linear_probe(embeddings, labels, random_seed=42):
    """Train linear classifier on embeddings."""
    X = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    clf = LogisticRegression(max_iter=1000, random_state=random_seed, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def analyze_embeddings(filepath, method_name, stage):
    """Analyze embeddings and return metrics."""
    data = torch.load(filepath, map_location='cpu')
    # Handle both 'embeddings' (BYOL) and 'activations' (LPL) keys
    if 'embeddings' in data:
        embeddings = data['embeddings']
    elif 'activations' in data:
        embeddings = data['activations']
    else:
        raise KeyError(f"Neither 'embeddings' nor 'activations' found in {filepath}")
    labels = data['labels']
    
    # Check for NaN
    has_nan = torch.isnan(embeddings).any().item()
    std = embeddings.std().item()
    
    # Compute distances
    intra_dist, inter_dist, ratio = compute_distances(embeddings, labels)
    
    # Linear probe
    accuracy = linear_probe(embeddings, labels)
    
    # L2 norm statistics
    norms = torch.norm(embeddings, dim=1)
    mean_norm = norms.mean().item()
    
    return {
        'method': method_name,
        'stage': stage,
        'std': std,
        'has_nan': has_nan,
        'accuracy': accuracy,
        'intra_dist': intra_dist,
        'inter_dist': inter_dist,
        'ratio': ratio,
        'mean_norm': mean_norm,
    }


def main():
    """Compare BYOL and LPL embeddings on CIFAR-10."""
    
    output_dir = Path('outputs/activations')
    
    print("="*100)
    print("BYOL vs LPL COMPARISON ON CIFAR-10")
    print("="*100)
    
    results = []
    
    # Load BYOL CIFAR-10 results
    byol_before = output_dir / 'byol_embeddings_before.pt'
    byol_after = output_dir / 'byol_embeddings_after.pt'
    
    if byol_before.exists() and byol_after.exists():
        print("Loading BYOL embeddings...")
        results.append(analyze_embeddings(byol_before, 'BYOL', 'Before'))
        results.append(analyze_embeddings(byol_after, 'BYOL', 'After'))
    else:
        print(f"Warning: BYOL CIFAR-10 files not found")
    
    # Load LPL CIFAR-10 results (from train_lpl.py)
    lpl_before = output_dir / 'activations_before.pt'
    lpl_after = output_dir / 'activations_after.pt'
    
    if lpl_before.exists() and lpl_after.exists():
        print("Loading LPL embeddings...")
        results.append(analyze_embeddings(lpl_before, 'LPL', 'Before'))
        results.append(analyze_embeddings(lpl_after, 'LPL', 'After'))
    else:
        print(f"Warning: LPL CIFAR-10 files not found")
    
    # Print comparison table
    print(f"\n{'Method':<20} {'Stage':<10} {'Accuracy':<12} {'Intra-Dist':<12} {'Inter-Dist':<12} {'Ratio':<10} {'Std':<10} {'L2 Norm':<10}")
    print("-"*100)
    
    for r in results:
        print(f"{r['method']:<20} {r['stage']:<10} "
              f"{r['accuracy']:<12.4f} "
              f"{r['intra_dist']:<12.4f} "
              f"{r['inter_dist']:<12.4f} "
              f"{r['ratio']:<10.4f} "
              f"{r['std']:<10.4f} "
              f"{r['mean_norm']:<10.4f}")
    
    print("="*100)
    
    # Detailed comparison
    byol_after = next((r for r in results if r['method'] == 'BYOL' and r['stage'] == 'After'), None)
    lpl_after = next((r for r in results if r['method'] == 'LPL' and r['stage'] == 'After'), None)
    
    if byol_after and lpl_after:
        print("\n" + "="*100)
        print("DETAILED COMPARISON (After Training)")
        print("="*100)
        
        print(f"\n[Classification Performance]")
        print(f"  BYOL Accuracy: {byol_after['accuracy']:.4f}")
        print(f"  LPL Accuracy:  {lpl_after['accuracy']:.4f}")
        acc_diff = byol_after['accuracy'] - lpl_after['accuracy']
        print(f"  Difference: {acc_diff:+.4f} ({'BYOL' if acc_diff > 0 else 'LPL'} is better)")
        
        print(f"\n[Class Separation]")
        print(f"  BYOL Inter/Intra Ratio: {byol_after['ratio']:.4f}")
        print(f"  LPL Inter/Intra Ratio:  {lpl_after['ratio']:.4f}")
        ratio_diff = byol_after['ratio'] - lpl_after['ratio']
        print(f"  Difference: {ratio_diff:+.4f} ({'BYOL' if ratio_diff > 0 else 'LPL'} has better separation)")
        
        print(f"\n[Representation Statistics]")
        print(f"  BYOL Std: {byol_after['std']:.4f} (L2 normalized, constrained to unit sphere)")
        print(f"  LPL Std:  {lpl_after['std']:.4f} (tanh squashed to [-5, 5])")
        print(f"  BYOL L2 Norm: {byol_after['mean_norm']:.4f} (all = 1.0 by design)")
        print(f"  LPL L2 Norm:  {lpl_after['mean_norm']:.4f} (varies)")
        
        print(f"\n[Key Differences - CIFAR-10 Results]")
        print(f"  1. Learning Objective:")
        print(f"     - BYOL: Minimizes distance between augmented views (contrastive)")
        print(f"     - LPL:  Minimizes prediction error + Hebbian + stabilization (predictive)")
        print(f"  2. Representation Space:")
        print(f"     - BYOL: Unit hypersphere (normalized, constrained)")
        print(f"     - LPL:  Unbounded but clipped space (tanh to [-5, 5])")
        print(f"  3. Update Strategy:")
        print(f"     - BYOL: Only first conv layer updated (local learning)")
        print(f"     - LPL:  All weights updated using local learning rules")
        print(f"  4. Convergence:")
        print(f"     - BYOL: Steady loss decrease (0.002-0.003 range)")
        print(f"     - LPL:  Direct local learning rules, no explicit loss")
        
    print("\n" + "="*100)


if __name__ == "__main__":
    main()

