"""Compare BYOL and LPL embeddings using linear probes and distance metrics."""

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
    
    # Inter-class distances
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
    embeddings = data['embeddings']
    labels = data['labels']
    
    # Check for NaN
    has_nan = torch.isnan(embeddings).any().item()
    std = embeddings.std().item()
    
    # Compute distances
    intra_dist, inter_dist, ratio = compute_distances(embeddings, labels)
    
    # Linear probe
    accuracy = linear_probe(embeddings, labels)
    
    # L2 norm statistics (for BYOL, should be ~1.0)
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
    """Compare BYOL and LPL embeddings."""
    
    output_dir = Path('outputs/activations')
    
    print("="*100)
    print("BYOL vs LPL COMPARISON")
    print("="*100)
    
    results = []
    
    # Load BYOL synthetic shapes results
    byol_before = output_dir / 'byol_shapes_embeddings_before.pt'
    byol_after = output_dir / 'byol_shapes_embeddings_after.pt'
    
    if byol_before.exists() and byol_after.exists():
        results.append(analyze_embeddings(byol_before, 'BYOL', 'Before'))
        results.append(analyze_embeddings(byol_after, 'BYOL', 'After'))
    
    # Load LPL hierarchical results (use Layer1 for comparison)
    lpl_before = output_dir / 'hierarchical_activations_before.pt'
    lpl_after = output_dir / 'hierarchical_activations_after.pt'
    
    if lpl_before.exists() and lpl_after.exists():
        data_before = torch.load(lpl_before, map_location='cpu')
        data_after = torch.load(lpl_after, map_location='cpu')
        
        # Save Layer1 temporarily for analysis
        temp_before = {'embeddings': data_before['layer1_activations'], 'labels': data_before['labels']}
        temp_after = {'embeddings': data_after['layer1_activations'], 'labels': data_after['labels']}
        
        torch.save(temp_before, output_dir / 'temp_lpl_before.pt')
        torch.save(temp_after, output_dir / 'temp_lpl_after.pt')
        
        results.append(analyze_embeddings(output_dir / 'temp_lpl_before.pt', 'LPL (Layer1)', 'Before'))
        results.append(analyze_embeddings(output_dir / 'temp_lpl_after.pt', 'LPL (Layer1)', 'After'))
    
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
    if len(results) >= 4:
        byol_after = next(r for r in results if r['method'] == 'BYOL' and r['stage'] == 'After')
        lpl_after = next(r for r in results if r['method'] == 'LPL (Layer1)' and r['stage'] == 'After')
        
        print("\n" + "="*100)
        print("DETAILED COMPARISON (After Training)")
        print("="*100)
        
        print(f"\n[Classification Performance]")
        print(f"  BYOL Accuracy: {byol_after['accuracy']:.4f}")
        print(f"  LPL Accuracy:  {lpl_after['accuracy']:.4f}")
        better_acc = "BYOL" if byol_after['accuracy'] > lpl_after['accuracy'] else "LPL"
        print(f"  -> {better_acc} has better classification accuracy")
        
        print(f"\n[Class Separation]")
        print(f"  BYOL Inter/Intra Ratio: {byol_after['ratio']:.4f}")
        print(f"  LPL Inter/Intra Ratio:  {lpl_after['ratio']:.4f}")
        better_sep = "BYOL" if byol_after['ratio'] > lpl_after['ratio'] else "LPL"
        print(f"  -> {better_sep} has better class separation")
        
        print(f"\n[Representation Statistics]")
        print(f"  BYOL Std: {byol_after['std']:.4f} (L2 normalized, constrained)")
        print(f"  LPL Std:  {lpl_after['std']:.4f} (not normalized, unbounded)")
        print(f"  BYOL L2 Norm: {byol_after['mean_norm']:.4f} (all = 1.0 by design)")
        print(f"  LPL L2 Norm:  {lpl_after['mean_norm']:.4f} (varies)")
        
        print(f"\n[Key Differences]")
        print(f"  1. Normalization:")
        print(f"     - BYOL: Embeddings are L2 normalized (unit length)")
        print(f"     - LPL:  Embeddings are not normalized (tanh squashing to [-5, 5])")
        print(f"  2. Learning Mechanism:")
        print(f"     - BYOL: Contrastive learning (predict target network projection)")
        print(f"     - LPL:  Predictive learning (predict next representation)")
        print(f"  3. Objective:")
        print(f"     - BYOL: Minimize distance between augmented views")
        print(f"     - LPL:  Minimize prediction error + Hebbian + stabilization")
        print(f"  4. Representation Space:")
        print(f"     - BYOL: Constrained to unit hypersphere")
        print(f"     - LPL:  Unbounded but clipped to prevent explosion")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()


