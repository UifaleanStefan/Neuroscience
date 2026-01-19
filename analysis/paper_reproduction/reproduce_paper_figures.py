"""
Reproduce figures from Halvagal & Zenke (2023) Nature Neuroscience paper.

This script attempts to reproduce key figures using available datasets and models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = Path("analysis/paper_reproduction")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.joinpath("metrics").mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("json").mkdir(exist_ok=True)

def load_activation_file(filepath):
    """Load activation file and handle different formats."""
    data = torch.load(filepath, map_location='cpu')
    
    if 'activations' in data:
        activations = data['activations']
        labels = data['labels']
    elif 'layer1_activations' in data:
        activations = data['layer1_activations']
        labels = data['labels']
    elif 'mlp_layer1_activations' in data:
        activations = data['mlp_layer1_activations']
        labels = data['labels']
    else:
        raise ValueError(f"Unknown format. Keys: {list(data.keys())}")
    
    return activations.numpy(), labels.numpy()

def compute_linear_probe(activations, labels, n_splits=3):
    """Compute linear readout accuracy."""
    accuracies = []
    for split in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels, test_size=0.2, random_state=42+split, stratify=labels
        )
        clf = LogisticRegression(max_iter=1000, random_state=42+split, C=1.0, solver='lbfgs', n_jobs=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    return np.mean(accuracies), np.std(accuracies)

def compute_participation_ratio(activations):
    """
    Compute participation ratio (effective dimensionality) of representations.
    
    Participation ratio = (trace(C))^2 / trace(C^2)
    where C is the covariance matrix of activations.
    """
    # Center activations
    activations_centered = activations - activations.mean(axis=0, keepdims=True)
    
    # Compute covariance matrix
    C = np.cov(activations_centered.T)
    
    # Participation ratio
    trace_C = np.trace(C)
    trace_C2 = np.trace(C @ C)
    
    if trace_C2 > 0:
        participation_ratio = (trace_C ** 2) / trace_C2
    else:
        participation_ratio = 0.0
    
    return float(participation_ratio)

def compute_mean_activity(activations):
    """Compute mean activity (average activation magnitude) per sample."""
    return float(np.mean(np.abs(activations)))

def figure_3_representation_metrics():
    """
    FIGURE 3: Deep Representation Metrics
    - Linear Readout vs Layer (for multi-layer models)
    - Representation Dimensionality vs Layer
    - Mean Activity vs Layer
    
    This shows metrics ACROSS LAYERS within the same model (depth progression),
    not across different experiments (training steps).
    """
    print("="*70)
    print("FIGURE 3: Deep Representation Metrics")
    print("="*70)
    
    datasets_to_check = {
        'Synthetic Shapes': Path("outputs/grid_experiments"),
        'MNIST': Path("outputs/grid_experiments"),
        'Fashion-MNIST': Path("outputs/grid_experiments"),
        'CIFAR-10': Path("outputs/grid_experiments"),
        'STL-10': Path("outputs/grid_experiments"),
    }
    
    results = {}
    
    # Find multi-layer models (2-layer and 3-layer MLPs)
    for dataset_name, base_dir in datasets_to_check.items():
        print(f"\nProcessing {dataset_name}...")
        
        # Map dataset name to prefix
        dataset_prefix_map = {
            'Synthetic Shapes': 'synthetic_shapes',
            'MNIST': 'mnist',
            'Fashion-MNIST': 'fashion_mnist',
            'CIFAR-10': 'cifar10',
            'STL-10': 'stl10'
        }
        
        dataset_prefix = dataset_prefix_map.get(dataset_name)
        if not dataset_prefix:
            print(f"  Warning: Unknown dataset {dataset_name}, skipping")
            continue
        
        # Find 2-layer and 3-layer experiments
        # Prefer 3-layer for more depth, but use 2-layer if 3-layer not available
        layer3_dirs = list(base_dir.glob(f"run_*_{dataset_prefix}_*_mlp_3layer_*_tanh_full_lpl"))
        layer2_dirs = list(base_dir.glob(f"run_*_{dataset_prefix}_*_mlp_2layer_*_tanh_full_lpl"))
        
        # Choose a single experiment to analyze (prefer 3-layer, then 2-layer)
        # Use 5k or 10k steps for good training (avoid 1k too short, 50k may collapse)
        selected_dir = None
        
        if layer3_dirs:
            # Try to find 3-layer with 5k or 10k steps
            for steps in [5000, 10000]:
                for dir_path in layer3_dirs:
                    if f"{steps}steps" in dir_path.name:
                        selected_dir = dir_path
                        break
                if selected_dir:
                    break
            # Fallback to any 3-layer
            if not selected_dir and layer3_dirs:
                selected_dir = layer3_dirs[0]
        
        if not selected_dir and layer2_dirs:
            # Try to find 2-layer with 5k or 10k steps
            for steps in [5000, 10000]:
                for dir_path in layer2_dirs:
                    if f"{steps}steps" in dir_path.name:
                        selected_dir = dir_path
                        break
                if selected_dir:
                    break
            # Fallback to any 2-layer
            if not selected_dir and layer2_dirs:
                selected_dir = layer2_dirs[0]
        
        if not selected_dir:
            print(f"  No multi-layer models found for {dataset_name}")
            continue
        
        print(f"  Selected model: {selected_dir.name}")
        
        # Load activations from this single model
        activations_file = selected_dir / "activations_after.pt"
        if not activations_file.exists():
            print(f"  Activation file not found: {activations_file}")
            continue
        
        try:
            data = torch.load(activations_file, map_location='cpu')
        except Exception as e:
            print(f"  Error loading {activations_file}: {e}")
            continue
        
        # Extract layer activations from this model
        all_layers_data = []
        
        # Check for 3-layer model
        if 'layer1_activations' in data and 'layer2_activations' in data and 'layer3_activations' in data:
            # 3-layer model: layer1 → layer2 → layer3
            labels = data['labels'].numpy()
            
            for layer_key, layer_name in [
                ('layer1_activations', 'Layer 1'),
                ('layer2_activations', 'Layer 2'),
                ('layer3_activations', 'Layer 3')
            ]:
                if layer_key in data:
                    activations = data[layer_key].numpy()
                    all_layers_data.append({
                        'layer': layer_name,
                        'activations': activations,
                        'labels': labels
                    })
        
        # Check for 2-layer model
        elif 'layer1_activations' in data and 'layer2_activations' in data:
            # 2-layer model: layer1 → layer2
            labels = data['labels'].numpy()
            
            for layer_key, layer_name in [
                ('layer1_activations', 'Layer 1'),
                ('layer2_activations', 'Layer 2')
            ]:
                if layer_key in data:
                    activations = data[layer_key].numpy()
                    all_layers_data.append({
                        'layer': layer_name,
                        'activations': activations,
                        'labels': labels
                    })
        else:
            print(f"  Warning: {selected_dir.name} does not have multi-layer activations")
            print(f"  Available keys: {list(data.keys())}")
            continue
        
        if not all_layers_data:
            print(f"  No layer data extracted for {dataset_name}")
            continue
        
        # Compute metrics for each layer within this model
        linear_readouts = []
        participation_ratios = []
        mean_activities = []
        layer_names = []
        
        for layer_data in all_layers_data:
            activations = layer_data['activations']
            labels = layer_data['labels']
            
            # Sample subset if too large
            if len(activations) > 1000:
                indices = np.random.choice(len(activations), 1000, replace=False)
                activations_sub = activations[indices]
                labels_sub = labels[indices]
            else:
                activations_sub = activations
                labels_sub = labels
            
            # Linear readout
            acc_mean, acc_std = compute_linear_probe(activations_sub, labels_sub)
            linear_readouts.append(acc_mean)
            
            # Participation ratio
            pr = compute_participation_ratio(activations_sub)
            participation_ratios.append(pr)
            
            # Mean activity
            ma = compute_mean_activity(activations_sub)
            mean_activities.append(ma)
            
            layer_names.append(layer_data['layer'])
        
        results[dataset_name] = {
            'linear_readout': {
                'values': linear_readouts,
                'layers': layer_names
            },
            'participation_ratio': {
                'values': participation_ratios,
                'layers': layer_names
            },
            'mean_activity': {
                'values': mean_activities,
                'layers': layer_names
            }
        }
        
        print(f"  Computed metrics for {len(layer_names)} layers: {layer_names}")
    
    # Generate plots
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Linear Readout
        ax = axes[0]
        for dataset_name, data in results.items():
            layers = data['linear_readout']['layers']
            values = data['linear_readout']['values']
            ax.plot(range(len(values)), values, marker='o', label=dataset_name)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Linear Readout Accuracy')
        ax.set_title('Linear Readout vs Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Participation Ratio
        ax = axes[1]
        for dataset_name, data in results.items():
            layers = data['participation_ratio']['layers']
            values = data['participation_ratio']['values']
            ax.plot(range(len(values)), values, marker='o', label=dataset_name)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Participation Ratio (Dimensionality)')
        ax.set_title('Representation Dimensionality vs Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean Activity
        ax = axes[2]
        for dataset_name, data in results.items():
            layers = data['mean_activity']['layers']
            values = data['mean_activity']['values']
            ax.plot(range(len(values)), values, marker='o', label=dataset_name)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Mean Activity')
        ax.set_title('Mean Activity vs Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "figure_3_representation_metrics.png", bbox_inches='tight')
        print(f"\nSaved: {OUTPUT_DIR / 'figure_3_representation_metrics.png'}")
        plt.close()
    
    return results

def figure_2_single_neuron_selectivity():
    """
    FIGURE 2: Single Neuron LPL Selectivity
    Creates synthetic 2D clusters with temporal transitions.
    """
    print("="*70)
    print("FIGURE 2: Single Neuron LPL Selectivity")
    print("="*70)
    print("Note: This requires implementing single-neuron LPL training.")
    print("Skipping for now - would need custom implementation.")
    
    # Placeholder for future implementation
    return None

def figure_4_swap_selectivity():
    """
    FIGURE 4: Swap Selectivity Change
    Uses swap experiment results we already have.
    NOTE: After-swap activations have much larger scale due to ReLU unbounded growth.
    This is expected behavior, not a bug.
    """
    print("="*70)
    print("FIGURE 4: Swap Selectivity Change")
    print("="*70)
    
    swap_file = Path("outputs/activations/swap_experiment.pt")
    
    if not swap_file.exists():
        print("Swap experiment file not found. Run swap experiment first.")
        return None
    
    data = torch.load(swap_file, map_location='cpu')
    activations_before = data['activations_before'].numpy()
    activations_after = data['activations_after'].numpy()
    labels_before = data['labels_before'].numpy()
    labels_after = data['labels_after'].numpy()
    
    print(f"Loaded swap data: {len(activations_before)} samples")
    print(f"Before scale: mean={activations_before.mean():.2f}, max={activations_before.max():.2f}")
    print(f"After scale: mean={activations_after.mean():.2f}, max={activations_after.max():.2f}")
    print("NOTE: Scale difference is expected (ReLU unbounded growth during training)")
    
    # Compute selectivity changes
    # For each class, compute mean activation change
    unique_labels = np.unique(labels_before)
    
    selectivity_before = {}
    selectivity_after = {}
    
    for label in unique_labels:
        mask_before = labels_before == label
        mask_after = labels_after == label
        
        # Mean activation per class (averaged across neurons)
        if mask_before.sum() > 0:
            selectivity_before[label] = activations_before[mask_before].mean(axis=0)
        if mask_after.sum() > 0:
            selectivity_after[label] = activations_after[mask_after].mean(axis=0)
    
    # Normalize for visualization (divide by max to show relative selectivity patterns)
    # But keep raw values in the returned data
    max_before = max([sel.max() for sel in selectivity_before.values()]) if selectivity_before else 1.0
    max_after = max([sel.max() for sel in selectivity_after.values()]) if selectivity_after else 1.0
    
    # Plot selectivity change
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before swap
    ax = axes[0]
    for label in sorted(unique_labels[:10]):  # Show all 10 classes (CIFAR-10 swap uses 10 classes)
        if label in selectivity_before:
            # Use normalized values for visualization
            sel_norm = selectivity_before[label] / max_before if max_before > 0 else selectivity_before[label]
            ax.plot(sel_norm[:128], label=f'Class {label}', alpha=0.7)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Normalized Mean Activation')
    ax.set_title('Selectivity Before Swap (Normalized)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # After swap
    ax = axes[1]
    for label in sorted(unique_labels[:10]):
        if label in selectivity_after:
            # Use normalized values for visualization (scale differs dramatically)
            sel_norm = selectivity_after[label] / max_after if max_after > 0 else selectivity_after[label]
            ax.plot(sel_norm[:128], label=f'Class {label}', alpha=0.7)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Normalized Mean Activation')
    ax.set_title('Selectivity After Swap (Normalized)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_swap_selectivity.png", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'figure_4_swap_selectivity.png'}")
    plt.close()
    
    return {
        'selectivity_before': {int(k): v.tolist() for k, v in selectivity_before.items()},
        'selectivity_after': {int(k): v.tolist() for k, v in selectivity_after.items()}
    }

def compute_ablation_comparison():
    """Compare full LPL vs ablation conditions (for paper comparison)."""
    print("="*70)
    print("Ablation Comparison")
    print("="*70)
    
    ablation_files = [
        ('Full LPL (control)', None),  # Would need to find a full LPL run
        ('No Hebbian', 'outputs/activations/activations_ablation_hebb.pt'),
        ('No Predictive', 'outputs/activations/activations_ablation_pred.pt'),
        ('No Stabilization', 'outputs/activations/activations_ablation_stab.pt'),
        ('Shuffled Temporal', 'outputs/activations/activations_ablation_shuffle.pt'),
    ]
    
    results = {}
    
    for name, filepath in ablation_files:
        if filepath is None:
            continue
            
        if not Path(filepath).exists():
            continue
        
        activations, labels = load_activation_file(filepath)
        
        # Sample subset
        if len(activations) > 1000:
            indices = np.random.choice(len(activations), 1000, replace=False)
            activations_sub = activations[indices]
            labels_sub = labels[indices]
        else:
            activations_sub = activations
            labels_sub = labels
        
        acc_mean, acc_std = compute_linear_probe(activations_sub, labels_sub)
        pr = compute_participation_ratio(activations_sub)
        ma = compute_mean_activity(activations_sub)
        
        results[name] = {
            'linear_readout': acc_mean,
            'participation_ratio': pr,
            'mean_activity': ma
        }
    
    # Plot comparison
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(results.keys())
        accs = [results[n]['linear_readout'] for n in names]
        
        ax.bar(names, accs, alpha=0.7)
        ax.set_ylabel('Linear Readout Accuracy')
        ax.set_title('LPL Ablation Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ablation_comparison.png", bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR / 'ablation_comparison.png'}")
        plt.close()
    
    return results

def main():
    """Main function to generate all figures."""
    print("\n" + "="*70)
    print("LPL Paper Figure Reproduction")
    print("="*70 + "\n")
    
    all_results = {}
    
    # Figure 3: Representation Metrics
    try:
        fig3_results = figure_3_representation_metrics()
        all_results['figure_3'] = fig3_results
    except Exception as e:
        print(f"Error generating Figure 3: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 4: Swap Selectivity
    try:
        fig4_results = figure_4_swap_selectivity()
        all_results['figure_4'] = fig4_results
    except Exception as e:
        print(f"Error generating Figure 4: {e}")
        import traceback
        traceback.print_exc()
    
    # Ablation Comparison
    try:
        ablation_results = compute_ablation_comparison()
        all_results['ablation_comparison'] = ablation_results
    except Exception as e:
        print(f"Error generating ablation comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # Save JSON results
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    json_file = OUTPUT_DIR.joinpath("json/paper_reproduction_results.json")
    with open(json_file, 'w') as f:
        json.dump(convert_to_json_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {json_file}")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")
    
    print("\nNOTE: Figure 2 (Single Neuron Selectivity) and Figure 5/6 (Spiking/STDP)")
    print("require additional implementations not currently available in the codebase.")

if __name__ == "__main__":
    main()
