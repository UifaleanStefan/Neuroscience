"""
Plotting utilities for activation analysis.

Implements:
- Dimensionality reduction visualizations (PCA, t-SNE, UMAP)
- Scaling law plots
- Architecture comparison plots
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        warnings.warn("UMAP not available. Install with: pip install umap-learn")
except ImportError:
    raise ImportError("scikit-learn required for dimensionality reduction")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_pca(
    activations: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    title: str = "PCA Visualization",
    color_by_time: bool = False,
    time_indices: Optional[torch.Tensor] = None
) -> None:
    """
    Plot PCA visualization (first 2 components).
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        output_path: Path to save the figure
        title: Plot title
        color_by_time: If True, color by time index instead of class
        time_indices: Optional time indices for coloring
    """
    X = activations.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    # Fit PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_by_time and time_indices is not None:
        # Color by time index
        t = time_indices.detach().cpu().numpy()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Time Index')
    else:
        # Color by class label
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=[colors[i]], label=f'Class {int(label)}',
                alpha=0.6, s=20
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_tsne(
    activations: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    title: str = "t-SNE Visualization",
    perplexity: int = 30,
    color_by_time: bool = False,
    time_indices: Optional[torch.Tensor] = None
) -> None:
    """
    Plot t-SNE visualization.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        output_path: Path to save the figure
        title: Plot title
        perplexity: t-SNE perplexity parameter
        color_by_time: If True, color by time index instead of class
        time_indices: Optional time indices for coloring
    """
    X = activations.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    # Limit sample size if too large (t-SNE is slow)
    max_samples = 5000
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
        if time_indices is not None:
            time_indices = time_indices[indices]
    
    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_by_time and time_indices is not None:
        t = time_indices.detach().cpu().numpy()
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Time Index')
    else:
        # Color by class label
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                c=[colors[i]], label=f'Class {int(label)}',
                alpha=0.6, s=20
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_umap(
    activations: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    title: str = "UMAP Visualization",
    n_neighbors: int = 15,
    color_by_time: bool = False,
    time_indices: Optional[torch.Tensor] = None
) -> None:
    """
    Plot UMAP visualization.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        output_path: Path to save the figure
        title: Plot title
        n_neighbors: UMAP n_neighbors parameter
        color_by_time: If True, color by time index instead of class
        time_indices: Optional time indices for coloring
    """
    if not HAS_UMAP:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    X = activations.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    # Fit UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_by_time and time_indices is not None:
        t = time_indices.detach().cpu().numpy()
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=t, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Time Index')
    else:
        # Color by class label
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(
                X_umap[mask, 0], X_umap[mask, 1],
                c=[colors[i]], label=f'Class {int(label)}',
                alpha=0.6, s=20
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_scaling_curves(
    results_by_steps: Dict[int, List[Dict]],
    metric: str,
    output_path: Path,
    title: str = "Scaling Law",
    ylabel: str = None
) -> None:
    """
    Plot scaling curves (metric vs training steps).
    
    Args:
        results_by_steps: Dictionary mapping step counts to lists of result dicts
        metric: Metric name to plot (e.g., 'accuracy_mean', 'ratio')
        output_path: Path to save the figure
        title: Plot title
        ylabel: Y-axis label (defaults to metric name)
    """
    steps = sorted(results_by_steps.keys())
    means = []
    stds = []
    
    for step in steps:
        results = results_by_steps[step]
        values = [r.get(metric, np.nan) for r in results if metric in r]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            means.append(np.mean(values))
            stds.append(np.std(values) if len(values) > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(0.0)
    
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean with error bars
    ax.plot(steps, means, 'o-', linewidth=2, markersize=8, label='Mean')
    ax.fill_between(steps, means - stds, means + stds, alpha=0.3, label='±1 std')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel(ylabel or metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_architecture_comparison(
    architecture_results: Dict[str, List[Dict]],
    metric: str,
    output_path: Path,
    title: str = "Architecture Comparison",
    ylabel: str = None
) -> None:
    """
    Plot bar chart comparing architectures.
    
    Args:
        architecture_results: Dictionary mapping architecture names to lists of result dicts
        metric: Metric name to compare (e.g., 'accuracy_mean', 'ratio')
        output_path: Path to save the figure
        title: Plot title
        ylabel: Y-axis label (defaults to metric name)
    """
    architectures = list(architecture_results.keys())
    means = []
    stds = []
    
    for arch in architectures:
        results = architecture_results[arch]
        values = [r.get(metric, np.nan) for r in results if metric in r]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            means.append(np.mean(values))
            stds.append(np.std(values) if len(values) > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(0.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(architectures))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Architecture')
    ax.set_ylabel(ylabel or metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(architectures, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        if not np.isnan(mean):
            ax.text(i, mean + std + 0.01 * max(means), f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_distance_vs_pca_dimension(
    activations: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    title: str = "Distance vs PCA Dimension",
    max_dims: int = 50
) -> None:
    """
    Plot intra/inter-class distance as a function of PCA dimension.
    
    Args:
        activations: Tensor of shape (N, D) with activations
        labels: Tensor of shape (N,) with class labels
        output_path: Path to save the figure
        title: Plot title
        max_dims: Maximum number of PCA dimensions to use
    """
    from .metrics import compute_class_distances
    
    X = activations.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    # Limit dimensions
    n_dims = min(X.shape[1], max_dims)
    
    # Fit PCA
    pca = PCA(n_components=n_dims, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Compute distances for increasing number of dimensions
    dims = []
    intra_means = []
    inter_means = []
    ratios = []
    
    for d in range(2, n_dims + 1, max(1, n_dims // 20)):
        X_d = torch.tensor(X_pca[:, :d])
        y_t = torch.tensor(y)
        
        dist_results = compute_class_distances(X_d, y_t)
        
        dims.append(d)
        intra_means.append(dist_results['intra_mean'])
        inter_means.append(dist_results['inter_mean'])
        ratios.append(dist_results['ratio'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Intra-class distance
    axes[0].plot(dims, intra_means, 'o-', linewidth=2)
    axes[0].set_xlabel('PCA Dimensions')
    axes[0].set_ylabel('Intra-Class Distance')
    axes[0].set_title('Intra-Class Distance vs PCA Dimensions')
    axes[0].grid(True, alpha=0.3)
    
    # Inter-class distance
    axes[1].plot(dims, inter_means, 'o-', linewidth=2, color='orange')
    axes[1].set_xlabel('PCA Dimensions')
    axes[1].set_ylabel('Inter-Class Distance')
    axes[1].set_title('Inter-Class Distance vs PCA Dimensions')
    axes[1].grid(True, alpha=0.3)
    
    # Ratio
    axes[2].plot(dims, ratios, 'o-', linewidth=2, color='green')
    axes[2].set_xlabel('PCA Dimensions')
    axes[2].set_ylabel('Separation Ratio (Inter/Intra)')
    axes[2].set_title('Separation Ratio vs PCA Dimensions')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

