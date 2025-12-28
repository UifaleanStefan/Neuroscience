"""
Example usage of the analysis framework.

This script demonstrates how to use the analysis tools programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.metrics import (
    analyze_single_file,
    linear_probe,
    compute_class_distances,
    compute_variance_diagnostics
)
from analysis.plots import plot_pca, plot_tsne

import torch


def example_analyze_single_file():
    """Example: Analyze a single activation file."""
    filepath = "outputs/grid_experiments/run_001_*/activations_after.pt"
    
    # Analyze layer1
    results = analyze_single_file(
        filepath,
        layer_name='layer1',
        stage='after',
        random_seed=42
    )
    
    print("Analysis Results:")
    print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  Separation Ratio: {results['ratio']:.4f}")
    print(f"  Status: {results['status']}")
    print(f"  Std: {results['std']:.4f}")


def example_custom_metrics():
    """Example: Compute metrics on custom activations."""
    # Create dummy data
    N, D = 1000, 128
    activations = torch.randn(N, D)
    labels = torch.randint(0, 10, (N,))
    
    # Linear probe
    probe_results = linear_probe(activations, labels, n_splits=3)
    print(f"Linear Probe Accuracy: {probe_results['accuracy_mean']:.4f} ± {probe_results['accuracy_std']:.4f}")
    
    # Distance analysis
    distance_results = compute_class_distances(activations, labels)
    print(f"Separation Ratio: {distance_results['ratio']:.4f}")
    
    # Variance diagnostics
    variance_results = compute_variance_diagnostics(activations)
    print(f"Status: {variance_results['status']}")
    print(f"Std: {variance_results['std']:.4f}")


def example_plots():
    """Example: Create dimensionality reduction plots."""
    # Create dummy data
    N, D = 500, 64
    activations = torch.randn(N, D)
    labels = torch.randint(0, 5, (N,))
    
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # PCA
    plot_pca(
        activations, labels,
        output_dir / "pca_example.png",
        title="Example PCA Visualization"
    )
    print(f"Saved PCA plot to {output_dir / 'pca_example.png'}")
    
    # t-SNE
    plot_tsne(
        activations, labels,
        output_dir / "tsne_example.png",
        title="Example t-SNE Visualization"
    )
    print(f"Saved t-SNE plot to {output_dir / 'tsne_example.png'}")


if __name__ == "__main__":
    print("Analysis Framework Examples")
    print("=" * 50)
    
    # Uncomment to run examples:
    # example_analyze_single_file()
    # example_custom_metrics()
    # example_plots()
    
    print("\nRun the main analysis script:")
    print("  python -m analysis.run_all_analyses --base-dir . --output-dir analysis_outputs")

