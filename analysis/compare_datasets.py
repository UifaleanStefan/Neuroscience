"""
Compare results between MNIST and synthetic_shapes datasets.

Creates comparison plots and analysis showing how the two datasets
differ in terms of accuracy, separation ratios, and other metrics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def parse_steps(steps_str: str) -> int:
    """Parse steps string to integer."""
    s = steps_str.lower().strip()
    if s.endswith('k'):
        return int(float(s[:-1]) * 1000)
    elif s.endswith('m'):
        return int(float(s[:-1]) * 1000000)
    else:
        return int(s)


def load_results(results_file: Path) -> List[Dict]:
    """Load analysis results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def filter_results(results: List[Dict], dataset: str) -> List[Dict]:
    """Filter results for a specific dataset."""
    return [r for r in results if r['dataset'] == dataset]


def extract_metric(results: List[Dict], metric: str, layer: str = 'layer1_after') -> Dict[Tuple, float]:
    """
    Extract a metric from results, indexed by (model, steps).
    
    Returns:
        Dictionary mapping (model, steps) -> metric value
    """
    metric_dict = {}
    for result in results:
        if layer in result['layers']:
            layer_data = result['layers'][layer]
            if 'error' not in layer_data and metric in layer_data:
                key = (result['model'], result['steps'])
                metric_dict[key] = layer_data[metric]
    return metric_dict


def plot_metric_comparison(
    mnist_results: List[Dict],
    shapes_results: List[Dict],
    metric: str,
    metric_label: str,
    output_path: Path,
    layer: str = 'layer1_after'
):
    """Create side-by-side comparison plot for a metric."""
    # Extract metrics
    mnist_metrics = extract_metric(mnist_results, metric, layer)
    shapes_metrics = extract_metric(shapes_results, metric, layer)
    
    # Find common configurations
    common_configs = set(mnist_metrics.keys()) & set(shapes_metrics.keys())
    
    if not common_configs:
        print(f"Warning: No common configurations for {metric}")
        return
    
    # Prepare data for plotting
    models = sorted(set(m[0] for m in common_configs))
    steps_list = sorted(set(m[1] for m in common_configs), key=parse_steps)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: By model (averaged across steps)
    model_mnist = defaultdict(list)
    model_shapes = defaultdict(list)
    
    for (model, steps), value in mnist_metrics.items():
        if (model, steps) in common_configs:
            model_mnist[model].append(value)
    
    for (model, steps), value in shapes_metrics.items():
        if (model, steps) in common_configs:
            model_shapes[model].append(value)
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    mnist_means = [np.mean(model_mnist.get(m, [np.nan])) for m in models]
    mnist_stds = [np.std(model_mnist.get(m, [np.nan])) for m in models]
    shapes_means = [np.mean(model_shapes.get(m, [np.nan])) for m in models]
    shapes_stds = [np.std(model_shapes.get(m, [np.nan])) for m in models]
    
    axes[0].bar(x_pos - width/2, mnist_means, width, yerr=mnist_stds,
                label='MNIST', alpha=0.8, capsize=5, color='steelblue')
    axes[0].bar(x_pos + width/2, shapes_means, width, yerr=shapes_stds,
                label='Synthetic Shapes', alpha=0.8, capsize=5, color='coral')
    
    axes[0].set_xlabel('Model Architecture')
    axes[0].set_ylabel(metric_label)
    axes[0].set_title(f'{metric_label} by Architecture (averaged across steps)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: By steps (averaged across models)
    steps_mnist = defaultdict(list)
    steps_shapes = defaultdict(list)
    
    for (model, steps), value in mnist_metrics.items():
        if (model, steps) in common_configs:
            steps_mnist[steps].append(value)
    
    for (model, steps), value in shapes_metrics.items():
        if (model, steps) in common_configs:
            steps_shapes[steps].append(value)
    
    steps_sorted = sorted(steps_mnist.keys(), key=parse_steps)
    x_pos_steps = np.arange(len(steps_sorted))
    
    mnist_step_means = [np.mean(steps_mnist.get(s, [np.nan])) for s in steps_sorted]
    mnist_step_stds = [np.std(steps_mnist.get(s, [np.nan])) for s in steps_sorted]
    shapes_step_means = [np.mean(steps_shapes.get(s, [np.nan])) for s in steps_sorted]
    shapes_step_stds = [np.std(steps_shapes.get(s, [np.nan])) for s in steps_sorted]
    
    axes[1].bar(x_pos_steps - width/2, mnist_step_means, width, yerr=mnist_step_stds,
                label='MNIST', alpha=0.8, capsize=5, color='steelblue')
    axes[1].bar(x_pos_steps + width/2, shapes_step_means, width, yerr=shapes_step_stds,
                label='Synthetic Shapes', alpha=0.8, capsize=5, color='coral')
    
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel(metric_label)
    axes[1].set_title(f'{metric_label} by Training Steps (averaged across models)')
    axes[1].set_xticks(x_pos_steps)
    axes[1].set_xticklabels(steps_sorted, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def plot_scaling_comparison(
    mnist_results: List[Dict],
    shapes_results: List[Dict],
    metric: str,
    metric_label: str,
    output_path: Path,
    model: str = None,
    layer: str = 'layer1_after'
):
    """Create scaling curve comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset_name, results in [('MNIST', mnist_results), ('Synthetic Shapes', shapes_results)]:
        if model:
            results = [r for r in results if r['model'] == model]
        
        # Group by steps
        by_steps = defaultdict(list)
        for result in results:
            if layer in result['layers']:
                layer_data = result['layers'][layer]
                if 'error' not in layer_data and metric in layer_data:
                    by_steps[result['steps']].append(layer_data[metric])
        
        if not by_steps:
            continue
        
        steps_sorted = sorted(by_steps.keys(), key=parse_steps)
        means = [np.mean(by_steps[s]) for s in steps_sorted]
        stds = [np.std(by_steps[s]) for s in steps_sorted]
        steps_numeric = [parse_steps(s) for s in steps_sorted]
        
        ax.errorbar(steps_numeric, means, yerr=stds, marker='o', label=dataset_name,
                   linewidth=2, markersize=8, capsize=5, capthick=2)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    title = f'{metric_label} vs Training Steps'
    if model:
        title += f' ({model})'
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved scaling comparison: {output_path}")


def create_comparison_table(
    mnist_results: List[Dict],
    shapes_results: List[Dict],
    output_path: Path,
    layer: str = 'layer1_after'
) -> pd.DataFrame:
    """Create a comparison table."""
    rows = []
    
    # Get all unique configurations
    all_configs = set()
    for result in mnist_results + shapes_results:
        if layer in result['layers']:
            all_configs.add((result['dataset'], result['model'], result['steps']))
    
    for dataset, model, steps in sorted(all_configs):
        if dataset not in ['mnist', 'synthetic_shapes']:
            continue
        
        results_list = mnist_results if dataset == 'mnist' else shapes_results
        result = next((r for r in results_list 
                      if r['dataset'] == dataset and r['model'] == model and r['steps'] == steps), None)
        
        if result and layer in result['layers']:
            layer_data = result['layers'][layer]
            if 'error' not in layer_data:
                rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Steps': steps,
                    'Accuracy': layer_data.get('accuracy_mean', np.nan),
                    'Separation_Ratio': layer_data.get('ratio', np.nan),
                    'Std': layer_data.get('std', np.nan),
                    'Status': layer_data.get('status', 'UNKNOWN')
                })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"Saved comparison table: {output_path}")
    
    return df


def generate_comparison_summary(
    mnist_results: List[Dict],
    shapes_results: List[Dict],
    output_path: Path,
    layer: str = 'layer1_after'
) -> str:
    """Generate a text summary comparing the two datasets."""
    lines = [
        "# MNIST vs Synthetic Shapes Comparison",
        "",
        "This document compares the performance of LPL models on MNIST and synthetic_shapes datasets.",
        ""
    ]
    
    # Overall statistics
    lines.append("## Overall Statistics")
    lines.append("")
    
    for dataset_name, results in [('MNIST', mnist_results), ('Synthetic Shapes', shapes_results)]:
        lines.append(f"### {dataset_name}")
        
        accuracies = []
        ratios = []
        statuses = []
        
        for result in results:
            if layer in result['layers']:
                layer_data = result['layers'][layer]
                if 'error' not in layer_data:
                    if 'accuracy_mean' in layer_data:
                        accuracies.append(layer_data['accuracy_mean'])
                    if 'ratio' in layer_data:
                        ratios.append(layer_data['ratio'])
                    if 'status' in layer_data:
                        statuses.append(layer_data['status'])
        
        if accuracies:
            lines.append(f"- Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            lines.append(f"  - Range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
        
        if ratios:
            lines.append(f"- Mean Separation Ratio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
            lines.append(f"  - Range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
        
        if statuses:
            healthy = sum(1 for s in statuses if s == 'HEALTHY')
            collapsed = sum(1 for s in statuses if s == 'COLLAPSED')
            saturated = sum(1 for s in statuses if s == 'SATURATED')
            lines.append(f"- Status: {healthy} HEALTHY, {collapsed} COLLAPSED, {saturated} SATURATED")
        
        lines.append("")
    
    # Key differences
    lines.append("## Key Differences")
    lines.append("")
    
    mnist_accs = []
    shapes_accs = []
    mnist_ratios = []
    shapes_ratios = []
    
    for result in mnist_results:
        if layer in result['layers']:
            layer_data = result['layers'][layer]
            if 'error' not in layer_data:
                if 'accuracy_mean' in layer_data:
                    mnist_accs.append(layer_data['accuracy_mean'])
                if 'ratio' in layer_data:
                    mnist_ratios.append(layer_data['ratio'])
    
    for result in shapes_results:
        if layer in result['layers']:
            layer_data = result['layers'][layer]
            if 'error' not in layer_data:
                if 'accuracy_mean' in layer_data:
                    shapes_accs.append(layer_data['accuracy_mean'])
                if 'ratio' in layer_data:
                    shapes_ratios.append(layer_data['ratio'])
    
    if mnist_accs and shapes_accs:
        acc_diff = np.mean(shapes_accs) - np.mean(mnist_accs)
        lines.append(f"- **Accuracy Gap**: Synthetic Shapes is {acc_diff:.4f} higher on average")
        lines.append(f"  - MNIST: {np.mean(mnist_accs):.4f}, Shapes: {np.mean(shapes_accs):.4f}")
    
    if mnist_ratios and shapes_ratios:
        ratio_diff = np.mean(shapes_ratios) - np.mean(mnist_ratios)
        lines.append(f"- **Separation Ratio Gap**: Synthetic Shapes is {ratio_diff:.4f} higher on average")
        lines.append(f"  - MNIST: {np.mean(mnist_ratios):.4f}, Shapes: {np.mean(shapes_ratios):.4f}")
    
    lines.append("")
    
    # Per-model comparison
    lines.append("## Per-Model Comparison")
    lines.append("")
    
    mnist_by_model = defaultdict(list)
    shapes_by_model = defaultdict(list)
    
    for result in mnist_results:
        if layer in result['layers']:
            layer_data = result['layers'][layer]
            if 'error' not in layer_data and 'accuracy_mean' in layer_data:
                mnist_by_model[result['model']].append(layer_data['accuracy_mean'])
    
    for result in shapes_results:
        if layer in result['layers']:
            layer_data = result['layers'][layer]
            if 'error' not in layer_data and 'accuracy_mean' in layer_data:
                shapes_by_model[result['model']].append(layer_data['accuracy_mean'])
    
    common_models = set(mnist_by_model.keys()) & set(shapes_by_model.keys())
    
    for model in sorted(common_models):
        mnist_mean = np.mean(mnist_by_model[model])
        shapes_mean = np.mean(shapes_by_model[model])
        diff = shapes_mean - mnist_mean
        lines.append(f"- **{model}**:")
        lines.append(f"  - MNIST: {mnist_mean:.4f}, Shapes: {shapes_mean:.4f}, Difference: {diff:+.4f}")
    
    lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Saved comparison summary: {output_path}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare MNIST and synthetic_shapes results")
    parser.add_argument(
        '--results-file',
        type=str,
        default='analysis_outputs/all_results.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_outputs/dataset_comparison',
        help='Output directory for comparison results'
    )
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    # Load results
    print("Loading results...")
    all_results = load_results(results_file)
    
    # Filter by dataset
    mnist_results = filter_results(all_results, 'mnist')
    shapes_results = filter_results(all_results, 'synthetic_shapes')
    
    print(f"Found {len(mnist_results)} MNIST configurations")
    print(f"Found {len(shapes_results)} synthetic_shapes configurations")
    
    if not mnist_results or not shapes_results:
        print("Error: Need results from both datasets")
        return
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    
    metrics = [
        ('accuracy_mean', 'Accuracy'),
        ('ratio', 'Separation Ratio'),
        ('std', 'Standard Deviation')
    ]
    
    for metric, label in metrics:
        plot_path = output_dir / f"{metric}_comparison.png"
        plot_metric_comparison(mnist_results, shapes_results, metric, label, plot_path)
    
    # Create scaling comparisons for each model
    print("\nCreating scaling curve comparisons...")
    models = set(r['model'] for r in mnist_results) & set(r['model'] for r in shapes_results)
    
    for model in models:
        for metric, label in metrics:
            plot_path = output_dir / f"scaling_{metric}_{model.replace('/', '_')}.png"
            plot_scaling_comparison(mnist_results, shapes_results, metric, label, plot_path, model=model)
    
    # Create comparison table
    print("\nCreating comparison table...")
    table_path = output_dir / "comparison_table.csv"
    create_comparison_table(mnist_results, shapes_results, table_path)
    
    # Generate summary
    print("\nGenerating comparison summary...")
    summary_path = output_dir / "comparison_summary.md"
    generate_comparison_summary(mnist_results, shapes_results, summary_path)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE".center(70))
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - Comparison plots: {output_dir}")
    print(f"  - Comparison table: {table_path}")
    print(f"  - Comparison summary: {summary_path}")


if __name__ == "__main__":
    main()


