"""
Generate written summary from analysis results.

Creates interpretable text summaries of the analysis results,
including notes on anomalies and failures.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import numpy as np


def analyze_dataset_results(results: List[Dict], dataset: str) -> str:
    """
    Generate summary text for a specific dataset.
    
    Args:
        results: List of result dictionaries for this dataset
        dataset: Dataset name
        
    Returns:
        Formatted summary string
    """
    lines = [f"## {dataset.upper()} Dataset", ""]
    
    # Group by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r['model']].append(r)
    
    # Statistics across all configurations
    all_accuracies = []
    all_ratios = []
    all_statuses = []
    
    for result in results:
        if 'layer1_after' in result['layers']:
            layer = result['layers']['layer1_after']
            if 'error' not in layer and 'accuracy_mean' in layer:
                all_accuracies.append(layer['accuracy_mean'])
            if 'error' not in layer and 'ratio' in layer:
                all_ratios.append(layer['ratio'])
            if 'error' not in layer and 'status' in layer:
                all_statuses.append(layer['status'])
    
    lines.append(f"### Overall Statistics")
    if all_accuracies:
        lines.append(f"- Mean accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
        lines.append(f"- Max accuracy: {np.max(all_accuracies):.4f}")
        lines.append(f"- Min accuracy: {np.min(all_accuracies):.4f}")
    if all_ratios:
        lines.append(f"- Mean separation ratio: {np.mean(all_ratios):.4f} ± {np.std(all_ratios):.4f}")
    if all_statuses:
        healthy_count = sum(1 for s in all_statuses if s == 'HEALTHY')
        collapsed_count = sum(1 for s in all_statuses if s == 'COLLAPSED')
        saturated_count = sum(1 for s in all_statuses if s == 'SATURATED')
        lines.append(f"- Healthy: {healthy_count}/{len(all_statuses)}, "
                    f"Collapsed: {collapsed_count}/{len(all_statuses)}, "
                    f"Saturated: {saturated_count}/{len(all_statuses)}")
    lines.append("")
    
    # Per-model analysis
    lines.append(f"### Architecture Performance")
    model_summaries = []
    
    for model, model_results in sorted(by_model.items()):
        model_accuracies = []
        model_ratios = []
        model_statuses = []
        
        for result in model_results:
            if 'layer1_after' in result['layers']:
                layer = result['layers']['layer1_after']
                if 'error' not in layer:
                    if 'accuracy_mean' in layer:
                        model_accuracies.append(layer['accuracy_mean'])
                    if 'ratio' in layer:
                        model_ratios.append(layer['ratio'])
                    if 'status' in layer:
                        model_statuses.append(layer['status'])
        
        if model_accuracies:
            avg_acc = np.mean(model_accuracies)
            avg_ratio = np.mean(model_ratios) if model_ratios else 0.0
            healthy_pct = (sum(1 for s in model_statuses if s == 'HEALTHY') / len(model_statuses) * 100) if model_statuses else 0.0
            
            model_summaries.append((model, avg_acc, avg_ratio, healthy_pct))
    
    # Sort by accuracy
    model_summaries.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, avg_acc, avg_ratio, healthy_pct) in enumerate(model_summaries, 1):
        lines.append(f"{i}. **{model}**: Accuracy={avg_acc:.4f}, Ratio={avg_ratio:.4f}, "
                    f"Healthy={healthy_pct:.1f}%")
    lines.append("")
    
    # Training length sensitivity
    lines.append(f"### Training Length Sensitivity")
    
    # Group by steps
    by_steps = defaultdict(list)
    for result in results:
        by_steps[result['steps']].append(result)
    
    step_accuracies = {}
    for steps, step_results in sorted(by_steps.items(), key=lambda x: parse_steps(x[0])):
        accs = []
        for result in step_results:
            if 'layer1_after' in result['layers']:
                layer = result['layers']['layer1_after']
                if 'error' not in layer and 'accuracy_mean' in layer:
                    accs.append(layer['accuracy_mean'])
        if accs:
            step_accuracies[steps] = np.mean(accs)
    
    if step_accuracies:
        best_steps = max(step_accuracies.items(), key=lambda x: x[1])
        lines.append(f"- Best performance at {best_steps[0]} steps: {best_steps[1]:.4f}")
        
        if len(step_accuracies) > 1:
            steps_sorted = sorted(step_accuracies.items(), key=lambda x: parse_steps(x[0]))
            first_acc = steps_sorted[0][1]
            last_acc = steps_sorted[-1][1]
            improvement = last_acc - first_acc
            lines.append(f"- Accuracy improvement from {steps_sorted[0][0]} to {steps_sorted[-1][0]}: "
                        f"{improvement:+.4f}")
    lines.append("")
    
    # Anomalies and failures
    lines.append(f"### Anomalies and Failures")
    anomalies = []
    
    for result in results:
        config_str = f"{result['model']} {result['steps']} seed={result['seed']}"
        
        for layer_key, layer in result['layers'].items():
            if 'error' in layer:
                anomalies.append(f"- ERROR in {config_str} {layer_key}: {layer['error']}")
            elif layer.get('status') == 'COLLAPSED':
                anomalies.append(f"- COLLAPSED in {config_str} {layer_key}: std={layer.get('std', 'N/A'):.6f}")
            elif layer.get('status') == 'SATURATED':
                anomalies.append(f"- SATURATED in {config_str} {layer_key}: {layer.get('pct_saturated', 'N/A'):.1f}% saturated")
            elif layer.get('has_nan', False):
                anomalies.append(f"- NaN detected in {config_str} {layer_key}")
            elif layer.get('accuracy_mean', 0) < 0.1:
                anomalies.append(f"- Very low accuracy in {config_str} {layer_key}: {layer.get('accuracy_mean', 0):.4f}")
    
    if anomalies:
        lines.extend(anomalies)
    else:
        lines.append("- No anomalies detected")
    lines.append("")
    
    return "\n".join(lines)


def parse_steps(steps_str: str) -> int:
    """Parse steps string to integer."""
    s = steps_str.lower().strip()
    if s.endswith('k'):
        return int(float(s[:-1]) * 1000)
    elif s.endswith('m'):
        return int(float(s[:-1]) * 1000000)
    else:
        return int(s)


def main():
    parser = argparse.ArgumentParser(description="Generate written summary from analysis results")
    parser.add_argument(
        '--results-file',
        type=str,
        default='analysis_outputs/all_results.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_outputs/summary.md',
        help='Output markdown file path'
    )
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_file = Path(args.output)
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for result in all_results:
        by_dataset[result['dataset']].append(result)
    
    # Generate summary
    summary_lines = [
        "# LPL Activation Analysis Summary",
        "",
        "This document summarizes the results from comprehensive activation analysis.",
        ""
    ]
    
    # Add dataset summaries
    for dataset in sorted(by_dataset.keys()):
        dataset_summary = analyze_dataset_results(by_dataset[dataset], dataset)
        summary_lines.append(dataset_summary)
        summary_lines.append("")
    
    # Overall conclusions
    summary_lines.append("## Overall Conclusions")
    summary_lines.append("")
    
    # Best performing architecture per dataset
    summary_lines.append("### Best Performing Architecture by Dataset")
    for dataset in sorted(by_dataset.keys()):
        results = by_dataset[dataset]
        best_model = None
        best_acc = -1
        
        by_model = defaultdict(list)
        for r in results:
            by_model[r['model']].append(r)
        
        for model, model_results in by_model.items():
            accs = []
            for result in model_results:
                if 'layer1_after' in result['layers']:
                    layer = result['layers']['layer1_after']
                    if 'error' not in layer and 'accuracy_mean' in layer:
                        accs.append(layer['accuracy_mean'])
            if accs and np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
                best_model = model
        
        if best_model:
            summary_lines.append(f"- **{dataset}**: {best_model} (accuracy: {best_acc:.4f})")
    
    summary_lines.append("")
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("\n".join(summary_lines))
    
    print(f"Summary written to: {output_file}")


if __name__ == "__main__":
    main()

