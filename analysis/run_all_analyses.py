"""
Main orchestrator for comprehensive activation analysis.

This script:
1. Discovers activation files from the expected directory structure
2. Runs per-file analysis (metrics, visualizations)
3. Aggregates results across configurations
4. Generates scaling curves and architecture comparisons
5. Creates summary tables
"""

import torch
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.metrics import (
    analyze_single_file,
    compute_before_after_deltas,
    compare_layers,
    normalize_activation_format
)
from analysis.plots import (
    plot_pca,
    plot_tsne,
    plot_umap,
    plot_scaling_curves,
    plot_architecture_comparison,
    plot_distance_vs_pca_dimension,
    HAS_UMAP
)


class ActivationFile:
    """Represents an activation file with metadata."""
    
    def __init__(self, filepath: Path, dataset: str, model: str, steps: str, seed: str, stage: str):
        self.filepath = filepath
        self.dataset = dataset
        self.model = model
        self.steps = steps
        self.seed = seed
        self.stage = stage  # 'before' or 'after'
    
    def parse_steps(self) -> int:
        """Convert steps string to integer (e.g., '1k' -> 1000)."""
        s = self.steps.lower().strip()
        if s.endswith('k'):
            return int(float(s[:-1]) * 1000)
        elif s.endswith('m'):
            return int(float(s[:-1]) * 1000000)
        else:
            return int(s)
    
    def __repr__(self):
        return f"ActivationFile({self.dataset}, {self.model}, {self.steps}, {self.seed}, {self.stage})"


def discover_activation_files(base_dir: Path) -> List[ActivationFile]:
    """
    Discover activation files from directory structure.
    
    Expected structure:
    activations/
      dataset={shapes|mnist|...}/
        model={lpl_1layer|...}/
          steps={500|1k|...}/
            seed={0|1|2}/
              before.pt
              after.pt
    
    Also handles legacy structure:
    outputs/grid_experiments/run_*/activations_before.pt
    """
    files = []
    
    # Try new structure first
    activations_base = base_dir / "activations"
    if activations_base.exists():
        for dataset_dir in activations_base.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name.replace("dataset=", "")
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name.replace("model=", "")
                
                for steps_dir in model_dir.iterdir():
                    if not steps_dir.is_dir():
                        continue
                    steps = steps_dir.name.replace("steps=", "")
                    
                    for seed_dir in steps_dir.iterdir():
                        if not seed_dir.is_dir():
                            continue
                        seed = seed_dir.name.replace("seed=", "")
                        
                        for stage in ['before', 'after']:
                            pt_file = seed_dir / f"{stage}.pt"
                            if pt_file.exists():
                                files.append(ActivationFile(
                                    pt_file, dataset, model, steps, seed, stage
                                ))
    
    # Also check legacy structure (grid experiments)
    grid_experiments = base_dir / "outputs" / "grid_experiments"
    if grid_experiments.exists():
        for exp_dir in grid_experiments.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Parse experiment name to extract metadata
            # Format: run_XXX_dataset_STEPSsteps_architecture_activation_rule
            exp_name = exp_dir.name
            
            # Try to extract information from metadata.json
            metadata_file = exp_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    dataset = metadata.get('dataset', 'unknown')
                    steps = str(metadata.get('steps', 'unknown'))
                    architecture = metadata.get('architecture', 'unknown')
                    
                    # Map architecture to model name
                    if '1layer' in architecture:
                        model = 'lpl_1layer'
                    elif '2layer' in architecture:
                        model = 'lpl_2layer'
                    elif '3layer' in architecture:
                        model = 'lpl_3layer'
                    elif 'conv_mlp' in architecture:
                        model = 'conv_mlp_hybrid'
                    elif 'backprop' in architecture:
                        model = 'baseline'
                    else:
                        model = architecture
                    
                    # Default seed to 0 for legacy experiments
                    seed = '0'
                    
                    # Check for activation files
                    for stage in ['before', 'after']:
                        pt_file = exp_dir / f"activations_{stage}.pt"
                        if pt_file.exists():
                            files.append(ActivationFile(
                                pt_file, dataset, model, steps, seed, stage
                            ))
                except Exception as e:
                    print(f"Warning: Could not parse metadata for {exp_dir}: {e}")
                    continue
    
    return files


def group_files_by_config(files: List[ActivationFile]) -> Dict[Tuple, Dict[str, ActivationFile]]:
    """
    Group files by configuration (dataset, model, steps, seed).
    
    Returns:
        Dictionary mapping (dataset, model, steps, seed) to {'before': file, 'after': file}
    """
    grouped = defaultdict(dict)
    
    for file in files:
        key = (file.dataset, file.model, file.steps, file.seed)
        grouped[key][file.stage] = file
    
    return dict(grouped)


def analyze_single_configuration(
    config_key: Tuple,
    files: Dict[str, ActivationFile],
    output_dir: Path,
    random_seed: int = 42
) -> Dict:
    """
    Analyze a single configuration (all layers, before/after).
    
    Returns:
        Dictionary with all analysis results
    """
    dataset, model, steps, seed = config_key
    
    results = {
        'dataset': dataset,
        'model': model,
        'steps': steps,
        'seed': seed,
        'layers': {}
    }
    
    # Process before and after files
    for stage in ['before', 'after']:
        if stage not in files:
            print(f"Warning: Missing {stage} file for {config_key}")
            continue
        
        file = files[stage]
        
        # Load data to check available layers
        try:
            data = torch.load(file.filepath, map_location='cpu')
            normalized = normalize_activation_format(data)
            
            # Analyze each available layer
            for layer_name in ['layer1', 'layer2']:
                if layer_name not in normalized:
                    continue
                
                layer_key = f"{layer_name}_{stage}"
                
                # Run analysis
                try:
                    analysis = analyze_single_file(
                        str(file.filepath),
                        layer_name=layer_name,
                        stage=stage,
                        random_seed=random_seed
                    )
                    results['layers'][layer_key] = analysis
                except Exception as e:
                    print(f"Error analyzing {file.filepath} {layer_name}: {e}")
                    results['layers'][layer_key] = {'error': str(e)}
        
        except Exception as e:
            print(f"Error loading {file.filepath}: {e}")
            continue
    
    return results


def create_dimensionality_reduction_plots(
    file: ActivationFile,
    layer_name: str,
    output_dir: Path
) -> None:
    """Create PCA, t-SNE, and UMAP plots for a single file/layer."""
    try:
        data = torch.load(file.filepath, map_location='cpu')
        normalized = normalize_activation_format(data)
        
        if layer_name not in normalized:
            return
        
        activations = normalized[layer_name]
        labels = normalized['labels']
        
        # Determine if we have time indices (for shapes dataset)
        time_indices = None
        color_by_time = False
        if file.dataset in ['shapes', 'synthetic_shapes'] and 'time_index' in normalized:
            time_indices = normalized['time_index']
            color_by_time = True
        
        # Create output directory
        layer_output_dir = output_dir / "figures" / file.dataset / file.model / file.steps / file.seed
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        
        # PCA
        pca_path = layer_output_dir / f"pca_{layer_name}_{file.stage}.png"
        plot_pca(
            activations, labels, pca_path,
            title=f"PCA - {file.dataset} {file.model} {file.steps} {layer_name} {file.stage}",
            color_by_time=color_by_time,
            time_indices=time_indices
        )
        
        # t-SNE
        tsne_path = layer_output_dir / f"tsne_{layer_name}_{file.stage}.png"
        plot_tsne(
            activations, labels, tsne_path,
            title=f"t-SNE - {file.dataset} {file.model} {file.steps} {layer_name} {file.stage}",
            color_by_time=color_by_time,
            time_indices=time_indices
        )
        
        # UMAP (if available)
        if HAS_UMAP:
            umap_path = layer_output_dir / f"umap_{layer_name}_{file.stage}.png"
            plot_umap(
                activations, labels, umap_path,
                title=f"UMAP - {file.dataset} {file.model} {file.steps} {layer_name} {file.stage}",
                color_by_time=color_by_time,
                time_indices=time_indices
            )
    
    except Exception as e:
        print(f"Error creating plots for {file.filepath} {layer_name}: {e}")


def aggregate_scaling_laws(
    all_results: List[Dict],
    dataset: str,
    model: str,
    layer_name: str,
    output_dir: Path
) -> None:
    """Create scaling law plots (metric vs training steps)."""
    # Filter results
    relevant = [
        r for r in all_results
        if r['dataset'] == dataset and r['model'] == model
    ]
    
    if not relevant:
        return
    
    # Group by steps
    results_by_steps = defaultdict(list)
    
    for result in relevant:
        steps_str = result['steps']
        try:
            if steps_str.lower().endswith('k'):
                steps = int(float(steps_str[:-1]) * 1000)
            else:
                steps = int(steps_str)
        except:
            continue
        
        # Get layer results
        layer_key = f"{layer_name}_after"
        if layer_key in result['layers']:
            layer_result = result['layers'][layer_key]
            if 'error' not in layer_result:
                results_by_steps[steps].append(layer_result)
    
    if not results_by_steps:
        return
    
    # Create plots directory
    plots_dir = output_dir / "scaling_curves" / dataset / model
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot different metrics
    metrics = ['accuracy_mean', 'ratio', 'std']
    
    for metric in metrics:
        plot_path = plots_dir / f"{metric}_{layer_name}.png"
        plot_scaling_curves(
            results_by_steps,
            metric,
            plot_path,
            title=f"{metric.replace('_', ' ').title()} vs Training Steps - {dataset} {model} {layer_name}",
            ylabel=metric.replace('_', ' ').title()
        )


def aggregate_architecture_comparison(
    all_results: List[Dict],
    dataset: str,
    steps: str,
    layer_name: str,
    output_dir: Path
) -> None:
    """Create architecture comparison plots."""
    # Filter results
    relevant = [
        r for r in all_results
        if r['dataset'] == dataset and r['steps'] == steps
    ]
    
    if not relevant:
        return
    
    # Group by architecture
    architecture_results = defaultdict(list)
    
    for result in relevant:
        model = result['model']
        layer_key = f"{layer_name}_after"
        if layer_key in result['layers']:
            layer_result = result['layers'][layer_key]
            if 'error' not in layer_result:
                architecture_results[model].append(layer_result)
    
    if len(architecture_results) < 2:
        return
    
    # Create plots directory
    plots_dir = output_dir / "architecture_comparisons" / dataset
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot different metrics
    metrics = ['accuracy_mean', 'ratio', 'std']
    
    for metric in metrics:
        plot_path = plots_dir / f"{metric}_{layer_name}_steps_{steps}.png"
        plot_architecture_comparison(
            architecture_results,
            metric,
            plot_path,
            title=f"{metric.replace('_', ' ').title()} Comparison - {dataset} {steps} steps {layer_name}",
            ylabel=metric.replace('_', ' ').title()
        )


def create_summary_tables(all_results: List[Dict], output_dir: Path) -> None:
    """Create summary tables for each dataset."""
    # Group by dataset
    by_dataset = defaultdict(list)
    for result in all_results:
        by_dataset[result['dataset']].append(result)
    
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset, results in by_dataset.items():
        # Create summary DataFrame
        rows = []
        
        for result in results:
            # Get after training results for layer1
            layer_key = 'layer1_after'
            if layer_key in result['layers']:
                layer_result = result['layers'][layer_key]
                if 'error' not in layer_result:
                    rows.append({
                        'Dataset': dataset,
                        'Model': result['model'],
                        'Steps': result['steps'],
                        'Seed': result['seed'],
                        'Accuracy': layer_result.get('accuracy_mean', np.nan),
                        'Accuracy_Std': layer_result.get('accuracy_std', np.nan),
                        'Separation_Ratio': layer_result.get('ratio', np.nan),
                        'Std': layer_result.get('std', np.nan),
                        'Status': layer_result.get('status', 'UNKNOWN')
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            
            # Save CSV
            csv_path = tables_dir / f"{dataset}_summary.csv"
            df.to_csv(csv_path, index=False)
            
            # Create LaTeX table
            latex_path = tables_dir / f"{dataset}_summary.tex"
            with open(latex_path, 'w') as f:
                f.write(df.to_latex(index=False, float_format="%.4f"))
            
            print(f"Created summary table for {dataset}: {csv_path}, {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive activation analysis")
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory containing activations/ or outputs/'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_outputs',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip dimensionality reduction plots (faster)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("LPL ACTIVATION ANALYSIS".center(70))
    print("="*70)
    
    # Discover files
    print("\nDiscovering activation files...")
    files = discover_activation_files(base_dir)
    print(f"Found {len(files)} activation files")
    
    if not files:
        print("No activation files found! Check --base-dir argument.")
        return
    
    # Group by configuration
    grouped = group_files_by_config(files)
    print(f"Found {len(grouped)} unique configurations")
    
    # Analyze each configuration
    print("\nAnalyzing configurations...")
    all_results = []
    
    for i, (config_key, config_files) in enumerate(grouped.items()):
        dataset, model, steps, seed = config_key
        print(f"  [{i+1}/{len(grouped)}] {dataset} {model} {steps} seed={seed}")
        
        try:
            result = analyze_single_configuration(
                config_key, config_files, output_dir, args.random_seed
            )
            all_results.append(result)
            
            # Create dimensionality reduction plots
            if not args.skip_plots:
                for stage in ['before', 'after']:
                    if stage in config_files:
                        file = config_files[stage]
                        # Try both layer1 and layer2
                        for layer_name in ['layer1', 'layer2']:
                            create_dimensionality_reduction_plots(
                                file, layer_name, output_dir
                            )
        
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save raw results
    results_file = output_dir / "all_results.json"
    
    def convert_to_json_serializable(obj):
        """Recursively convert numpy/torch types to JSON-serializable types."""
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return float(obj.item())
            else:
                return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.number)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # Try to convert unknown types
            try:
                if hasattr(obj, 'item'):
                    return float(obj.item())
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return str(obj)
            except:
                return str(obj)
    
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for r in all_results:
            json_r = {k: convert_to_json_serializable(v) for k, v in r.items() if k != 'layers'}
            json_r['layers'] = {}
            for layer_key, layer_data in r['layers'].items():
                json_r['layers'][layer_key] = convert_to_json_serializable(layer_data)
            json_results.append(json_r)
        
        json.dump(json_results, f, indent=2)
    print(f"\nSaved raw results to {results_file}")
    
    # Create aggregations
    print("\nCreating aggregations...")
    
    # Get unique datasets, models, steps
    datasets = set(r['dataset'] for r in all_results)
    models = set(r['model'] for r in all_results)
    steps_list = set(r['steps'] for r in all_results)
    
    # Scaling laws
    print("  Creating scaling law plots...")
    for dataset in datasets:
        for model in models:
            for layer_name in ['layer1', 'layer2']:
                try:
                    aggregate_scaling_laws(all_results, dataset, model, layer_name, output_dir)
                except Exception as e:
                    print(f"    Error creating scaling laws for {dataset} {model} {layer_name}: {e}")
    
    # Architecture comparisons
    print("  Creating architecture comparison plots...")
    for dataset in datasets:
        for steps in steps_list:
            for layer_name in ['layer1', 'layer2']:
                try:
                    aggregate_architecture_comparison(all_results, dataset, steps, layer_name, output_dir)
                except Exception as e:
                    print(f"    Error creating architecture comparison for {dataset} {steps} {layer_name}: {e}")
    
    # Summary tables
    print("  Creating summary tables...")
    try:
        create_summary_tables(all_results, output_dir)
    except Exception as e:
        print(f"    Error creating summary tables: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE".center(70))
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - Raw results: {results_file}")
    print(f"  - Figures: {output_dir / 'figures'}")
    print(f"  - Scaling curves: {output_dir / 'scaling_curves'}")
    print(f"  - Architecture comparisons: {output_dir / 'architecture_comparisons'}")
    print(f"  - Summary tables: {output_dir / 'tables'}")


if __name__ == "__main__":
    main()

