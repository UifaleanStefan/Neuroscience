"""
Generate experiment_registry.json containing all experiment information.

This registry serves as a global index of all experiments.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Architecture dimension mapping (same as in standardize_experiments.py)
ARCHITECTURE_DIMS = {
    'mnist': {
        'mlp_1layer_128': {'d_in': 784, 'd_out': 128, 'layer_dims': [784, 128]},
        'mlp_2layer_128_64': {'d_in': 784, 'd_hidden': 128, 'd_out': 64, 'layer_dims': [784, 128, 64]},
        'mlp_3layer_256_128_64': {'d_in': 784, 'd_hidden1': 256, 'd_hidden2': 128, 'd_out': 64, 'layer_dims': [784, 256, 128, 64]},
        'conv_mlp_hybrid': {'input_channels': 1, 'input_size': 28, 'conv_out_channels': 16, 
                           'mlp_hidden': 128, 'mlp_out': 64, 'layer_dims': ['conv(1->16,28x28)', 12544, 128, 64]},
    },
    'synthetic_shapes': {
        'mlp_1layer_128': {'d_in': 1024, 'd_out': 128, 'layer_dims': [1024, 128]},
        'mlp_2layer_128_64': {'d_in': 1024, 'd_hidden': 128, 'd_out': 64, 'layer_dims': [1024, 128, 64]},
        'mlp_3layer_128_64_32': {'d_in': 1024, 'd_hidden1': 128, 'd_hidden2': 64, 'd_out': 32, 'layer_dims': [1024, 128, 64, 32]},
        'conv_mlp_hybrid': {'input_channels': 1, 'input_size': 32, 'conv_out_channels': 8, 
                           'layer_dims': ['conv(1->8,32x32)', 8192, 32]},
    }
}


def extract_run_id_from_path(path: Path) -> str:
    """Extract run ID from directory path."""
    match = re.match(r'run_(\d+)_', path.name)
    if match:
        return f"run_{match.group(1)}"
    return path.name


def get_experiment_status(exp_dir: Path) -> str:
    """Determine experiment status."""
    has_before = (exp_dir / 'activations_before.pt').exists()
    has_after = (exp_dir / 'activations_after.pt').exists()
    
    if has_before and has_after:
        return 'completed'
    elif has_before and not has_after:
        return 'incomplete'
    elif not has_before:
        return 'not_started'
    return 'unknown'


def get_layer_dims_from_metadata(metadata: dict) -> list:
    """Extract layer dimensions from metadata."""
    dataset = metadata.get('dataset', '')
    arch = metadata.get('architecture', '')
    
    if arch in ARCHITECTURE_DIMS.get(dataset, {}):
        return ARCHITECTURE_DIMS[dataset][arch].get('layer_dims', [])
    
    # Fallback to metadata fields
    if 'layer_dimensions' in metadata:
        return metadata['layer_dimensions']
    
    dims = []
    if 'd_in' in metadata:
        dims.append(metadata['d_in'])
    if 'd_hidden1' in metadata:
        dims.append(metadata['d_hidden1'])
    if 'd_hidden' in metadata:
        dims.append(metadata['d_hidden'])
    if 'd_hidden2' in metadata:
        dims.append(metadata['d_hidden2'])
    if 'd_out' in metadata:
        dims.append(metadata['d_out'])
    
    return dims if dims else None


def extract_notes_from_analysis(exp_dir: Path, metadata: dict) -> list:
    """Extract notes about experiment status (collapse, OOM, etc.)."""
    notes = []
    
    # Check for seed mismatch
    if metadata.get('seed_mismatch'):
        notes.append(f"Seed mismatch: trained with seed={metadata.get('original_seed', 'unknown')} but metadata says {metadata.get('seed')}")
    
    # Check for conv-mlp step limitation
    if 'conv_mlp' in metadata.get('architecture', ''):
        if metadata.get('steps', 0) == 20000:
            notes.append("Limited to 20k steps due to GPU OOM (intended: 50k)")
    
    # Check if analysis results exist to detect collapses
    # This would require loading analysis results, so we'll skip for now
    # and let the validation script handle it
    
    return notes


def map_architecture_to_model(architecture: str) -> str:
    """Map architecture string to model name."""
    if '1layer' in architecture:
        return 'lpl_1layer'
    elif '2layer' in architecture:
        return 'lpl_2layer'
    elif '3layer' in architecture:
        return 'lpl_3layer'
    elif 'conv_mlp' in architecture:
        return 'conv_mlp_hybrid'
    elif 'backprop' in architecture:
        return 'baseline_backprop'
    return architecture


def main():
    """Generate experiment registry."""
    base_dir = Path('outputs/grid_experiments')
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return
    
    # Load analysis results if available to get collapse information
    analysis_results = {}
    analysis_file = Path('analysis_outputs/all_results.json')
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
            # Index by (dataset, model, steps, seed)
            for result in analysis_data:
                key = (result['dataset'], result['model'], str(result['steps']), str(result['seed']))
                analysis_results[key] = result
    
    # Find all metadata files
    metadata_files = list(base_dir.glob('*/metadata.json'))
    
    registry = {
        'generated_at': str(Path.cwd()),
        'total_experiments': len(metadata_files),
        'global_seed': 42,
        'experiments': []
    }
    
    print("="*70)
    print("GENERATING EXPERIMENT REGISTRY".center(70))
    print("="*70)
    print(f"Processing {len(metadata_files)} experiments...\n")
    
    for metadata_path in sorted(metadata_files):
        exp_dir = metadata_path.parent
        run_id = extract_run_id_from_path(exp_dir)
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract information
            dataset = metadata.get('dataset', 'unknown')
            architecture = metadata.get('architecture', 'unknown')
            steps = metadata.get('steps', 0)
            seed = metadata.get('seed', 42)
            model = map_architecture_to_model(architecture)
            
            # Get layer dimensions
            layer_dims = get_layer_dims_from_metadata(metadata)
            
            # Get status
            status = get_experiment_status(exp_dir)
            
            # Get notes
            notes = extract_notes_from_analysis(exp_dir, metadata)
            
            # Check analysis results for collapse status and other issues
            # Try matching with both seed=42 and seed=0 (analysis results might use seed 0)
            analysis_key_42 = (dataset, model, str(steps), '42')
            analysis_key_0 = (dataset, model, str(steps), '0')
            analysis = None
            
            if analysis_key_42 in analysis_results:
                analysis = analysis_results[analysis_key_42]
            elif analysis_key_0 in analysis_results:
                analysis = analysis_results[analysis_key_0]
                if seed == 42:
                    notes.append(f"Analysis results found with seed=0, but metadata says seed=42 (possible seed mismatch)")
            
            if analysis:
                # Check for collapse in all layers
                for layer_key in ['layer1_after', 'layer2_after', 'layer3_after']:
                    if layer_key in analysis.get('layers', {}):
                        layer_result = analysis['layers'][layer_key]
                        if layer_result.get('status') == 'COLLAPSED':
                            layer_name = layer_key.replace('_after', '')
                            std_val = layer_result.get('std', 'N/A')
                            if std_val != 'N/A' and not isinstance(std_val, str):
                                notes.append(f"{layer_name} COLLAPSED: std={std_val:.6f}")
                            else:
                                notes.append(f"{layer_name} COLLAPSED")
                        elif layer_result.get('status') == 'SATURATED':
                            layer_name = layer_key.replace('_after', '')
                            pct_sat = layer_result.get('pct_saturated', 'N/A')
                            if pct_sat != 'N/A' and not isinstance(pct_sat, str):
                                notes.append(f"{layer_name} SATURATED: {pct_sat:.1f}%")
                            else:
                                notes.append(f"{layer_name} SATURATED")
                        if layer_result.get('has_nan'):
                            layer_name = layer_key.replace('_after', '')
                            notes.append(f"{layer_name} has NaN values")
                        if 'error' in layer_result:
                            layer_name = layer_key.replace('_after', '')
                            notes.append(f"{layer_name} ERROR: {layer_result['error']}")
            
            # Learning rule
            learning_rule = metadata.get('learning_rule', metadata.get('rule', 'full_lpl'))
            
            # Build registry entry
            entry = {
                'run_id': run_id,
                'dataset': dataset,
                'architecture': architecture,
                'model': model,
                'layer_dims': layer_dims,
                'steps': steps,
                'seed': seed,
                'learning_rule': learning_rule,
                'status': status,
                'notes': notes if notes else []
            }
            
            # Add conv-mlp specific fields if applicable
            if 'conv_mlp' in architecture:
                entry['max_feasible_steps'] = metadata.get('max_feasible_steps', 20000)
                entry['intended_long_run_steps'] = metadata.get('intended_long_run_steps', 50000)
            
            registry['experiments'].append(entry)
            
        except Exception as e:
            print(f"ERROR processing {run_id}: {e}")
            continue
    
    # Sort experiments by run_id
    registry['experiments'].sort(key=lambda x: int(re.findall(r'\d+', x['run_id'])[0]))
    
    # Save registry
    registry_path = Path('experiment_registry.json')
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\nRegistry saved to: {registry_path}")
    print(f"Total experiments: {len(registry['experiments'])}")
    
    # Print summary by dataset
    by_dataset = defaultdict(list)
    for exp in registry['experiments']:
        by_dataset[exp['dataset']].append(exp)
    
    print("\nSummary by dataset:")
    for dataset, exps in sorted(by_dataset.items()):
        print(f"  {dataset}: {len(exps)} experiments")
    
    print("\n" + "="*70)
    print("REGISTRY GENERATION COMPLETE".center(70))
    print("="*70)


if __name__ == "__main__":
    main()

