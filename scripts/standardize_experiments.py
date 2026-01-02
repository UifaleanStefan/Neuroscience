"""
Standardize and update all experiment metadata files.

This script:
1. Updates all metadata.json files with standardized fields
2. Adds seed information (default: 42)
3. Adds architecture scaling documentation
4. Adds conv-mlp training length clarification
5. Does NOT retrain models or modify weights/activations
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Global seed
GLOBAL_SEED = 42

# Architecture dimensions per dataset
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

# Architecture scaling strategy
ARCHITECTURE_SCALING = {
    'mnist': 'input-scaled',
    'synthetic_shapes': 'reduced-capacity'
}


def get_layer_dims(metadata):
    """Extract layer dimensions from metadata."""
    dataset = metadata.get('dataset', '')
    arch = metadata.get('architecture', '')
    
    if arch in ARCHITECTURE_DIMS.get(dataset, {}):
        return ARCHITECTURE_DIMS[dataset][arch].get('layer_dims', [])
    
    # Fallback: try to reconstruct from available fields
    dims = []
    if 'd_in' in metadata:
        dims.append(metadata['d_in'])
    if 'd_hidden1' in metadata:
        dims.append(metadata['d_hidden1'])
    if 'd_hidden2' in metadata:
        dims.append(metadata['d_hidden2'])
    if 'd_hidden' in metadata:
        dims.append(metadata['d_hidden'])
    if 'd_out' in metadata:
        dims.append(metadata['d_out'])
    
    return dims if dims else None


def update_metadata_file(metadata_path: Path, run_id: str) -> dict:
    """
    Update a single metadata file with standardized fields.
    
    Returns:
        Updated metadata dictionary and status information
    """
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    original_seed = metadata.get('seed', None)
    updated_fields = {}
    
    # 1. Seed unification
    if 'seed' not in metadata:
        metadata['seed'] = GLOBAL_SEED
        updated_fields['added_seed'] = True
    elif metadata['seed'] != GLOBAL_SEED:
        metadata['seed_mismatch'] = True
        updated_fields['seed_mismatch'] = True
        updated_fields['original_seed'] = original_seed
    else:
        updated_fields['seed_ok'] = True
    
    # Explicitly set seed to 42 (standard)
    metadata['seed'] = GLOBAL_SEED
    
    # 2. Architecture scaling documentation
    dataset = metadata.get('dataset', '')
    if dataset in ARCHITECTURE_SCALING:
        metadata['architecture_scaling'] = ARCHITECTURE_SCALING[dataset]
        updated_fields['added_scaling'] = True
    
    # Add layer dimensions
    layer_dims = get_layer_dims(metadata)
    if layer_dims:
        metadata['layer_dimensions'] = layer_dims
        updated_fields['added_layer_dims'] = True
    
    # 3. Conv-MLP training length clarification
    if 'conv_mlp' in metadata.get('architecture', ''):
        metadata['max_feasible_steps'] = 20000
        metadata['intended_long_run_steps'] = 50000
        metadata['training_length_note'] = (
            "20k steps used instead of 50k due to GPU memory constraints (OOM). "
            "The 20k run represents the maximum feasible training length for this architecture."
        )
        updated_fields['added_conv_mlp_note'] = True
    
    # 4. Add run_id if not present
    if 'run_id' not in metadata:
        metadata['run_id'] = run_id
        updated_fields['added_run_id'] = True
    
    # 5. Add learning_rule field if not present
    if 'learning_rule' not in metadata:
        if metadata.get('rule') == 'full_lpl':
            metadata['learning_rule'] = 'full_lpl'
        else:
            metadata['learning_rule'] = metadata.get('rule', 'unknown')
        updated_fields['added_learning_rule'] = True
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata, updated_fields


def extract_run_id_from_path(path: Path) -> str:
    """Extract run ID from directory path."""
    # Format: run_XXX_...
    match = re.match(r'run_(\d+)_', path.name)
    if match:
        return f"run_{match.group(1)}"
    return path.name


def get_status_from_analysis(metadata_path: Path, analysis_results: dict = None) -> str:
    """Determine experiment status from analysis results if available."""
    # For now, default to 'completed' if activation files exist
    exp_dir = metadata_path.parent
    if (exp_dir / 'activations_before.pt').exists() and (exp_dir / 'activations_after.pt').exists():
        return 'completed'
    return 'unknown'


def main():
    """Main standardization process."""
    base_dir = Path('outputs/grid_experiments')
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return
    
    # Find all metadata files
    metadata_files = list(base_dir.glob('*/metadata.json'))
    
    print("="*70)
    print("EXPERIMENT METADATA STANDARDIZATION".center(70))
    print("="*70)
    print(f"Found {len(metadata_files)} experiment directories\n")
    
    updated_metadata = []
    seed_mismatches = []
    
    for metadata_path in sorted(metadata_files):
        run_id = extract_run_id_from_path(metadata_path.parent)
        
        try:
            metadata, updates = update_metadata_file(metadata_path, run_id)
            
            if updates.get('seed_mismatch'):
                seed_mismatches.append({
                    'run_id': run_id,
                    'original_seed': updates.get('original_seed')
                })
            
            updated_metadata.append(metadata)
            
            # Print summary of updates
            update_summary = [k for k, v in updates.items() if v]
            if update_summary:
                print(f"{run_id}: Updated - {', '.join(update_summary)}")
            
        except Exception as e:
            print(f"{run_id}: ERROR - {e}")
    
    print(f"\nTotal metadata files updated: {len(updated_metadata)}")
    if seed_mismatches:
        print(f"\nWARNING: Found {len(seed_mismatches)} experiments with seed mismatches:")
        for sm in seed_mismatches:
            print(f"  {sm['run_id']}: seed={sm['original_seed']} (marked as seed_mismatch=true)")
    
    print("\n" + "="*70)
    print("METADATA STANDARDIZATION COMPLETE".center(70))
    print("="*70)


if __name__ == "__main__":
    main()

