"""
Validation script to check for experiment inconsistencies.

Checks for:
- Seed inconsistencies
- Missing metadata fields
- Conv-MLP step ambiguity
- Architecture scaling documentation
"""

import json
import re
from pathlib import Path
from collections import defaultdict

GLOBAL_SEED = 42


def extract_run_id_from_path(path: Path) -> str:
    """Extract run ID from directory path."""
    match = re.match(r'run_(\d+)_', path.name)
    if match:
        return f"run_{match.group(1)}"
    return path.name


def validate_metadata(metadata_path: Path) -> dict:
    """Validate a single metadata file."""
    run_id = extract_run_id_from_path(metadata_path.parent)
    
    issues = {
        'run_id': run_id,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        issues['errors'].append(f"Cannot read metadata file: {e}")
        return issues
    
    # Check 1: Seed consistency
    seed = metadata.get('seed')
    if seed is None:
        issues['errors'].append("Missing 'seed' field")
    elif seed != GLOBAL_SEED:
        if metadata.get('seed_mismatch'):
            issues['warnings'].append(f"Seed mismatch detected and marked: seed={seed} (expected {GLOBAL_SEED})")
        else:
            issues['errors'].append(f"Seed mismatch not marked: seed={seed} (expected {GLOBAL_SEED})")
    
    # Check 2: Required fields
    required_fields = ['dataset', 'architecture', 'steps', 'seed']
    for field in required_fields:
        if field not in metadata:
            issues['errors'].append(f"Missing required field: '{field}'")
    
    # Check 3: Architecture scaling documentation
    dataset = metadata.get('dataset', '')
    if 'architecture_scaling' not in metadata:
        issues['warnings'].append("Missing 'architecture_scaling' field")
    elif dataset in ['mnist', 'synthetic_shapes']:
        expected = 'input-scaled' if dataset == 'mnist' else 'reduced-capacity'
        if metadata.get('architecture_scaling') != expected:
            issues['warnings'].append(f"Architecture scaling may be incorrect: got '{metadata.get('architecture_scaling')}', expected '{expected}'")
    
    # Check 4: Layer dimensions
    if 'layer_dimensions' not in metadata:
        issues['warnings'].append("Missing 'layer_dimensions' field")
    
    # Check 5: Conv-MLP specific validation
    if 'conv_mlp' in metadata.get('architecture', ''):
        if 'max_feasible_steps' not in metadata:
            issues['warnings'].append("Conv-MLP experiment missing 'max_feasible_steps' field")
        if 'intended_long_run_steps' not in metadata:
            issues['warnings'].append("Conv-MLP experiment missing 'intended_long_run_steps' field")
        if 'training_length_note' not in metadata:
            issues['warnings'].append("Conv-MLP experiment missing 'training_length_note' field")
        
        steps = metadata.get('steps', 0)
        if steps == 20000:
            if metadata.get('intended_long_run_steps') != 50000:
                issues['warnings'].append(f"Conv-MLP with 20k steps should have intended_long_run_steps=50000")
        elif steps == 50000:
            issues['info'].append("Conv-MLP with 50k steps - check if this completed successfully or had OOM")
    
    # Check 6: Learning rule field
    if 'learning_rule' not in metadata:
        issues['warnings'].append("Missing 'learning_rule' field (inferred from 'rule')")
    
    # Check 7: Run ID consistency
    if 'run_id' not in metadata:
        issues['warnings'].append("Missing 'run_id' field in metadata")
    elif metadata['run_id'] != run_id:
        issues['warnings'].append(f"Run ID mismatch: metadata says '{metadata['run_id']}', expected '{run_id}'")
    
    return issues


def main():
    """Main validation process."""
    base_dir = Path('outputs/grid_experiments')
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return
    
    # Find all metadata files
    metadata_files = list(base_dir.glob('*/metadata.json'))
    
    print("="*70)
    print("EXPERIMENT VALIDATION".center(70))
    print("="*70)
    print(f"Validating {len(metadata_files)} experiments...\n")
    
    all_issues = []
    error_count = 0
    warning_count = 0
    
    for metadata_path in sorted(metadata_files):
        issues = validate_metadata(metadata_path)
        all_issues.append(issues)
        
        if issues['errors']:
            error_count += len(issues['errors'])
        if issues['warnings']:
            warning_count += len(issues['warnings'])
    
    # Print results
    has_errors = False
    has_warnings = False
    
    for issues in all_issues:
        if issues['errors']:
            has_errors = True
            print(f"\n{issues['run_id']} - ERRORS:")
            for error in issues['errors']:
                print(f"  [X] {error}")
        
        if issues['warnings']:
            has_warnings = True
            print(f"\n{issues['run_id']} - WARNINGS:")
            for warning in issues['warnings']:
                print(f"  [!] {warning}")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY".center(70))
    print("="*70)
    print(f"Total experiments validated: {len(metadata_files)}")
    print(f"Total errors: {error_count}")
    print(f"Total warnings: {warning_count}")
    
    if error_count == 0 and warning_count == 0:
        print("\n[OK] All experiments passed validation!")
    elif error_count == 0:
        print(f"\n[OK] No errors found, but {warning_count} warning(s) to review.")
    else:
        print(f"\n[ERROR] Found {error_count} error(s) that need to be fixed.")
    
    # Group issues by type
    issue_types = defaultdict(list)
    for issues in all_issues:
        for error in issues['errors']:
            issue_type = error.split(':')[0] if ':' in error else error
            issue_types[issue_type].append(issues['run_id'])
        for warning in issues['warnings']:
            issue_type = warning.split(':')[0] if ':' in warning else warning
            issue_types[issue_type].append(issues['run_id'])
    
    if issue_types:
        print("\nIssue breakdown:")
        for issue_type, run_ids in sorted(issue_types.items()):
            print(f"  {issue_type}: {len(run_ids)} occurrences ({len(set(run_ids))} unique experiments)")
    
    print("="*70)
    
    # Return exit code
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

