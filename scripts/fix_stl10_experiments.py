"""Script to fix STL-10 experiment files: replace train=True with split='train'."""

import re
from pathlib import Path

# STL-10 experiment files (run_070 through run_086)
STL10_EXPERIMENTS = list(range(70, 87))


def fix_stl10_experiment(file_path):
    """Fix STL-10 experiment file: replace train=True with split='train'."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace train=True with split='train'
    content = re.sub(
        r'STL10TemporalPairDataset\(\s*train=True,',
        "STL10TemporalPairDataset(\n        split='train',",
        content
    )
    
    content = re.sub(
        r'create_stl10_temporal_pair_dataset\(\s*train=True,',
        "create_stl10_temporal_pair_dataset(\n        split='train',",
        content
    )
    
    # Also fix any remaining train=True patterns
    content = re.sub(
        r'train=True',
        "split='train'",
        content
    )
    
    # Fix comment in export function
    content = re.sub(
        r'# Image should be 3D \(3, 32, 32\) from STL-10',
        '# Image should be 3D (3, 96, 96) from STL-10',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    """Fix all STL-10 experiment files."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    experiments_dir = project_root / 'experiments'
    
    print("Fixing STL-10 experiment files...")
    print("=" * 70)
    
    for run_num in STL10_EXPERIMENTS:
        file_path = experiments_dir / f'run_grid_exp_{run_num:03d}.py'
        if file_path.exists():
            fix_stl10_experiment(file_path)
            print(f"Fixed: run_grid_exp_{run_num:03d}.py")
    
    print("=" * 70)
    print("All fixes applied!")


if __name__ == "__main__":
    main()


