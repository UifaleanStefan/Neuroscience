"""Script to create 17 STL-10 experiment files from CIFAR-10 templates."""

import os
import re
from pathlib import Path

# Mapping: STL-10 run number -> CIFAR-10 source run number
# STL-10: run_070 to run_086 (17 files)
# CIFAR-10: run_053 to run_069 (17 files)
RUN_MAPPING = {
    70: 53, 71: 54, 72: 55, 73: 56, 74: 57, 75: 58, 76: 59, 77: 60,
    78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69
}

def replace_content(content, run_num_stl, run_num_cifar):
    """Replace CIFAR-10 specific content with STL-10 content."""
    
    # Replace dataset name in config and comments
    content = content.replace('cifar10', 'stl10')
    content = content.replace('CIFAR-10', 'STL-10')
    content = content.replace('CIFAR10', 'STL10')
    
    # Replace dataset imports
    content = content.replace(
        'from data.cifar10 import CIFAR10TemporalPairDataset, create_cifar10_temporal_pair_dataset',
        'from data.stl10 import STL10TemporalPairDataset, create_stl10_temporal_pair_dataset'
    )
    
    # Replace dataset class names
    content = content.replace('CIFAR10TemporalPairDataset', 'STL10TemporalPairDataset')
    content = content.replace('create_cifar10_temporal_pair_dataset', 'create_stl10_temporal_pair_dataset')
    
    # Replace image dimensions: 32x32 -> 96x96
    content = re.sub(r'32\s*\*\s*32', '96 * 96', content)  # 32*32 -> 96*96
    content = re.sub(r'32x32', '96x96', content)
    content = re.sub(r'\(32,\s*32\)', '(96, 96)', content)
    content = re.sub(r'32\s*for\s*32x32', '96 for 96x96', content)
    
    # Replace flattened dimension: 3072 -> 27648 (3*96*96 for RGB)
    content = re.sub(r'\bd_in\':\s*3\s*\*\s*32\s*\*\s*32', "d_in': 3 * 96 * 96", content)
    content = re.sub(r'3072', '27648', content)  # Replace 3072 with 27648
    # More specific: replace in comments about flattened dimensions
    content = re.sub(r'flattened to 3072', 'flattened to 27648', content)
    content = re.sub(r'\(3x32x32 = 3072\)', '(3x96x96 = 27648)', content)
    content = re.sub(r'3x32x32 = 3072', '3x96x96 = 27648', content)
    
    # For MLP experiments: update d_in calculation
    content = re.sub(r"'d_in':\s*3\s*\*\s*32\s*\*\s*32", "'d_in': 3 * 96 * 96", content)
    
    # Replace RGB references - keep RGB for STL-10 too
    # (no change needed, STL-10 is also RGB)
    
    # Replace run numbers in docstrings and print statements
    content = re.sub(rf'#0{run_num_cifar:02d}', f'#0{run_num_stl:02d}', content)
    content = re.sub(rf'GRID EXPERIMENT #{run_num_cifar:03d}', f'GRID EXPERIMENT #{run_num_stl:03d}', content)
    content = re.sub(rf'run_grid_exp_{run_num_cifar:03d}', f'run_grid_exp_{run_num_stl:03d}', content)
    
    # Replace run numbers in output directory paths
    content = re.sub(rf'run_{run_num_cifar:03d}_', f'run_{run_num_stl:03d}_', content)
    
    # For Conv-MLP experiments: update input_size (32 -> 96)
    content = re.sub(r"'input_size':\s*32", "'input_size': 96", content)
    content = re.sub(r'input_size=32', 'input_size=96', content)
    content = re.sub(r'Input image size \(32 for 32x32\)', 'Input image size (96 for 96x96)', content)
    content = re.sub(r'\(32 for 32x32\)', '(96 for 96x96)', content)
    
    # Update Conv-MLP padding/kernel comments
    # For 96x96 with kernel=5, padding=2: output size = 96x96 (same as 32x32 with padding=2, kernel=5)
    content = re.sub(r'With padding=2, kernel=5, stride=1: output size = input_size \(32x32\)',
                     'With padding=2, kernel=5, stride=1: output size = input_size (96x96)', content)
    
    # Update image shape comments in forward pass
    content = re.sub(r'Input tensor of shape \(3, H, W\) for RGB image or \(H, W\)',
                     'Input tensor of shape (3, H, W) for RGB image or (H, W) for grayscale, or (C, H, W)', content)
    
    # Update flattening comments in export functions
    content = re.sub(r'Flatten image to 1D tensor \(3x32x32 = 3072\)',
                     'Flatten image to 1D tensor (3x96x96 = 27648)', content)
    content = re.sub(r'Flatten images to 1D tensors \(3x32x32 = 3072\)',
                     'Flatten images to 1D tensors (3x96x96 = 27648)', content)
    content = re.sub(r'Flatten image to 1D \(3x32x32 = 3072\)',
                     'Flatten image to 1D (3x96x96 = 27648)', content)
    
    # For Conv-MLP forward: handle 3-channel images
    # The existing code should work, but let's update comments
    content = re.sub(r'Add channel dimension if needed: \(1, H, W\) for grayscale or \(3, H, W\) for RGB',
                     'Add channel dimension if needed: (1, H, W) for grayscale or (3, H, W) for RGB', content)
    
    # Update dataset comments
    content = re.sub(r'dataset: CIFAR10TemporalPairDataset',
                     'dataset: STL10TemporalPairDataset', content)
    
    # Update export function docstrings
    content = re.sub(r'dataset: CIFAR10TemporalPairDataset \(can be temporal pair',
                     'dataset: STL10TemporalPairDataset (can be temporal pair', content)
    
    # Update image shape return comments
    content = re.sub(r'where images are tensors of shape \(3, 32, 32\)',
                     'where images are tensors of shape (3, 96, 96)', content)
    
    # Update Conv-MLP update method comments for reshaping
    content = re.sub(r'# For CIFAR-10: 3\*32\*32 = 3072',
                     '# For STL-10: 3*96*96 = 27648', content)
    content = re.sub(r'if x_t\.shape\[0\] == 3 \* 32 \* 32:',
                     'if x_t.shape[0] == 3 * 96 * 96:', content)
    content = re.sub(r'# RGB image: reshape to \(3, 32, 32\)',
                     '# RGB image: reshape to (3, 96, 96)', content)
    content = re.sub(r'x_t = x_t\.reshape\(3, 32, 32\)',
                     'x_t = x_t.reshape(3, 96, 96)', content)
    content = re.sub(r'x_t1 = x_t1\.reshape\(3, 32, 32\)',
                     'x_t1 = x_t1.reshape(3, 96, 96)', content)
    
    # Update export function reshaping
    content = re.sub(r'# Flattened - reshape to \(3, 32, 32\)',
                     '# Flattened - reshape to (3, 96, 96)', content)
    content = re.sub(r'image = image\.reshape\(3, 32, 32\)',
                     'image = image.reshape(3, 96, 96)', content)
    
    # Update print statements
    content = re.sub(r'\(3x32x32 flattened\)', '(3x96x96 flattened)', content)
    
    # Update translate_range - STL-10 uses 4 instead of 2
    content = re.sub(r"'translate_range':\s*2", "'translate_range': 4", content)
    content = re.sub(r'translate_range=2', 'translate_range=4', content)
    
    return content


def create_stl10_experiment_file(run_num_stl, run_num_cifar):
    """Create an STL-10 experiment file from CIFAR-10 template."""
    
    # File paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cifar_file = project_root / 'experiments' / f'run_grid_exp_{run_num_cifar:03d}.py'
    stl_file = project_root / 'experiments' / f'run_grid_exp_{run_num_stl:03d}.py'
    
    # Read CIFAR-10 template
    with open(cifar_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace content
    content = replace_content(content, run_num_stl, run_num_cifar)
    
    # Write STL-10 file
    with open(stl_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created: run_grid_exp_{run_num_stl:03d}.py (from run_grid_exp_{run_num_cifar:03d}.py)")


def main():
    """Create all 17 STL-10 experiment files."""
    print("Creating STL-10 experiment files...")
    print("=" * 70)
    
    for run_num_stl, run_num_cifar in sorted(RUN_MAPPING.items()):
        create_stl10_experiment_file(run_num_stl, run_num_cifar)
    
    print("=" * 70)
    print(f"Successfully created {len(RUN_MAPPING)} STL-10 experiment files!")
    print("Files: run_070.py through run_086.py")


if __name__ == "__main__":
    main()


