"""Script to inspect .pt files in outputs/activations/ directory."""

import torch
from pathlib import Path


def print_tensor_stats(tensor, name="tensor"):
    """
    Print basic statistics for a tensor.
    
    Args:
        tensor: PyTorch tensor
        name: Name/label for the tensor
    """
    print(f"    {name}:")
    print(f"      Shape: {tensor.shape}")
    print(f"      Dtype: {tensor.dtype}")
    
    # Check if tensor contains NaN values
    if tensor.is_floating_point():
        nan_count = torch.isnan(tensor).sum().item()
        if nan_count > 0:
            print(f"      WARNING: Contains {nan_count} NaN values!")
    
    # For floating point tensors, compute full statistics
    if tensor.is_floating_point():
        # Check for NaN values
        has_nan = torch.isnan(tensor).any()
        if has_nan:
            # Compute stats only on non-NaN values
            valid_tensor = tensor[~torch.isnan(tensor)]
            if len(valid_tensor) > 0:
                print(f"      Mean: {valid_tensor.mean().item():.6f} (excluding NaN)")
                print(f"      Std: {valid_tensor.std().item():.6f} (excluding NaN)")
                print(f"      Min: {valid_tensor.min().item():.6f} (excluding NaN)")
                print(f"      Max: {valid_tensor.max().item():.6f} (excluding NaN)")
            else:
                print(f"      Mean: nan (all values are nan)")
                print(f"      Std: nan")
                print(f"      Min: nan")
                print(f"      Max: nan")
        else:
            # No NaN values, compute normal statistics
            print(f"      Mean: {tensor.mean().item():.6f}")
            print(f"      Std: {tensor.std().item():.6f}")
            print(f"      Min: {tensor.min().item():.6f}")
            print(f"      Max: {tensor.max().item():.6f}")
    else:
        # For integer tensors (e.g., labels), only show min/max
        print(f"      Min: {tensor.min().item()}")
        print(f"      Max: {tensor.max().item()}")
        print(f"      Unique values: {torch.unique(tensor).numel()}")


def inspect_pt_file(file_path):
    """
    Load and inspect a .pt file.
    
    Args:
        file_path: Path to the .pt file
    """
    print(f"\n{'='*60}")
    print(f"File: {file_path.name}")
    print(f"{'='*60}")
    
    # Load the file
    data = torch.load(file_path, map_location='cpu')
    
    # Check if it's a dictionary or a single tensor
    if isinstance(data, dict):
        # Dictionary format (e.g., with 'activations' and 'labels')
        print(f"Type: Dictionary")
        print(f"Keys: {list(data.keys())}")
        
        # Process each key in the dictionary
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print_tensor_stats(value, name=key)
            else:
                print(f"    {key}: {type(value)} - {value}")
    
    elif isinstance(data, torch.Tensor):
        # Single tensor format
        print(f"Type: Single tensor")
        print_tensor_stats(data, name="tensor")
    
    else:
        # Unknown format
        print(f"Type: {type(data)}")
        print(f"Content: {data}")


def main():
    """Main function to iterate over all .pt files in outputs/activations/."""
    
    # Get the outputs/activations directory
    activations_dir = Path('outputs/activations')
    
    if not activations_dir.exists():
        print(f"Directory {activations_dir} does not exist!")
        return
    
    # Find all .pt files
    pt_files = list(activations_dir.glob('*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in {activations_dir}")
        return
    
    print(f"Found {len(pt_files)} .pt file(s) in {activations_dir}")
    
    # Process each file
    for pt_file in sorted(pt_files):
        try:
            inspect_pt_file(pt_file)
        except Exception as e:
            print(f"\nError loading {pt_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print("Inspection complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

