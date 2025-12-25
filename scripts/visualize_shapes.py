"""Visualization script for synthetic shapes dataset."""

import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from data.synthetic_shapes import SyntheticShapesDataset, create_temporal_pair_dataset


def visualize_shapes(num_samples_per_class=4, seed=42):
    """
    Visualize samples from the synthetic shapes dataset.
    
    Args:
        num_samples_per_class: Number of samples to show per shape class
        seed: Random seed for reproducibility
    """
    dataset = SyntheticShapesDataset(num_samples=100, seed=seed)
    
    # Get samples for each class
    shape_names = ['Vertical Bar', 'Horizontal Bar', 'Diagonal Bar', 'Cross']
    fig, axes = plt.subplots(4, num_samples_per_class, figsize=(num_samples_per_class * 2, 8))
    fig.suptitle('Synthetic Shapes Dataset - All Classes', fontsize=16)
    
    for class_idx in range(4):
        # Find samples of this class
        class_samples = []
        for i in range(len(dataset)):
            if dataset.labels[i].item() == class_idx:
                image, _ = dataset[i]
                class_samples.append(image)
                if len(class_samples) >= num_samples_per_class:
                    break
        
        # Plot samples for this class
        for j, image in enumerate(class_samples):
            ax = axes[class_idx, j]
            ax.imshow(image.numpy(), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(shape_names[class_idx], fontsize=12, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/shapes_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to outputs/shapes_visualization.png")
    plt.show()


def visualize_temporal_pairs(num_pairs=8, seed=42):
    """
    Visualize temporal pairs (x_t, x_t1) from the dataset.
    
    Args:
        num_pairs: Number of temporal pairs to show
        seed: Random seed for reproducibility
    """
    dataset = create_temporal_pair_dataset(num_samples=100, seed=seed)
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(4, num_pairs * 2))
    fig.suptitle('Temporal Pairs (x_t, x_t1) - Same Shape, Different Transformations', fontsize=14)
    
    shape_names = ['Vertical Bar', 'Horizontal Bar', 'Diagonal Bar', 'Cross']
    
    for i in range(num_pairs):
        x_t, x_t1, label = dataset[i]
        
        # Plot x_t
        ax1 = axes[i, 0]
        ax1.imshow(x_t.numpy(), cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')
        if i == 0:
            ax1.set_title('x_t', fontsize=12)
        if i == num_pairs - 1:
            ax1.text(0.5, -0.1, f'Label: {label} ({shape_names[label]})', 
                    transform=ax1.transAxes, ha='center', fontsize=10)
        
        # Plot x_t1
        ax2 = axes[i, 1]
        ax2.imshow(x_t1.numpy(), cmap='gray', vmin=0, vmax=1)
        ax2.axis('off')
        if i == 0:
            ax2.set_title('x_t1', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('outputs/temporal_pairs_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to outputs/temporal_pairs_visualization.png")
    plt.show()


def visualize_all_classes_base_shapes(seed=42):
    """
    Visualize base shapes (without transformations) for all classes.
    
    Args:
        seed: Random seed (not used for base shapes, but kept for consistency)
    """
    dataset = SyntheticShapesDataset(num_samples=1, seed=seed)
    shape_names = ['Vertical Bar', 'Horizontal Bar', 'Diagonal Bar', 'Cross']
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle('Base Shapes (No Transformations)', fontsize=16)
    
    for class_idx in range(4):
        # Create base shape directly
        base_shape = dataset._create_base_shape(class_idx, 32)
        ax = axes[class_idx]
        ax.imshow(base_shape.numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(shape_names[class_idx], fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/base_shapes_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to outputs/base_shapes_visualization.png")
    plt.show()


def main():
    """Main function to run visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize synthetic shapes dataset')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'shapes', 'pairs', 'base'],
                       help='Visualization mode: all, shapes, pairs, or base')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples per class (for shapes mode)')
    parser.add_argument('--num-pairs', type=int, default=8,
                       help='Number of temporal pairs (for pairs mode)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create outputs directory if it doesn't exist
    Path('outputs').mkdir(exist_ok=True)
    
    if args.mode == 'all' or args.mode == 'shapes':
        print("Visualizing shape samples...")
        visualize_shapes(num_samples_per_class=args.num_samples, seed=args.seed)
    
    if args.mode == 'all' or args.mode == 'pairs':
        print("\nVisualizing temporal pairs...")
        visualize_temporal_pairs(num_pairs=args.num_pairs, seed=args.seed)
    
    if args.mode == 'all' or args.mode == 'base':
        print("\nVisualizing base shapes...")
        visualize_all_classes_base_shapes(seed=args.seed)


if __name__ == "__main__":
    main()





