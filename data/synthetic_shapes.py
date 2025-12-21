"""Synthetic shapes dataset generation utilities."""

import torch
import math
from torch.utils.data import Dataset


class SyntheticShapesDataset(Dataset):
    """
    Synthetic shapes dataset generating grayscale images of simple shapes.
    
    Shapes: vertical bar, horizontal bar, diagonal bar, and cross.
    Supports random transformations: translations, rotations, and Gaussian noise.
    Can generate temporal pairs for LPL training.
    """
    
    def __init__(self, 
                 num_samples=1000,
                 image_size=32,
                 translate_range=3,
                 rotation_range=10,
                 noise_std=0.05,
                 return_temporal_pair=False,
                 seed=None):
        """
        Initialize synthetic shapes dataset.
        
        Args:
            num_samples: Number of samples in dataset
            image_size: Size of generated images (image_size x image_size)
            translate_range: Maximum translation in pixels (±translate_range)
            rotation_range: Maximum rotation in degrees (±rotation_range)
            noise_std: Standard deviation of Gaussian noise
            return_temporal_pair: If True, returns (x_t, x_t1, label) instead of (image, label)
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.translate_range = translate_range
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.return_temporal_pair = return_temporal_pair
        
        # Shape types: 0=vertical bar, 1=horizontal bar, 2=diagonal bar, 3=cross
        self.num_classes = 4
        
        # Pre-generate shape labels for each sample
        if seed is not None:
            torch.manual_seed(seed)
        self.labels = torch.randint(0, self.num_classes, (num_samples,))
    
    def __len__(self):
        """Return dataset size."""
        return self.num_samples
    
    def _create_base_shape(self, shape_type, image_size):
        """
        Create a base shape without transformations.
        
        Args:
            shape_type: Integer 0-3 (vertical, horizontal, diagonal, cross)
            image_size: Size of the image
            
        Returns:
            Tensor of shape (image_size, image_size) with the base shape
        """
        image = torch.zeros(image_size, image_size)
        center = image_size // 2
        thickness = 3  # Thickness of bars
        
        if shape_type == 0:  # Vertical bar
            start_col = max(0, center - thickness // 2)
            end_col = min(image_size, center + thickness // 2 + 1)
            image[:, start_col:end_col] = 1.0
            
        elif shape_type == 1:  # Horizontal bar
            start_row = max(0, center - thickness // 2)
            end_row = min(image_size, center + thickness // 2 + 1)
            image[start_row:end_row, :] = 1.0
            
        elif shape_type == 2:  # Diagonal bar (top-left to bottom-right)
            for i in range(image_size):
                for j in range(image_size):
                    # Check if point is near diagonal line y = x
                    dist_to_diagonal = abs(i - j)
                    if dist_to_diagonal <= thickness // 2:
                        image[i, j] = 1.0
            
        elif shape_type == 3:  # Cross (vertical + horizontal bars)
            # Vertical bar
            start_col = max(0, center - thickness // 2)
            end_col = min(image_size, center + thickness // 2 + 1)
            image[:, start_col:end_col] = 1.0
            # Horizontal bar
            start_row = max(0, center - thickness // 2)
            end_row = min(image_size, center + thickness // 2 + 1)
            image[start_row:end_row, :] = 1.0
        
        return image
    
    def _apply_transform(self, image, translate_x, translate_y, rotation_deg):
        """
        Apply translation and rotation to an image.
        
        Args:
            image: Tensor of shape (H, W)
            translate_x: Translation in x direction (pixels)
            translate_y: Translation in y direction (pixels)
            rotation_deg: Rotation angle in degrees
            
        Returns:
            Transformed image tensor of shape (H, W)
        """
        # Convert to 4D format (batch=1, channel=1, H, W) for affine transformation
        image_4d = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Convert rotation to radians
        angle_rad = math.radians(rotation_deg)
        
        # Normalize translation to [-1, 1] range (PyTorch expects normalized coordinates)
        # For 32x32 image, 1 pixel = 2/32 = 0.0625 in normalized coordinates
        norm_factor = 2.0 / self.image_size
        translate_x_norm = translate_x * norm_factor
        translate_y_norm = translate_y * norm_factor
        
        # Create affine transformation matrix
        # PyTorch's affine_grid expects: [theta_11, theta_12, tx]
        #                                 [theta_21, theta_22, ty]
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        theta = torch.tensor([[
            [cos_angle, -sin_angle, translate_x_norm],
            [sin_angle, cos_angle, translate_y_norm]
        ]], dtype=torch.float32)
        
        # Create grid and apply transformation
        grid = torch.nn.functional.affine_grid(
            theta, 
            image_4d.size(), 
            align_corners=False
        )
        transformed = torch.nn.functional.grid_sample(
            image_4d, 
            grid, 
            align_corners=False,
            padding_mode='zeros'
        )
        
        return transformed.squeeze(0).squeeze(0)  # Back to (H, W)
    
    
    def _generate_image(self, shape_type, transform_seed=None):
        """
        Generate a single transformed image.
        
        Args:
            shape_type: Integer 0-3 indicating shape type
            transform_seed: Seed for generating transformations (None = truly random)
            
        Returns:
            Tensor of shape (image_size, image_size) with transformed shape
        """
        # Generate random transformation parameters
        if transform_seed is not None:
            # Use a deterministic generator for reproducible transformations
            gen = torch.Generator()
            gen.manual_seed(transform_seed)
            translate_x = torch.rand(1, generator=gen).item() * 2 * self.translate_range - self.translate_range
            translate_y = torch.rand(1, generator=gen).item() * 2 * self.translate_range - self.translate_range
            rotation_deg = torch.rand(1, generator=gen).item() * 2 * self.rotation_range - self.rotation_range
            noise_seed = transform_seed + 1000  # Different seed for noise
        else:
            translate_x = torch.rand(1).item() * 2 * self.translate_range - self.translate_range
            translate_y = torch.rand(1).item() * 2 * self.translate_range - self.translate_range
            rotation_deg = torch.rand(1).item() * 2 * self.rotation_range - self.rotation_range
            noise_seed = None
        
        # Create base shape
        base_image = self._create_base_shape(shape_type, self.image_size)
        
        # Apply transformations
        transformed = self._apply_transform(base_image, translate_x, translate_y, rotation_deg)
        
        # Add noise (with deterministic seed if provided)
        if noise_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(noise_seed)
            noise = torch.randn(transformed.shape, generator=gen) * self.noise_std
        else:
            noise = torch.randn_like(transformed) * self.noise_std
        final_image = torch.clamp(transformed + noise, 0.0, 1.0)
        
        return final_image
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            If return_temporal_pair=False: (image, label)
            If return_temporal_pair=True: (x_t, x_t1, label)
            where images are tensors of shape (image_size, image_size)
        """
        label = self.labels[idx].item()
        
        if self.return_temporal_pair:
            # Generate two different views of the same shape (temporal pair)
            # Use different seeds to ensure different transformations
            x_t = self._generate_image(label, transform_seed=idx * 2)
            x_t1 = self._generate_image(label, transform_seed=idx * 2 + 1)
            return x_t, x_t1, label
        else:
            # Generate a single image (use idx as seed for reproducibility)
            image = self._generate_image(label, transform_seed=idx)
            return image, label


# Convenience function for creating temporal pair datasets
def create_temporal_pair_dataset(num_samples=1000, **kwargs):
    """
    Create a dataset that returns temporal pairs (x_t, x_t1, label).
    
    Args:
        num_samples: Number of samples
        **kwargs: Additional arguments passed to SyntheticShapesDataset
        
    Returns:
        SyntheticShapesDataset configured for temporal pairs
    """
    return SyntheticShapesDataset(
        num_samples=num_samples,
        return_temporal_pair=True,
        **kwargs
    )


if __name__ == "__main__":
    # Test the dataset
    print("Testing SyntheticShapesDataset...")
    
    # Test single image mode
    dataset = SyntheticShapesDataset(num_samples=10, seed=42)
    print(f"Dataset size: {len(dataset)}")
    
    image, label = dataset[0]
    print(f"Sample image shape: {image.shape}, label: {label}")
    print(f"Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    
    # Test temporal pair mode
    temporal_dataset = create_temporal_pair_dataset(num_samples=10, seed=42)
    x_t, x_t1, label = temporal_dataset[0]
    print(f"\nTemporal pair - x_t shape: {x_t.shape}, x_t1 shape: {x_t1.shape}, label: {label}")
    
    print("\nDataset test completed successfully!")
