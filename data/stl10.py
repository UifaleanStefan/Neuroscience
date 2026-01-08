"""STL-10 dataset loading with temporal pair support for LPL training."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F


class STL10TemporalPairDataset(Dataset):
    """
    STL-10 dataset with temporal pair support for LPL training.
    
    Creates temporal pairs by applying small random transformations:
    - Translation (±4 pixels for 96x96 images)
    - Gaussian noise
    """
    
    def __init__(self, 
                 root='./data',
                 split='train',
                 return_temporal_pair=False,
                 translate_range=4,
                 noise_std=0.05,
                 seed=None):
        """
        Initialize STL-10 dataset.
        
        Args:
            root: Root directory for STL-10 data
            split: 'train', 'test', or 'unlabeled' (default: 'train')
            return_temporal_pair: If True, returns (x_t, x_t1, label) instead of (image, label)
            translate_range: Maximum translation in pixels (±translate_range)
            noise_std: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility
        """
        self.return_temporal_pair = return_temporal_pair
        self.translate_range = translate_range
        self.noise_std = noise_std
        
        # Load STL-10 dataset
        # STL-10 images are 96x96 RGB (3 channels)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
        ])
        
        self.dataset = torchvision.datasets.STL10(
            root=root,
            split=split,
            download=True,
            transform=transform
        )
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.dataset)
    
    def _apply_translation(self, image, translate_x, translate_y):
        """
        Apply translation to image using padding and cropping.
        
        Args:
            image: Tensor of shape (3, H, W) or (H, W, 3) or (H, W)
            translate_x: Translation in x direction (pixels)
            translate_y: Translation in y direction (pixels)
            
        Returns:
            Translated image tensor of shape (3, H, W)
        """
        # Ensure image is 3D (C, H, W) where C=3 for RGB
        if image.dim() == 3:
            if image.shape[0] == 3:
                pass  # Already (3, H, W)
            elif image.shape[2] == 3:
                image = image.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            else:
                raise ValueError(f"Unexpected 3D image shape: {image.shape}")
        elif image.dim() == 2:
            raise ValueError(f"2D image shape not supported for STL-10 (expecting 3 channels)")
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Pad image to allow translation
        pad_size = abs(translate_x) + abs(translate_y) + 1
        padded = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Calculate crop coordinates
        h, w = image.shape[-2:]
        start_x = pad_size + translate_x
        start_y = pad_size + translate_y
        
        # Crop back to original size
        translated = padded[:, start_y:start_y+h, start_x:start_x+w]
        
        return translated
    
    def _apply_noise(self, image, noise_std):
        """
        Add Gaussian noise to image.
        
        Args:
            image: Tensor of shape (3, H, W)
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy image tensor, clamped to [0, 1]
        """
        noise = torch.randn_like(image) * noise_std
        noisy = image + noise
        return torch.clamp(noisy, 0.0, 1.0)
    
    def _apply_transformations(self, image, transform_seed=None):
        """
        Apply random transformations to create a view of the image.
        
        Args:
            image: Original image tensor (3, 96, 96) for STL-10
            transform_seed: Seed for deterministic transformations
            
        Returns:
            Transformed image tensor of shape (3, 96, 96)
        """
        # Ensure image is (3, H, W)
        if image.dim() == 3:
            if image.shape[0] == 3:
                pass  # Already (3, H, W)
            elif image.shape[2] == 3:
                image = image.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            else:
                raise ValueError(f"Unexpected 3D image shape: {image.shape}")
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}, expected (3, H, W)")
        
        # Use generator for reproducibility if seed provided
        if transform_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(transform_seed)
            translate_x = torch.randint(-self.translate_range, self.translate_range + 1, (1,), generator=gen).item()
            translate_y = torch.randint(-self.translate_range, self.translate_range + 1, (1,), generator=gen).item()
            # Use different seed for noise
            noise_gen = torch.Generator()
            noise_gen.manual_seed(transform_seed + 1000)
            noise_scale = torch.rand(1, generator=noise_gen).item() * self.noise_std
        else:
            translate_x = torch.randint(-self.translate_range, self.translate_range + 1, (1,)).item()
            translate_y = torch.randint(-self.translate_range, self.translate_range + 1, (1,)).item()
            noise_scale = self.noise_std
        
        # Apply translation
        translated = self._apply_translation(image, translate_x, translate_y)
        
        # Apply noise
        transformed = self._apply_noise(translated, noise_scale)
        
        # Ensure output is (3, 96, 96)
        assert transformed.shape == (3, 96, 96), f"Unexpected output shape: {transformed.shape}"
        
        return transformed
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            If return_temporal_pair=False: (image, label)
            If return_temporal_pair=True: (x_t, x_t1, label)
            where images are tensors of shape (3, 96, 96) in [0, 1]
        """
        # Get original image and label from STL-10
        # Note: STL10 returns (image, label) but label can be -1 for unlabeled split
        data = self.dataset[idx]
        
        if isinstance(data, tuple):
            image, label = data
        else:
            image = data
            label = -1  # Unlabeled data
        
        # STL-10 returns (3, 96, 96) tensor via ToTensor()
        # Ensure it's in the correct format
        if image.dim() == 3:
            if image.shape[0] == 3:
                pass  # Already (3, 96, 96)
            elif image.shape[2] == 3:
                image = image.permute(2, 0, 1)  # (96, 96, 3) -> (3, 96, 96)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        if self.return_temporal_pair:
            # Generate two different views of the same image (temporal pair)
            # Use different seeds to ensure different transformations
            x_t = self._apply_transformations(image, transform_seed=idx * 2)
            x_t1 = self._apply_transformations(image, transform_seed=idx * 2 + 1)
            return x_t, x_t1, label
        else:
            # Return single image (use idx as seed for reproducibility)
            image = self._apply_transformations(image, transform_seed=idx)
            return image, label


# Convenience function for creating temporal pair datasets
def create_stl10_temporal_pair_dataset(split='train', **kwargs):
    """
    Create an STL-10 dataset that returns temporal pairs (x_t, x_t1, label).
    
    Args:
        split: 'train', 'test', or 'unlabeled' (default: 'train')
        **kwargs: Additional arguments passed to STL10TemporalPairDataset
        
    Returns:
        STL10TemporalPairDataset configured for temporal pairs
    """
    return STL10TemporalPairDataset(
        split=split,
        return_temporal_pair=True,
        **kwargs
    )


if __name__ == "__main__":
    # Test the dataset
    print("Testing STL10TemporalPairDataset...")
    
    # Test single image mode
    dataset = STL10TemporalPairDataset(split='train', return_temporal_pair=False, seed=42)
    image, label = dataset[0]
    print(f"Single image mode: image shape={image.shape}, label={label}, dtype={image.dtype}")
    print(f"Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    
    # Test temporal pair mode
    temporal_dataset = create_stl10_temporal_pair_dataset(split='train', seed=42)
    x_t, x_t1, label = temporal_dataset[0]
    print(f"\nTemporal pair mode: x_t shape={x_t.shape}, x_t1 shape={x_t1.shape}, label={label}")
    print(f"x_t range: [{x_t.min().item():.3f}, {x_t.max().item():.3f}]")
    print(f"x_t1 range: [{x_t1.min().item():.3f}, {x_t1.max().item():.3f}]")
    
    print("\nDataset test completed successfully!")
