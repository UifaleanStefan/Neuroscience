# Datasets

This directory contains dataset loaders and utilities for LPL experiments.

## Available Datasets

### `synthetic_shapes.py`
Self-contained synthetic shapes dataset for controlled experiments.

**Features:**
- 4 shape types: vertical bar, horizontal bar, diagonal bar, cross
- 32×32 grayscale images
- Random transformations: translation, rotation, noise
- Supports temporal pairs for LPL training
- Deterministic generation with seeds

**Usage:**
```python
from data.synthetic_shapes import SyntheticShapesDataset, create_temporal_pair_dataset

# Single images
dataset = SyntheticShapesDataset(num_samples=1000, seed=42, return_temporal_pair=False)
image, label = dataset[0]  # Returns (32, 32) tensor, label

# Temporal pairs
temporal_dataset = create_temporal_pair_dataset(num_samples=1000, seed=42)
x_t, x_t1, label = temporal_dataset[0]  # Two views of same shape
```

**Shape Types:**
- Label 0: Vertical bar
- Label 1: Horizontal bar
- Label 2: Diagonal bar
- Label 3: Cross

**Parameters:**
- `num_samples`: Number of samples to generate
- `image_size`: Image size (default: 32)
- `max_translation`: Maximum translation in pixels (default: 3)
- `max_rotation_deg`: Maximum rotation in degrees (default: 10)
- `noise_std`: Standard deviation of Gaussian noise (default: 0.05)
- `seed`: Random seed for reproducibility (default: 42)
- `return_temporal_pair`: If True, returns (x_t, x_t1, label), else (image, label)

**Output:**
- Images: `torch.Tensor` of shape `(32, 32)`, dtype `float32`, values in `[0, 1]`
- Labels: Integer label (0-3)

**Use cases:**
- Controlled experiments with known structure
- Testing hierarchical LPL models
- Visualizing learned representations
- Debugging learning algorithms

**Visualization:**
```bash
python scripts/visualize_shapes.py
```

---

### `cifar.py`
CIFAR-10 dataset utilities and preprocessing.

**Usage:**
```python
import torchvision
import torchvision.transforms as transforms

# Standard CIFAR-10 loading
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
```

**Dataset details:**
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 32×32 RGB images
- 50,000 training images, 10,000 test images

**Preprocessing for LPL:**
- Convert to tensor: `transforms.ToTensor()` (normalizes to [0,1])
- Ensure float32 dtype
- Flatten to 1D for LPL input: `image.flatten()` (3072 dimensions)

**Temporal pairs:**
LPL training scripts create temporal pairs using weak augmentations:
- Small random translations
- Small random rotations
- Slight noise

---

### `stl10.py`
STL-10 dataset utilities (placeholder for future use).

**Note:** This file may be a placeholder. STL-10 is a 96×96 image dataset with 10 classes, commonly used in unsupervised learning research.

---

## Dataset Selection Guide

### For Single-Layer LPL:
- **CIFAR-10**: Natural images, tests generalization to real data
- **Synthetic Shapes**: Simple geometric patterns, easier to interpret

### For Hierarchical LPL:
- **Synthetic Shapes**: Recommended starting point
  - Simpler structure (fewer classes)
  - Known geometric features
  - Easier to visualize and interpret
  - Lower dimensional input (1024 vs 3072)

### For BYOL Baseline:
- **CIFAR-10**: Standard benchmark for contrastive learning
- **Synthetic Shapes**: Direct comparison with hierarchical LPL

---

## Data Format

All datasets should output:
- **Images**: `torch.Tensor`, dtype `float32`, values in `[0, 1]`
- **Labels**: Integer labels (0-indexed)

For LPL training:
- Images are flattened to 1D tensors
- CIFAR-10: `(3, 32, 32)` → `(3072,)`
- Synthetic Shapes: `(32, 32)` → `(1024,)`

For temporal pairs:
- Returns `(x_t, x_t1, label)` where `x_t` and `x_t1` are two views of the same underlying sample

---

## Downloading Datasets

### CIFAR-10
Automatically downloaded by `torchvision.datasets.CIFAR10`:
```python
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
```
Dataset will be saved to `data/cifar-10-batches-py/`.

### Synthetic Shapes
No download needed - generated on the fly:
```python
from data.synthetic_shapes import SyntheticShapesDataset
dataset = SyntheticShapesDataset(num_samples=1000, seed=42)
```

---

## Creating Custom Datasets

To create a custom dataset for LPL:

1. **Inherit from `torch.utils.data.Dataset`:**
   ```python
   from torch.utils.data import Dataset
   
   class MyDataset(Dataset):
       def __init__(self, ...):
           # Initialize
           pass
       
       def __len__(self):
           return self.num_samples
       
       def __getitem__(self, idx):
           # Return (image, label) or (x_t, x_t1, label) for temporal pairs
           return image, label
   ```

2. **Ensure correct format:**
   - Images: `float32`, shape `(H, W)` or `(C, H, W)`, values in `[0, 1]`
   - Labels: Integer

3. **For temporal pairs:**
   - Return `(x_t, x_t1, label)` where `x_t` and `x_t1` are correlated views
   - Use same underlying sample with different transformations/augmentations

---

## Notes

- All datasets use fixed random seeds for reproducibility
- Synthetic shapes are generated deterministically (same seed = same data)
- CIFAR-10 is downloaded once and reused
- Temporal pairs should preserve semantic identity (same object/viewpoint with different transformations)





