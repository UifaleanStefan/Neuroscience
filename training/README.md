# Training Scripts

This directory contains scripts for training LPL models and BYOL baselines.

## Scripts Overview

### `train_lpl.py`
Trains a single-layer LPL model on CIFAR-10.

**Usage:**
```bash
python training/train_lpl.py
```

**What it does:**
- Loads CIFAR-10 dataset
- Creates temporally correlated pairs using weak augmentations
- Trains single-layer LPL with Hebbian, predictive, and stabilization terms
- Exports activations before and after training

**Configuration:**
- Input dimension: 3072 (32x32x3 flattened)
- Output dimension: 128
- Learning rates: `lr_hebb=0.001`, `lr_pred=0.001`, `lr_stab=0.0005`
- Default training steps: 5000

**Output files:**
- `outputs/activations/activations_before.pt`
- `outputs/activations/activations_after.pt`

**File format:**
```python
{
    'activations': torch.Tensor,  # Shape: (num_samples, 128)
    'labels': torch.Tensor        # Shape: (num_samples,)
}
```

**Key features:**
- Input validation (float32, [0,1] range)
- NaN detection and warnings
- Weight clipping for stability
- Diagnostic printing every 500 steps

---

### `train_hierarchical_lpl.py`
Trains a 2-layer hierarchical LPL model on synthetic shapes.

**Usage:**
```bash
python training/train_hierarchical_lpl.py
```

**What it does:**
- Loads synthetic shapes dataset (32x32 grayscale)
- Creates temporal pairs with transformations
- Trains 2-layer hierarchical LPL model
- Exports activations from both layers before and after training

**Architecture:**
- Layer 1: Input (1024) -> Hidden (128)
- Layer 2: Hidden (128) -> Output (64)

**Configuration:**
- Both layers use same learning rates: `lr_hebb=0.001`, `lr_pred=0.001`, `lr_stab=0.0005`
- Default training steps: 10000

**Output files:**
- `outputs/activations/hierarchical_activations_before.pt`
- `outputs/activations/hierarchical_activations_after.pt`

**File format:**
```python
{
    'layer1_activations': torch.Tensor,  # Shape: (num_samples, 128)
    'layer2_activations': torch.Tensor,  # Shape: (num_samples, 64)
    'labels': torch.Tensor                # Shape: (num_samples,)
}
```

**Key features:**
- Independent local learning at each layer
- Layer 2 receives Layer 1 activations as input
- Both layers updated simultaneously using temporal pairs

---

### `train_byol.py`
Trains BYOL (Bootstrap Your Own Latent) as a contrastive learning baseline.

**Usage:**
```bash
# On CIFAR-10
python training/train_byol.py --dataset cifar10 --steps 1000 --batch-size 32

# On synthetic shapes
python training/train_byol.py --dataset shapes --steps 1000 --batch-size 32
```

**Arguments:**
- `--dataset`: Dataset to use (`cifar10` or `shapes`)
- `--steps`: Number of training steps (default: 1000)
- `--batch-size`: Batch size (default: 32)
- `--device`: Device to use (`cpu` or `cuda`, default: `cpu`)

**What it does:**
- Creates online and target networks (identical architecture)
- Only updates first convolutional layer (local learning constraint)
- Updates target network using EMA (tau=0.99)
- Uses BYOL loss (negative cosine similarity between online prediction and target projection)
- Exports embeddings before and after training

**Architecture:**
- 3 convolutional layers (64, 128, 256 channels)
- Global average pooling
- Projection head: 256 -> 512 -> 128
- Prediction head: 128 -> 512 -> 128 (online network only)

**Output files:**
- CIFAR-10: `byol_embeddings_before.pt`, `byol_embeddings_after.pt`, `byol_training_logs.json`
- Shapes: `byol_shapes_embeddings_before.pt`, `byol_shapes_embeddings_after.pt`, `byol_shapes_training_logs.json`

**File format:**
```python
{
    'embeddings': torch.Tensor,  # Shape: (num_samples, 256)
    'labels': torch.Tensor       # Shape: (num_samples,)
}
```

**Key features:**
- Local learning: Only first conv layer updated
- Target network updated via EMA (not gradients)
- Stop gradients on target network
- L2-normalized embeddings (unit length)

---

### `sanity_check.py`
Numerical stability test for LPL layer.

**Usage:**
```bash
python training/sanity_check.py
```

**What it does:**
- Tests single LPL layer with random synthetic data
- Monitors weight norms and activation variance
- Checks for NaN values
- Prints diagnostics every 500 steps

**Use cases:**
- Debug numerical instability
- Verify learning rule implementations
- Test hyperparameter settings

---

## Training Configuration

### Learning Rates
Default learning rates (found to be stable):
- `lr_hebb = 0.001`: Hebbian learning rate
- `lr_pred = 0.001`: Predictive learning rate  
- `lr_stab = 0.0005`: Stabilization learning rate (lower for stability)

### Stability Mechanisms
All training scripts include:
- **Weight clipping**: `torch.clamp(weights, -5.0, 5.0)`
- **Update norm clipping**: Maximum update norm = 1.0
- **Activation squashing**: `tanh(activations / 5.0) * 5.0`
- **NaN detection**: Warnings and early stopping if NaN detected
- **Input validation**: Ensures float32 and [0,1] range

### Model Dimensions

**Single-Layer LPL (CIFAR-10):**
- Input: 3072 (32×32×3)
- Output: 128

**Hierarchical LPL (Synthetic Shapes):**
- Layer 1 Input: 1024 (32×32×1)
- Layer 1 Output: 128
- Layer 2 Input: 128
- Layer 2 Output: 64

**BYOL:**
- Input: 3 channels (CIFAR-10) or 1 channel (shapes)
- Feature dim: 256
- Projection dim: 128

## Output Directory Structure

```
outputs/
├── activations/
│   ├── activations_before.pt
│   ├── activations_after.pt
│   ├── hierarchical_activations_before.pt
│   ├── hierarchical_activations_after.pt
│   ├── byol_embeddings_before.pt
│   ├── byol_embeddings_after.pt
│   └── ...
├── weights/
│   └── (model weights, if saved)
└── logs/
    └── (training logs)
```

## Tips

1. **Start with sanity check** to verify numerical stability
2. **Monitor NaN warnings** - if they appear, reduce learning rates
3. **Check activation statistics** using analysis scripts after training
4. **Use appropriate dataset** - hierarchical LPL is tested on synthetic shapes
5. **BYOL comparison** - train BYOL on same dataset for fair comparison

## Troubleshooting

**NaN values:**
- Reduce learning rates (try 0.0005)
- Check input normalization ([0,1] range)
- Verify weight clipping is working

**Collapsed representations:**
- Check activation std (should be > 0.1)
- Verify stabilization term is enabled
- Try adjusting learning rates

**Slow training:**
- Reduce number of steps for testing
- Use smaller batch sizes for BYOL
- Consider GPU acceleration (BYOL only, LPL is CPU-friendly)


