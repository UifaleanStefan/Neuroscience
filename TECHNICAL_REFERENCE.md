# Technical Reference - Implementation Details

## Architecture Specifications

### MLP Architectures

#### 1-Layer MLP
- **Input**: Dataset-specific (784, 3072, or 27648)
- **Output**: 128 units
- **Activation**: tanh (scaled to [-5, 5])

#### 2-Layer MLP
- **Input**: Dataset-specific
- **Hidden**: 128 units
- **Output**: 64 units
- **Activation**: tanh (scaled to [-5, 5])

#### 3-Layer MLP
- **MNIST**: 784 → 256 → 128 → 64
- **Other datasets**: Input → 128 → 64 → 32
- **Activation**: tanh (scaled to [-5, 5])

### Conv-MLP Hybrid

```
Input (C×H×W)
  ↓
Conv2d(C → 16, kernel=5, stride=1, padding=2)
  ↓
Flatten
  ↓
MLP: 128 → 64
```

**Channel counts**:
- MNIST/Fashion-MNIST: 1 channel (grayscale)
- CIFAR-10: 3 channels (RGB, 32×32)
- STL-10: 3 channels (RGB, 96×96)

## Dataset Specifications

### Input Dimensions

| Dataset | Image Size | Channels | Flattened | d_in |
|---------|-----------|----------|-----------|------|
| MNIST | 28×28 | 1 | 784 | 784 |
| Fashion-MNIST | 28×28 | 1 | 784 | 784 |
| CIFAR-10 | 32×32 | 3 | 3072 | 3072 |
| STL-10 | 96×96 | 3 | 27648 | 27648 |

### Normalization

**MNIST/Fashion-MNIST**:
```python
transforms.Normalize((0.1307,), (0.3081,))
```

**CIFAR-10**:
```python
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
```

**STL-10**:
```python
transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
```

### Temporal Pair Generation

All datasets generate temporal pairs with:
- **Translation**: Random translation (range depends on image size)
  - 28×28 images: range = 2
  - 32×32 images: range = 2
  - 96×96 images: range = 4
- **Noise**: Gaussian noise (std = 0.1)

## Learning Rule Parameters

### LPL (Local Predictive Learning)

```python
LayerConfig(
    lr_hebb=0.001,      # Hebbian learning rate
    lr_pred=0.001,       # Predictive learning rate
    lr_stab=0.0005,      # Stabilization learning rate
    use_hebb=True,       # Enable Hebbian learning
    use_pred=True,       # Enable predictive learning
    use_stab=True        # Enable stabilization
)
```

### Backpropagation

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # Temporal consistency: ||z(x_t) - z(x_{t+1})||^2
```

## Experiment Execution Pattern

### Standard Flow

1. **Initialize model** with seed=42
2. **Export activations before training**
3. **Train for N steps** (1000, 5000, 10000, 20000, or 50000)
4. **Export activations after training**
5. **Check activation health** (std > 0.1)
6. **Save all outputs**

### Long Run Handling

For experiments with ≥10000 steps:
- Progress reported every 500 steps
- Intermediate checkpoints at midpoint (for 10000+ step runs)
- Collapse detection (abort if std < 0.1)

## Output File Formats

### metadata.json
```json
{
  "dataset": "mnist",
  "architecture": "mlp_1layer_128",
  "steps": 1000,
  "learning_rule": "full_lpl",
  "seed": 42,
  "d_in": 784,
  "d_out": 128,
  ...
}
```

### training_logs.json
```json
{
  "step_100": {
    "weight_norm": 31.9983,
    "activation_norm": 56.5685,
    "activation_std": 5.0141
  },
  ...
}
```

### activations_*.pt
PyTorch tensors:
- **Before**: `(num_samples, num_units)` - activations at initialization
- **After**: `(num_samples, num_units)` - activations after training
- **Midpoint**: `(num_samples, num_units)` - activations at midpoint (if applicable)

For multi-layer models: Dictionary with keys `layer1_activations`, `layer2_activations`, etc.

## Critical Code Patterns

### RGB Image Handling in Conv-MLP

```python
# In forward() method
if len(x.shape) == 2:  # Flattened input
    batch_size = x.shape[0]
    x = x.view(batch_size, 3, H, W)  # Reshape to (B, C, H, W)
# Then apply convolution
```

### Dataset Loading Pattern

```python
# For STL-10 (special case)
dataset = STL10TemporalPairDataset(
    split='train',  # NOT train=True
    return_temporal_pair=True,
    seed=42
)

# For others
dataset = CIFAR10TemporalPairDataset(
    train=True,
    return_temporal_pair=True,
    seed=42
)
```

### Activation Health Check

```python
activation_std = activations.std().item()
if activation_std < 0.1:
    print(f"WARNING: Activation std ({activation_std}) is below 0.1 threshold!")
    # May indicate representation collapse
```

## File Naming Conventions

### Experiment Scripts
- Pattern: `run_grid_exp_XXX.py`
- XXX: 3-digit experiment number (001-086)

### Output Directories
- Pattern: `run_XXX_<dataset>_<steps>steps_<arch>_<activation>_<rule>`
- Example: `run_070_stl10_1000steps_mlp_1layer_128_tanh_full_lpl`

### Output Files
- `metadata.json`: Experiment configuration
- `training_logs.json`: Training statistics
- `activations_before.pt`: Pre-training activations
- `activations_after.pt`: Post-training activations
- `activations_midpoint.pt`: Midpoint activations (long runs)
- `embeddings_*.pt`: For backprop experiments

## Common Gotchas

1. **RGB vs Grayscale**: Always check channel count when reshaping
2. **STL-10 split parameter**: Use `split='train'` not `train=True`
3. **Input dimensions**: RGB = `3 * H * W`, not just `H * W`
4. **Translation range**: Scale with image size (larger images = larger range)
5. **Activation clamping**: tanh outputs clamped to [-5, 5] to prevent saturation

## Verification Checklist

Before considering experiments complete:
- [ ] All 17 experiments per dataset exist
- [ ] All have `metadata.json`
- [ ] All have `training_logs.json`
- [ ] All have activation/embedding files
- [ ] Activation std > 0.1 (no collapse)
- [ ] No NaN values in weights or activations
- [ ] File naming is consistent

## Performance Notes

- **Training time**: Varies by dataset size and steps
  - Small datasets (MNIST): ~minutes
  - Large datasets (STL-10): ~hours for long runs
- **Memory**: Conv-MLP experiments use more memory (especially STL-10)
- **GPU**: CUDA recommended for faster training

---

**Use this document for quick technical reference when implementing new features or debugging.**

