# Local Predictive Learning (LPL) Project - Complete Documentation

## Project Overview

This project implements and evaluates **Local Predictive Learning (LPL)** - a biologically-inspired learning paradigm that uses local learning rules (Hebbian, Predictive, and Stabilization) for unsupervised representation learning. The project compares LPL against standard backpropagation baselines across multiple datasets.

## Project Structure

```
lpl_project/
├── data/                    # Dataset loaders
│   ├── mnist.py            # MNIST dataset (28x28 grayscale)
│   ├── fashion_mnist.py    # Fashion-MNIST dataset (28x28 grayscale)
│   ├── cifar10.py          # CIFAR-10 dataset (32x32 RGB)
│   └── stl10.py            # STL-10 dataset (96x96 RGB)
├── experiments/             # Experiment scripts (run_grid_exp_XXX.py)
├── lpl_core/               # Core LPL implementation
├── outputs/                # Experiment results
│   └── grid_experiments/   # All experiment outputs
├── scripts/                # Utility scripts
└── analysis/               # Analysis tools
```

## Datasets

### 1. Synthetic Shapes (run_001 to run_017)
- **Input size**: 32x32 grayscale (1024 flattened)
- **Purpose**: Initial testing dataset
- **17 experiments**

### 2. MNIST (run_018 to run_034)
- **Input size**: 28x28 grayscale (784 flattened)
- **Normalization**: Standard ImageNet normalization
- **17 experiments**

### 3. Fashion-MNIST (run_036 to run_052)
- **Input size**: 28x28 grayscale (784 flattened)
- **Normalization**: Standard ImageNet normalization
- **17 experiments**

### 4. CIFAR-10 (run_053 to run_069)
- **Input size**: 32x32 RGB (3×32×32 = 3072 flattened)
- **Normalization**: CIFAR-10 specific: mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)
- **17 experiments**

### 5. STL-10 (run_070 to run_086)
- **Input size**: 96x96 RGB (3×96×96 = 27648 flattened)
- **Normalization**: STL-10 specific: mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27)
- **Dataset loading**: Uses `split='train'` (not `train=True`)
- **Translation range**: 4 pixels (for 96x96 images)
- **17 experiments**

## Experiment Structure

### Standard Pattern: 17 Experiments Per Dataset

Each dataset follows the same 17-experiment structure:

1. **MLP 1-layer (128 units)**: 4 experiments
   - 1000 steps
   - 5000 steps
   - 10000 steps
   - 50000 steps

2. **MLP 2-layer (128→64 units)**: 4 experiments
   - 1000 steps
   - 5000 steps
   - 10000 steps
   - 50000 steps

3. **MLP 3-layer**: 4 experiments
   - Architecture varies by dataset:
     - MNIST: 256→128→64 units
     - Others: 128→64→32 units
   - 1000 steps
   - 5000 steps
   - 10000 steps
   - 50000 steps

4. **Conv-MLP Hybrid**: 4 experiments
   - Conv layer: 16 channels, kernel=5, stride=1, padding=2
   - MLP head: 128→64 units
   - Step counts: 1000, 5000, 10000, 20000 (all datasets)

5. **Backpropagation Baseline**: 1 experiment
   - MLP 1-layer (128 units)
   - 10000 steps
   - Standard gradient descent with temporal consistency loss

### Experiment Numbering

- **run_001 to run_017**: Synthetic Shapes
- **run_018 to run_034**: MNIST
- **run_036 to run_052**: Fashion-MNIST (note: run_035 was deleted as duplicate)
- **run_053 to run_069**: CIFAR-10
- **run_070 to run_086**: STL-10

**Total: 85 experiments (17 per dataset × 5 datasets)**

## Key Implementation Details

### Input Dimensions by Dataset

- **MNIST/Fashion-MNIST**: `d_in = 28 * 28 = 784` (grayscale)
- **CIFAR-10**: `d_in = 3 * 32 * 32 = 3072` (RGB)
- **STL-10**: `d_in = 3 * 96 * 96 = 27648` (RGB)

### Conv-MLP Hybrid Architecture

For RGB datasets (CIFAR-10, STL-10), the Conv-MLP hybrid requires special handling:

1. **Input reshaping**: Flattened inputs must be reshaped to `(batch, 3, H, W)` before convolution
2. **Image dimensions**:
   - CIFAR-10: `(3, 32, 32)`
   - STL-10: `(3, 96, 96)`

### Dataset Loading Patterns

All datasets use temporal pair generation:
- **Translation**: Random translation (range varies by image size)
- **Noise**: Gaussian noise addition
- **Temporal pairs**: `(x_t, x_{t+1}, label)` for training
- **Single images**: `(image, label)` for export

### Activation Functions

- **All models**: `tanh` activation (scaled to [-5, 5] range)
- **Clamping**: Activations are clamped to prevent saturation

### Learning Rules

1. **Full LPL**: Hebbian + Predictive + Stabilization
   - Hebbian learning: `lr_hebb = 0.001`
   - Predictive learning: `lr_pred = 0.001`
   - Stabilization: `lr_stab = 0.0005`

2. **Backpropagation**: Standard gradient descent
   - Learning rate: `0.001`
   - Loss: Temporal consistency `L = ||z(x_t) - z(x_{t+1})||^2`

## Output Structure

Each experiment generates:

```
outputs/grid_experiments/run_XXX_<dataset>_<steps>steps_<arch>_<activation>_<rule>/
├── metadata.json           # Experiment configuration
├── training_logs.json      # Training statistics
├── activations_before.pt   # Activations at initialization
├── activations_after.pt    # Activations after training
└── (optional) activations_midpoint.pt  # For long runs
```

For backprop experiments:
- `embeddings_before.pt` and `embeddings_after.pt` (instead of activations)

## Important Files

### Dataset Loaders
- `data/mnist.py`: MNIST temporal pair dataset
- `data/fashion_mnist.py`: Fashion-MNIST temporal pair dataset
- `data/cifar10.py`: CIFAR-10 temporal pair dataset
- `data/stl10.py`: STL-10 temporal pair dataset

### Experiment Scripts
- `experiments/run_grid_exp_XXX.py`: Individual experiment scripts
- Pattern: Each script contains complete experiment setup and execution

### Utility Scripts
- `scripts/check_experiment_status.py`: Check which experiments are complete
- `scripts/verify_all_experiments_complete.py`: Verify all experiments have outputs
- `scripts/delete_extra_experiments.py`: Remove duplicate/extra experiments

## Recent Work Completed

### 1. Fashion-MNIST Experiments (run_036 to run_052)
- Created 17 experiment files
- All experiments run successfully
- Verified outputs match MNIST structure

### 2. CIFAR-10 Experiments (run_053 to run_069)
- Created dataset loader (`data/cifar10.py`)
- Created 17 experiment files
- Fixed input dimensions (`d_in = 3 * 32 * 32 = 3072`)
- Fixed Conv-MLP hybrid to handle RGB images
- All experiments run successfully

### 3. STL-10 Experiments (run_070 to run_086)
- Created dataset loader (`data/stl10.py`)
- Created 17 experiment files
- Fixed dataset loading (`split='train'` instead of `train=True`)
- Fixed input dimensions (`d_in = 3 * 96 * 96 = 27648`)
- Fixed Conv-MLP hybrid for 96x96 RGB images
- Set translation range to 4 pixels
- All experiments run successfully

### 4. Cleanup
- Deleted 5 extra MNIST experiments:
  - 4 duplicate `mlp_3layer_128_64_32` versions (run_026-029)
  - 1 extra `conv_mlp` 20000-step experiment (run_035)
- Deleted corresponding experiment file (`run_grid_exp_035.py`)

## Current Status

✅ **All 85 experiments are complete and verified**

- **synthetic_shapes**: 17 experiments (run_001 to run_017)
- **mnist**: 17 experiments (run_018 to run_034)
- **fashion_mnist**: 17 experiments (run_036 to run_052)
- **cifar10**: 17 experiments (run_053 to run_069)
- **stl10**: 17 experiments (run_070 to run_086)

All experiments have:
- ✅ Metadata files
- ✅ Training logs
- ✅ Activation/embedding exports
- ✅ Healthy activation statistics (std > 0.1, no collapse)

## Key Conventions

1. **Seed**: All experiments use `seed=42` for reproducibility
2. **Device**: CUDA when available, CPU otherwise
3. **Batch size**: Typically 1000 samples for export, 5000 for training
4. **Activation health check**: std > 0.1 threshold to detect collapse
5. **File naming**: `run_XXX_<dataset>_<steps>steps_<arch>_<activation>_<rule>`

## Common Issues and Solutions

### Issue: Conv-MLP Hybrid Input Shape
**Problem**: RGB images need reshaping from flattened to `(3, H, W)`
**Solution**: In `forward()` and `update()` methods, reshape: `x.view(batch_size, 3, H, W)`

### Issue: STL-10 Dataset Loading
**Problem**: `STL10` uses `split='train'` not `train=True`
**Solution**: Use `split='train'` parameter

### Issue: Input Dimensions
**Problem**: RGB datasets need `3 * H * W` not just `H * W`
**Solution**: Set `d_in = 3 * 32 * 32` for CIFAR-10, `3 * 96 * 96` for STL-10

### Issue: Translation Range
**Problem**: Larger images need larger translation ranges
**Solution**: Use `translate_range=4` for 96x96 images (STL-10)

## Next Steps (Potential)

1. **Analysis**: Analyze all 85 experiments for patterns
2. **Visualization**: Create visualizations comparing LPL vs backprop
3. **Reporting**: Generate comprehensive analysis reports
4. **Additional experiments**: If needed, extend to other datasets or architectures

## Verification Commands

To check experiment status:
```bash
python scripts/check_experiment_status.py
python scripts/verify_all_experiments_complete.py
```

To run a single experiment:
```bash
python experiments/run_grid_exp_XXX.py
```

## Notes

- All experiments use the same learning rule hyperparameters
- Activation statistics are logged to detect representation collapse
- Long runs (50000 steps) have intermediate checkpoints
- Conv-MLP experiments have different step patterns (some use 20000 instead of 50000)
- All outputs are saved in `outputs/grid_experiments/`

---

**Last Updated**: After completing all 85 experiments and cleanup
**Status**: All experiments complete and verified ✅

