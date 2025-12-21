# Latent Predictive Learning (LPL) Research Project

This repository contains the implementation and experiments for research on Latent Predictive Learning (LPL), a framework for learning predictive representations in neural networks using local learning rules (no backpropagation).

## Overview

LPL combines three learning mechanisms:
- **Hebbian Learning**: Strengthens connections between co-active neurons
- **Predictive Learning**: Learns to predict future representations
- **Stabilization/Decorrelation**: Prevents representational collapse and maintains diversity

The project also includes implementations of:
- **Hierarchical LPL**: Multi-layer LPL models with local learning at each layer
- **BYOL (Bootstrap Your Own Latent)**: Contrastive learning baseline for comparison

## Project Structure

```
lpl_project/
├── lpl_core/              # Core LPL components
│   ├── rules.py          # Learning rule primitives (Hebbian, predictive, stabilization)
│   ├── predictor.py      # Local linear predictor
│   ├── lpl_layer.py      # Single-layer LPL module
│   └── hierarchical_lpl.py  # Multi-layer hierarchical LPL
├── training/             # Training scripts
│   ├── train_lpl.py      # Train single-layer LPL on CIFAR-10
│   ├── train_hierarchical_lpl.py  # Train hierarchical LPL on synthetic shapes
│   ├── train_byol.py     # Train BYOL (contrastive baseline)
│   └── sanity_check.py   # Numerical stability tests
├── experiments/          # Experiment runners
│   ├── run_ablations.py  # Ablation studies
│   └── run_swap.py       # Swap exposure experiment (Li & DiCarlo 2008)
├── scripts/              # Analysis and utility scripts
│   ├── analyze_activations.py        # Analyze activation statistics
│   ├── linear_probe.py               # Linear classification probe
│   ├── analyze_swap_identity.py      # Swap experiment analysis
│   ├── analyze_hierarchical_probe.py # Hierarchical linear probe
│   ├── analyze_hierarchical_swap.py  # Hierarchical swap analysis
│   ├── compare_byol_lpl.py           # Compare BYOL vs LPL (shapes)
│   ├── compare_byol_lpl_cifar10.py   # Compare BYOL vs LPL (CIFAR-10)
│   ├── check_pt_files.py             # Inspect .pt files
│   ├── verify_hierarchical_activations.py  # Verify hierarchical activations
│   └── visualize_shapes.py           # Visualize synthetic shapes dataset
├── data/                 # Dataset loaders
│   ├── synthetic_shapes.py  # Synthetic shapes dataset (grayscale, 32x32)
│   ├── cifar.py          # CIFAR-10 dataset utilities
│   └── stl10.py          # STL-10 dataset utilities
└── outputs/              # Output directory
    ├── activations/      # Saved activation tensors
    ├── weights/          # Saved model weights
    └── logs/             # Training logs
```

## Requirements

- Python 3.7+
- PyTorch (tested with 1.9+)
- torchvision
- scikit-learn (for analysis scripts)
- numpy
- matplotlib (for visualization scripts)

Install dependencies:
```bash
pip install torch torchvision scikit-learn numpy matplotlib
```

## Quick Start

### 1. Train Single-Layer LPL on CIFAR-10
```bash
python training/train_lpl.py
```
This will:
- Train LPL on CIFAR-10 with temporally correlated views
- Export activations before and after training to `outputs/activations/`
- Save `activations_before.pt` and `activations_after.pt`

### 2. Train Hierarchical LPL on Synthetic Shapes
```bash
python training/train_hierarchical_lpl.py
```
This will:
- Train a 2-layer hierarchical LPL model on synthetic shapes
- Export activations from both layers before and after training
- Save `hierarchical_activations_before.pt` and `hierarchical_activations_after.pt`

### 3. Train BYOL (Baseline)
```bash
# On CIFAR-10
python training/train_byol.py --dataset cifar10 --steps 1000

# On synthetic shapes
python training/train_byol.py --dataset shapes --steps 1000
```

### 4. Run Experiments
```bash
# Ablation studies
python experiments/run_ablations.py

# Swap exposure experiment
python experiments/run_swap.py
```

### 5. Analyze Results
```bash
# Analyze activations
python scripts/analyze_activations.py

# Linear probe analysis
python scripts/linear_probe.py

# Compare BYOL vs LPL
python scripts/compare_byol_lpl_cifar10.py
```

## Features

### Core Components
- **Local Learning Rules**: No backpropagation, optimizers, or autograd
- **Numerical Stability**: Weight clipping, update normalization, and activation squashing
- **Multi-Layer Support**: Hierarchical LPL with independent local learning at each layer

### Datasets
- **CIFAR-10**: Natural images with weak augmentations for temporal pairs
- **Synthetic Shapes**: Simple geometric shapes (bars, crosses) with transformations
- **STL-10**: Additional natural image dataset support

### Experiments
- **Ablation Studies**: Disable individual learning terms (Hebbian, predictive, stabilization)
- **Swap Exposure**: Replicate Li & DiCarlo (2008) identity preservation experiment
- **BYOL Baseline**: Contrastive learning comparison

### Analysis Tools
- **Activation Statistics**: Mean, std, min, max, L2 norms
- **Linear Probing**: Classification accuracy on frozen features
- **Similarity Metrics**: Cosine similarity, intra/inter-class distances
- **Identity Preservation**: Swap experiment analysis

## Key Design Principles

1. **Local Learning**: Each layer updates independently using only local information
2. **No Global Gradients**: Learning rules are local and do not require backpropagation
3. **Stability**: Multiple mechanisms prevent numerical instability (clipping, normalization, regularization)
4. **Modularity**: Components can be combined and configured independently

## Configuration

Learning rates and other hyperparameters can be adjusted in training scripts. Default values:
- `lr_hebb = 0.001`: Hebbian learning rate
- `lr_pred = 0.001`: Predictive learning rate
- `lr_stab = 0.0005`: Stabilization learning rate

## Output Format

Activation files (`.pt`) contain dictionaries with:
- `activations`: Tensor of shape `(num_samples, feature_dim)`
- `labels`: Tensor of shape `(num_samples,)`
- For hierarchical models: `layer1_activations`, `layer2_activations`, `labels`
- For swap experiments: `activations_before`, `activations_after`, `labels_before`, `labels_after`

## Documentation

For detailed documentation on specific components:
- [Training Scripts](training/README.md)
- [Analysis Scripts](scripts/README.md)
- [Experiments](experiments/README.md)
- [Datasets](data/README.md)

## Citation

If you use this code in your research, please cite:
```
Latent Predictive Learning (LPL) Research Project
```

## License

This project is for research purposes.
