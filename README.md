# Latent Predictive Learning (LPL) Research Project

This repository contains the implementation and experiments for research on Latent Predictive Learning (LPL), a framework for learning predictive representations in neural networks using local learning rules (no backpropagation).

## Overview

LPL combines three learning mechanisms:
- **Hebbian Learning**: Strengthens connections between co-active neurons
- **Predictive Learning**: Learns to predict future representations
- **Stabilization/Decorrelation**: Prevents representational collapse and maintains diversity

## Project Structure

- **lpl_core/**: Core components including LPL layers, local predictors, and learning rule terms
  - `rules.py`: Learning rule primitives (Hebbian, predictive, stabilization)
  - `lpl_layer.py`: Single-layer LPL module
  - `predictor.py`: Local linear predictor
- **training/**: Training scripts, configuration management, and sanity check utilities
  - `train_lpl.py`: Main training script for CIFAR-10
  - `sanity_check.py`: Numerical stability tests
- **experiments/**: Experiment runners for base experiments, ablation studies, and swap experiments
  - `run_ablations.py`: Ablation studies (disable Hebbian, predictive, or stabilization terms)
  - `run_swap.py`: Swap exposure experiment (Li & DiCarlo 2008 replication)
- **data/**: Dataset loaders and preprocessing for CIFAR-10, STL-10, and synthetic shapes
- **scripts/**: Analysis and utility scripts
  - `analyze_activations.py`: Analyze activation statistics and cosine similarities
  - `check_pt_files.py`: Inspect .pt files for NaN values and statistics
- **outputs/**: Directory for storing activations, weights, and training logs

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn (for analysis scripts)

## Usage

### Training

```bash
# Train LPL on CIFAR-10
python training/train_lpl.py

# Run sanity check
python training/sanity_check.py

# Run ablation experiments
python experiments/run_ablations.py

# Run swap experiment
python experiments/run_swap.py
```

### Analysis

```bash
# Analyze activation files
python scripts/analyze_activations.py

# Check .pt files for issues
python scripts/check_pt_files.py
```

## Features

- **Local Learning Rules**: No backpropagation, optimizers, or autograd
- **Numerical Stability**: Weight clipping, update normalization, and activation squashing
- **Comprehensive Experiments**: Ablation studies and swap exposure experiments
- **Analysis Tools**: Activation statistics and similarity metrics

