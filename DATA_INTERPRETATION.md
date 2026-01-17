# Data Interpretation: What We Have From 85 Experiments

## Overview

We have completed **85 experiments** across 5 datasets, collecting comprehensive activation data, training logs, and metadata for analysis.

---

## 1. Data Collection Summary

### Experiment Coverage
- **Synthetic Shapes**: 17 experiments (run_001 to run_017)
- **MNIST**: 17 experiments (run_018 to run_034)
- **Fashion-MNIST**: 17 experiments (run_036 to run_052)
- **CIFAR-10**: 17 experiments (run_053 to run_069)
- **STL-10**: 17 experiments (run_070 to run_086)

**Total: 85 experiments, all complete**

### Architecture Coverage (per dataset)
Each dataset has:
- **MLP 1-layer** (128 units): 4 experiments (1k, 5k, 10k, 50k steps) + 1 backprop baseline
- **MLP 2-layer** (128→64 units): 4 experiments (1k, 5k, 10k, 50k steps)
- **MLP 3-layer** (varies by dataset): 4 experiments (1k, 5k, 10k, 50k steps)
- **Conv-MLP Hybrid** (16 channels → 128→64): 4 experiments (1k, 5k, 10k, 20k steps)

---

## 2. What Data We Have Per Experiment

### For Each Experiment Directory: `run_XXX_*/`

#### **metadata.json**
Experiment configuration:
```json
{
  "dataset": "mnist",
  "steps": 10000,
  "architecture": "mlp_2layer_128_64",
  "activation": "tanh",
  "rule": "full_lpl",
  "baseline": "none",
  "d_in": 784,
  "d_hidden": 128,
  "d_out": 64,
  "lr_hebb": 0.001,
  "lr_pred": 0.001,
  "lr_stab": 0.0005,
  "seed": 42
}
```

#### **training_logs.json**
Time-series training statistics:
- `step`: List of logged step numbers
- `weight_norm`: Weight matrix Frobenius norms at each step
- `activation_norm`: Activation vector norms at each step
- `activation_mean`: Per-layer activation means
- `activation_std`: Per-layer activation standard deviations
- For multi-layer: separate keys for each layer (e.g., `weight_norm_mlp_layer1`, `weight_norm_mlp_layer2`)

**Logging frequency**:
- Short runs (1k-5k): Every step or every 10 steps
- Long runs (10k+): Every 100 or 500 steps (memory optimization)

#### **activations_before.pt** & **activations_after.pt**
Activation representations at different stages:

**For 1-layer MLP:**
```python
{
    "activations": Tensor[1000, 128],  # (num_samples, output_dim)
    "labels": Tensor[1000]              # Class labels (int64)
}
```

**For 2-layer MLP:**
```python
{
    "layer1_activations": Tensor[1000, 128],  # Hidden layer
    "layer2_activations": Tensor[1000, 64],   # Output layer
    "labels": Tensor[1000]
}
```

**For 3-layer MLP:**
```python
{
    "layer1_activations": Tensor[1000, 256/128],  # First hidden
    "layer2_activations": Tensor[1000, 128/64],   # Second hidden
    "layer3_activations": Tensor[1000, 64/32],    # Output
    "labels": Tensor[1000]
}
```

**For Conv-MLP Hybrid:**
```python
{
    "conv_features": Tensor[1000, 16, 28, 28],  # Conv output feature maps
    "mlp_layer1_activations": Tensor[1000, 128], # MLP hidden layer
    "mlp_layer2_activations": Tensor[1000, 64],  # MLP output layer
    "labels": Tensor[1000]
}
```

**For Backprop Baselines:**
```python
{
    "embeddings": Tensor[1000, 128],  # Final embeddings (not activations)
    "labels": Tensor[1000]
}
```

**Optional checkpoints** (for long runs):
- `activations_5000steps.pt` (if ≥10k steps)
- `activations_10000steps.pt` (if ≥20k steps)
- `activations_midpoint.pt` (various naming conventions)

---

## 3. Activation Characteristics

### What Activations Represent

**LPL Models (tanh/ReLU activation):**
- **Before training**: Random projections of input data through randomly initialized weights
- **After training**: Learned representations shaped by:
  - **Hebbian learning**: Strengthens co-activated feature pairs
  - **Predictive learning**: Learns temporal consistency (predicting next frame)
  - **Stabilization**: Prevents collapse and maintains diversity

**Backprop Models (tanh scaled):**
- **Before training**: Random projections
- **After training**: Representations optimized via gradient descent for temporal consistency loss

### Activation Statistics (Examples)

**1-layer MLP (Synthetic Shapes, after 1k steps):**
- Shape: `[1000, 128]`
- Mean: 0.756, Std: 1.255
- Range: `[0.000, 8.974]` (ReLU, so min=0)

**2-layer MLP (Fashion-MNIST, after 10k steps):**
- Layer 1: Mean=8.551, Std=104.877, Range=`[0, 2496]`
- Layer 2: Mean=3677.065, Std=3206.977, Range=`[0, 12482]`
- **Observation**: Activations grow through layers (typical for hierarchical learning)

**Conv-MLP Hybrid (MNIST, after 10k steps):**
- Conv features: Mean=-0.050, Std=0.220, Range=`[-0.961, 0.848]` (tanh-scaled)
- MLP Layer 1: Mean=579.597, Std=2121.779, Range=`[0, 11996]` (ReLU)
- MLP Layer 2: Mean=185470.578, Std=187737.984, Range=`[0, 530820]` (ReLU)
- **Observation**: Large magnitude activations in deeper layers

**Backprop Baseline (MNIST, after 10k steps):**
- Embeddings: Mean=0.284, Std=4.574, Range=`[-4.96, 4.96]` (tanh-scaled to [-5, 5])

---

## 4. Key Metrics We Can Compute

### From Activations

1. **Linear Probing Accuracy**
   - Train a linear classifier on frozen activations
   - Measures task-relevant information in representations
   - **Interpretation**: Higher = more useful features for classification

2. **Intra/Inter-Class Distances**
   - **Intra-class distance**: Average L2 distance between samples of same class
   - **Inter-class distance**: Average L2 distance between samples of different classes
   - **Separation ratio**: `intra_mean / inter_mean`
   - **Interpretation**: Lower ratio = better class separation

3. **Activation Variance Diagnostics**
   - **Std > 0.1**: Representation is healthy (non-collapsed)
   - **Std < 0.1**: Representation has collapsed (warning sign)
   - **Activation distribution**: Mean, std, min, max per layer

4. **Dimensionality Analysis**
   - PCA analysis (effective dimensionality)
   - t-SNE / UMAP visualization (cluster structure)

### From Training Logs

1. **Weight Dynamics**
   - Weight norm over time (growth patterns)
   - Weight change rates (convergence analysis)

2. **Activation Dynamics**
   - Activation norm over time
   - Activation variance over time (collapse detection)
   - Layer-wise activation patterns

3. **Learning Trajectory**
   - Early training vs. late training behavior
   - Convergence patterns
   - Stability analysis

---

## 5. What This Data Tells Us

### Representation Learning
- **How well does LPL learn useful features?** → Compare linear probe accuracy across:
  - Training steps (1k vs. 5k vs. 10k vs. 50k)
  - Architectures (1-layer vs. 2-layer vs. 3-layer)
  - Datasets (simple vs. complex)

### Hierarchical Learning
- **How do representations evolve through layers?** → Compare layer1 vs. layer2 vs. layer3:
  - Activation statistics
  - Linear probe accuracy (layer-wise)
  - Distance metrics (do deeper layers separate classes better?)

### Training Dynamics
- **How do representations develop over time?** → Training log analysis:
  - When do representations stabilize?
  - Are there collapse events?
  - How does variance evolve?

### Architecture Comparison
- **Conv-MLP vs. Pure MLP**: Does spatial feature extraction help?
- **Deep vs. Shallow**: Does depth improve representations?
- **LPL vs. Backprop**: How do local learning rules compare to gradient descent?

### Dataset Characteristics
- **Dataset complexity**: How does LPL performance scale with dataset complexity?
- **Input dimensionality**: How do high-dimensional inputs (CIFAR-10, STL-10) affect learning?

---

## 6. Current Activation Function Status

**Important Note**: After recent changes:
- **MLP layers**: Use **ReLU** activation (`torch.relu`)
- **Conv layers** (in Conv-MLP Hybrid): Still use **tanh** activation (explicit `torch.tanh`)
- **Backprop models**: Use **tanh** scaled to `[-5, 5]` range

**Why this matters**:
- ReLU allows activations to grow unbounded (we see large values like 12,482)
- Tanh caps activations to `[-1, 1]` range
- This affects activation statistics and interpretation

---

## 7. Data Quality Checks

All experiments include:

✅ **NaN Detection**: Training aborts if NaN detected in weights/activations
✅ **Collapse Detection**: Warnings if activation std < 0.1 threshold
✅ **File Verification**: Scripts verify all expected files exist
✅ **Metadata Consistency**: All experiments have complete metadata

---

## 8. Analysis Framework

The `analysis/` directory provides tools for:

- **Linear probing**: `metrics.py::linear_probe()`
- **Distance metrics**: `metrics.py::compute_class_distances()`
- **Variance diagnostics**: `metrics.py::compute_variance_diagnostics()`
- **Visualization**: `plots.py` (PCA, t-SNE, UMAP, scaling curves)
- **Batch processing**: `run_all_analyses.py` (process all 85 experiments)

---

## 9. Key Research Questions We Can Answer

1. **Does LPL learn useful representations?**
   - Compare linear probe accuracy to chance/backprop baselines

2. **How does training length affect representations?**
   - Compare 1k vs. 5k vs. 10k vs. 50k step experiments

3. **Does depth help?**
   - Compare 1-layer vs. 2-layer vs. 3-layer performance

4. **How does dataset complexity affect learning?**
   - Compare Synthetic Shapes → MNIST → CIFAR-10 → STL-10

5. **Do representations collapse?**
   - Check activation std over training (collapse = std < 0.1)

6. **How do local learning rules compare to backprop?**
   - Compare LPL representations to backprop baselines

---

## 10. Next Steps for Analysis

1. **Run comprehensive analysis**: `python -m analysis.run_all_analyses`
2. **Compare architectures**: Plot linear probe accuracy vs. training steps
3. **Visualize representations**: t-SNE/UMAP of activations before/after training
4. **Analyze training dynamics**: Plot weight/activation norms over time
5. **Cross-dataset comparison**: How do representations scale with dataset complexity?

---

**All data is ready for analysis!** 🎉
