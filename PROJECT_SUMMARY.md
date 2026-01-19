# Local Predictive Learning (LPL) Project - Complete Summary

## Project Overview

This project implements and evaluates **Local Predictive Learning (LPL)**, a biologically-inspired unsupervised learning paradigm that uses local learning rules (without backpropagation) to learn representations from temporally correlated visual data. The project systematically explores LPL across multiple datasets, architectures, and training regimes, comparing it against backpropagation baselines.

---

## 1. Project Goals

### Primary Objectives

1. **Implement Local Predictive Learning**
   - Develop biologically-plausible learning rules that operate locally (no backpropagation)
   - Combine three mechanisms: Hebbian learning, Predictive learning, and Stabilization/decorrelation
   - Ensure numerical stability and prevent representational collapse

2. **Evaluate LPL Across Diverse Datasets**
   - Test on datasets varying in complexity: Synthetic Shapes → MNIST → Fashion-MNIST → CIFAR-10 → STL-10
   - Compare LPL against backpropagation baselines
   - Understand how dataset characteristics affect LPL performance

3. **Systematic Architecture Ablation**
   - Compare pure MLP architectures (1-layer, 2-layer, 3-layer)
   - Evaluate Conv-MLP hybrid architectures
   - Understand when convolutional features help vs. hurt

4. **Comprehensive Analysis Framework**
   - Develop metrics for representational quality (linear probing, k-NN, separation ratios)
   - Investigate failure modes and architectural limitations
   - Provide actionable insights for architecture selection

### Key Research Questions

- **Can local learning rules learn useful representations without backpropagation?**
- **How do different architectures (MLP vs Conv-MLP) perform across datasets?**
- **What are the failure modes of LPL, and can we understand them mechanistically?**
- **When do convolutional features help vs. hurt in LPL models?**

---

## 2. Ablation Studies

### Overview

Ablation experiments (`experiments/run_ablations.py`) systematically disable individual learning rule components to understand their contributions to representation learning.

### Ablation Conditions

1. **Full LPL** (Baseline)
   - Hebbian learning: `lr_hebb = 0.001`
   - Predictive learning: `lr_pred = 0.001`
   - Stabilization: `lr_stab = 0.0005`

2. **Ablation: No Hebbian** (`ablation_hebb`)
   - Disables Hebbian learning term
   - Tests contribution of co-activity-based learning

3. **Ablation: No Predictive** (`ablation_pred`)
   - Disables predictive learning term
   - Tests contribution of temporal prediction

4. **Ablation: No Stabilization** (`ablation_stab`)
   - Disables stabilization/decorrelation term
   - Tests contribution of representational diversity maintenance

5. **Ablation: Shuffled Temporal Pairs** (`ablation_shuffle`)
   - Breaks temporal correlation by randomizing `x_{t+1}`
   - Tests whether temporal structure is necessary for learning

### Dataset and Training

- **Dataset**: CIFAR-10
- **Training steps**: 5000
- **Output**: Activation files saved to `outputs/activations/activations_ablation_*.pt`

### Purpose

These ablations reveal which learning mechanisms are critical for successful representation learning and help understand the role of temporal structure in LPL.

---

## 3. Grid Experiment Design

### Overview

The grid experiment system (`experiments/run_grid_exp_XXX.py`) executes **85 total experiments** across 5 datasets, systematically varying architecture, training length, and learning rules.

### Experiment Structure

Each dataset follows the same **17-experiment pattern**:

#### Architecture Variants

1. **MLP 1-layer (128 units)**: 4 experiments
   - Training steps: 1000, 5000, 10000, 50000

2. **MLP 2-layer (128→64 units)**: 4 experiments
   - Training steps: 1000, 5000, 10000, 50000

3. **MLP 3-layer**: 4 experiments
   - Architecture varies by dataset:
     - MNIST: 256→128→64 units
     - Others: 128→64→32 units
   - Training steps: 1000, 5000, 10000, 50000

4. **Conv-MLP Hybrid**: 4 experiments
   - Conv layer: 16 channels, kernel=5, stride=1, padding=2
   - MLP head: 128→64 units
   - Training steps: 1000, 5000, 10000, **20000** (standardized across all datasets)

5. **Backpropagation Baseline**: 1 experiment
   - MLP 1-layer (128 units)
   - 10000 steps
   - Standard gradient descent with temporal consistency loss

### Datasets

| Dataset | Run Numbers | Input Size | Notes |
|---------|-------------|------------|-------|
| **Synthetic Shapes** | run_001-017 | 32×32 grayscale (1024) | Initial testing dataset |
| **MNIST** | run_018-034 | 28×28 grayscale (784) | Digit recognition |
| **Fashion-MNIST** | run_036-052 | 28×28 grayscale (784) | Fashion item recognition |
| **CIFAR-10** | run_053-069 | 32×32 RGB (3072) | Natural image classification |
| **STL-10** | run_070-086 | 96×96 RGB (27648) | Higher-resolution natural images |

**Total: 85 experiments (17 per dataset × 5 datasets)**

### Learning Rules

All LPL experiments use **Full LPL**:
- Hebbian: `lr_hebb = 0.001`
- Predictive: `lr_pred = 0.001`
- Stabilization: `lr_stab = 0.0005`

**Activation Function**: Initially `tanh` (scaled to [-5, 5]), later changed to **ReLU** for all models.

### Output Format

Each experiment generates:
```
outputs/grid_experiments/run_XXX_*/
├── metadata.json              # Experiment configuration
├── training_logs.json         # Training statistics
├── activations_before.pt      # Pre-training activations
└── activations_after.pt       # Post-training activations
```

For backprop experiments: `embeddings_before.pt`, `embeddings_after.pt`

### Key Implementation Details

- **Temporal pair generation**: All datasets generate `(x_t, x_{t+1}, label)` pairs using augmentations (translation, rotation, noise)
- **Numerical stability**: Weight clipping, update normalization, activation squashing
- **Representation health checks**: Monitor activation std (> 0.1 threshold to detect collapse)
- **Reproducibility**: All experiments use `seed=42`

---

## 4. Further Investigation: MNIST Conv-MLP Deep Dive

### Motivation

Initial grid experiment results revealed that **Conv-MLP underperforms MLP on MNIST** (31% vs 69% accuracy at 5k steps), despite **Conv-MLP outperforming MLP on other datasets** (Fashion-MNIST, CIFAR-10, STL-10). This unexpected result prompted a detailed investigation.

### Investigation Phases

#### Phase 1: Initial Analysis (`analysis/diagnose_mnist_conv_failure.py`)
- **Methodology**: PCA/t-SNE visualization, basic separation metrics
- **Limitations**: Unmatched step counts, unnormalized embeddings, speculative interpretations
- **Outcome**: Identified potential issues but lacked methodological rigor

#### Phase 2: Corrective Re-Analysis (`analysis/diagnose_mnist_conv_failure_corrected.py`)
- **Strict Protocol**:
  - Matched step counts only (1k, 5k, 10k)
  - Embedding normalization (zero-mean + L2-normalize)
  - Cosine distance metrics (not Euclidean)
  - No speculative causality from visualizations
- **Key Finding**: Separation ratios nearly identical despite accuracy gap

#### Phase 3: Extended Analysis (`analysis/mnist_extended_analysis.py`)
- **Additional Metrics**:
  - 1-NN and 5-NN classification accuracy
  - Mahalanobis distance-based separation
  - Silhouette scores (clustering quality)
- **Finding**: Conv-MLP has worse k-NN accuracy (40-53% of MLP) despite similar separation

#### Phase 4: Full Validation (`analysis/mnist_full_validation_analysis.py`)
- **New Metrics**:
  - **CKA (Centered Kernel Alignment)**: Representational similarity measure
  - **Per-class metrics**: Intra-class variance and accuracy per digit
- **Comprehensive Visualizations**: PCA, t-SNE, per-class variance plots for all configurations

### Key Findings

#### Primary Result

**Conv-MLP consistently underperforms MLP on MNIST** (mean accuracy difference: -26.1%) despite:
- ✅ **Nearly identical separation ratios** (mean diff ≈ 0.000003)
- ✅ **Moderate to high CKA alignment** (mean: 0.70, range: [0.58, 0.78])
- ✅ **Similar or lower global intra-class variance**

#### What Explains the Accuracy Gap

1. **k-NN Performance**
   - 1-NN accuracy: Conv-MLP only 40% of MLP (0.20 vs 0.59 at 1k steps)
   - 5-NN accuracy: Conv-MLP only 53% of MLP (0.235 vs 0.55 at 1k steps)
   - **Implication**: Worse local structure in embedding space

2. **Clustering Quality (Silhouette)**
   - MLP: -0.19 (mean across steps)
   - Conv-MLP: -0.49 (mean across steps)
   - **Implication**: More overlapping class boundaries

3. **Per-Class Patterns**
   - **Digit 1**: Conv-MLP shows extreme over-localization (93% variance reduction at 5k steps)
   - **Complex digits (4, 5, 6, 7, 9)**: Conv-MLP shows fragmentation (10-38% variance increase)
   - **Implication**: Fails to integrate features for complex patterns; over-localizes simple patterns

#### What Does NOT Explain the Gap

- ❌ **Separation ratio**: Nearly identical
- ❌ **Global intra-class variance**: Similar or lower for Conv-MLP
- ❌ **Representational structure**: CKA = 0.70 suggests similar structures

#### Proposed Explanation

The accuracy gap stems from **linear separability** or **feature alignment** issues:

1. **Different Linear Decision Boundaries**: Despite similar embeddings (CKA = 0.70), Conv-MLP embeddings are less linearly separable

2. **Feature Alignment**: Conv features (edges, textures) may not align well with digit classification task. MLP learns global digit structure directly; Conv-MLP must integrate local features, which fails.

3. **Digit-Specific Fragmentation**: Over-localization (digit 1) and fragmentation (complex digits) both harm classification.

### Methodology

- **Embedding Normalization**: Zero-mean per dimension, L2-normalize per sample
- **Metrics**: Linear probe (3 splits), k-NN (cosine), separation ratios, Mahalanobis, silhouette, CKA
- **Data**: 1000-sample subsets from saved activation files (full 10k test set not available)
- **Matched Configurations**: 1k, 5k, 10k steps (MLP vs Conv-MLP)
- **Unmatched**: 20k steps (Conv-MLP only; no MLP 20k available)

### Deliverables

All analysis outputs saved in `analysis/mnist_conv_mlp_extended/`:
- **Report**: `mnist_full_validation_report.md`
- **Metrics JSON**: `mnist_full_validation_metrics.json`
- **CSV Tables**: `metrics_table.csv`, `per_class_metrics.csv`
- **Visualizations**: 17 PNG files (PCA, t-SNE, per-class variance)

### Recommendations

1. **For MNIST**: Use **Pure MLP** (best: 69% at 5k steps)
2. **Avoid Conv-MLP on MNIST**: Hurts performance (31% best)
3. **Early Stopping**: MLP optimal at 5k steps; longer training degrades
4. **Architecture Selection**: Conv-MLP helps when spatial patterns matter (Fashion-MNIST, CIFAR-10, STL-10) but hurts on simple global patterns (MNIST)

---

## 5. Cross-Dataset Performance Summary

### Conv-MLP vs MLP Comparison

| Dataset | Best MLP | Best Conv-MLP | Winner | Interpretation |
|---------|----------|---------------|--------|----------------|
| **Synthetic Shapes** | 100.0% (1-layer 1k) | **99.8%** (10k) | MLP (but similar) | Both perform well |
| **MNIST** | **69.0%** (1-layer 5k) | 31.0% (5k) | **MLP** | Conv hurts on simple digits |
| **Fashion-MNIST** | 31.2% (1-layer 5k) | **50.0%** (5k) | **Conv-MLP** | Spatial patterns matter |
| **CIFAR-10** | 25.3% (1-layer 1k) | **27.7%** (5k) | **Conv-MLP** | Modest benefit |
| **STL-10** | 25.2% (1-layer 1k) | **31.3%** (10k) | **Conv-MLP** | Higher-res benefits |

### Key Pattern

**Conv-MLP outperforms MLP on complex spatial patterns** (Fashion-MNIST, CIFAR-10, STL-10) but **underperforms on simple global patterns** (MNIST).

**MNIST is the outlier**: Only dataset where Conv-MLP significantly hurts performance (-38% accuracy difference).

---

## 6. Technical Implementation Details

### Local Learning Rules

**No Backpropagation**: All LPL updates use local information only:
- No `autograd`
- No optimizers
- Manual weight updates via learning rule primitives

**Three Mechanisms**:
1. **Hebbian**: `ΔW = lr * outer(y_t, x_t)` - Strengthens co-active connections
2. **Predictive**: `ΔW = lr * outer(ŷ_{t+1} - y_t, x_t)` - Learns temporal prediction
3. **Stabilization**: `ΔW = -lr * (y_outer + identity_reg) @ W` - Prevents collapse

### Numerical Stability

Multiple safeguards prevent instability:
- Weight clipping (`|W| < 1.0`)
- Update normalization
- Activation squashing/clipping
- NaN detection and early stopping

### Activation Function Evolution

- **Initial**: `tanh` scaled to [-5, 5] range: `y = tanh(y / 5.0) * 5.0`
- **Later**: Changed to **ReLU**: `y = relu(y)` for all models
- **Rationale**: ReLU provides better numerical stability and non-negative activations

### Convolutional LPL

Custom implementation (`lpl_core/conv_lpl_layer.py`) enables LPL on convolutional filters:
- Uses `F.unfold` for patch extraction
- Handles 4D tensors (batch, channels, height, width)
- Separate learning rules for convolutional operations
- Memory-efficient with explicit `del` statements and `torch.no_grad()`

---

## 7. Analysis Framework

### Metrics Computed

1. **Classification Performance**:
   - Linear probe accuracy (logistic regression)
   - k-NN classification accuracy (1-NN, 5-NN)

2. **Representation Quality**:
   - Intra-class variance
   - Inter-class distance
   - Separation ratio (intra / inter)
   - Mahalanobis separation ratio
   - Silhouette score (clustering quality)

3. **Representational Alignment**:
   - CKA (Centered Kernel Alignment) - similarity between representations

4. **Per-Class Diagnostics**:
   - Intra-class variance per digit/class
   - Linear probe accuracy per digit/class

### Visualization Tools

- **PCA**: 2D principal component projections
- **t-SNE**: 2D nonlinear dimensionality reduction
- **Per-class plots**: Intra-class variance comparisons

### Normalization Protocol

All embeddings normalized before metric computation:
1. Zero-mean per dimension
2. L2-normalize per sample

This ensures scale-independent, comparable metrics across architectures.

---

## 8. Key Insights and Conclusions

### When Local Predictive Learning Works

✅ **Well-suited for**:
- Temporally correlated visual data
- Tasks where local features matter (Fashion-MNIST, natural images)
- Unsupervised representation learning from video-like sequences

✅ **Best architectures**:
- Pure MLP for simple global patterns (MNIST)
- Conv-MLP for complex spatial patterns (Fashion-MNIST, CIFAR-10, STL-10)

### Limitations and Failure Modes

❌ **Representation collapse**: Can occur with insufficient stabilization
❌ **MNIST anomaly**: Conv-MLP hurts performance despite helping elsewhere
❌ **Linear separability**: Even with similar embeddings (CKA = 0.70), linear classifiers can differ substantially
❌ **Long training degradation**: MLP accuracy can degrade with extended training (e.g., 10k+ steps)

### Architectural Insights

1. **Architecture selection matters**: Conv-MLP helps when spatial patterns matter, hurts on simple global patterns
2. **Early stopping beneficial**: Optimal performance often at intermediate training lengths (5k steps)
3. **Representation quality ≠ Classification accuracy**: Separation metrics can be similar despite large accuracy differences

### Biological Relevance

- ✅ Local learning rules align with biological plausibility (no backpropagation)
- ✅ Temporal prediction aligns with predictive coding theories
- ✅ Stabilization prevents representational collapse (maintains diversity)
- ⚠️ Performance still lags backpropagation on most tasks

---

## 9. Project Statistics

### Experiment Counts

- **Total experiments**: 85
- **Ablation studies**: 5 (on CIFAR-10)
- **Analysis scripts**: 10+
- **Visualizations generated**: 100+

### Datasets Covered

- **Synthetic Shapes**: 32×32 grayscale
- **MNIST**: 28×28 grayscale digits
- **Fashion-MNIST**: 28×28 grayscale fashion items
- **CIFAR-10**: 32×32 RGB natural images
- **STL-10**: 96×96 RGB natural images

### Codebase Size

- **Core modules**: ~5 (rules, layer, predictor, hierarchical, conv)
- **Dataset loaders**: 5
- **Experiment scripts**: 85+
- **Analysis scripts**: 10+

---

## 10. Future Directions

### Potential Extensions

1. **Additional Datasets**: Test on video datasets with strong temporal structure
2. **Architecture Variants**: Deeper networks, different conv architectures
3. **Hyperparameter Tuning**: Optimize learning rates per dataset/architecture
4. **Theoretical Analysis**: Understand why Conv-MLP fails on MNIST mechanistically
5. **Biological Validation**: Compare learned features to biological neural responses

### Open Questions

- Can LPL achieve backprop-level performance with better architectures/training?
- Why does Conv-MLP fail on MNIST specifically?
- How do different learning rule combinations affect performance?
- Can we predict when Conv-MLP will help vs. hurt?

---

## 11. Documentation and Resources

### Key Documentation Files

- `PROJECT_DOCUMENTATION.md`: Complete project overview
- `README.md`: Quick start and project structure
- `ANALYSIS_INTERPRETATION_COMPLETE.md`: Grid experiment results interpretation
- `analysis/MNIST_CONV_MLP_FULL_ANALYSIS_REPORT.md`: Deep dive on MNIST Conv-MLP failure

### Code Organization

- `lpl_core/`: Core LPL implementation
- `experiments/`: Grid experiment scripts and ablations
- `analysis/`: Analysis scripts and reports
- `data/`: Dataset loaders
- `outputs/`: All experiment results and activations

### Reproducibility

- All experiments use `seed=42`
- Complete metadata saved for each experiment
- All activations/embeddings exported for post-hoc analysis
- Version-controlled codebase with clear structure

---

## Summary

This project provides a comprehensive evaluation of **Local Predictive Learning (LPL)** across diverse datasets and architectures. Key achievements:

1. ✅ **85 systematic grid experiments** across 5 datasets
2. ✅ **Ablation studies** understanding learning rule contributions
3. ✅ **Deep investigation** of MNIST Conv-MLP failure mode
4. ✅ **Comprehensive analysis framework** with multiple metrics and visualizations
5. ✅ **Actionable insights** for architecture selection and training

**Main Finding**: LPL can learn useful representations without backpropagation, but architecture selection is critical. Conv-MLP helps on complex spatial patterns but hurts on simple global patterns (MNIST). The accuracy gap on MNIST stems from linear separability issues, not separation metrics.

---

*Last Updated: After completing full MNIST Conv-MLP investigation*  
*Project Status: All experiments complete, comprehensive analysis performed*
