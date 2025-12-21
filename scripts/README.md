# Analysis Scripts

This directory contains scripts for analyzing LPL activations, comparing methods, and visualizing results.

## Scripts Overview

### Activation Analysis

#### `analyze_activations.py`
Analyzes activation statistics from `.pt` files.

**Usage:**
```bash
python scripts/analyze_activations.py
```

**What it does:**
- Computes mean, std, min, max, L2 norms for activations
- Computes intra-class and inter-class L2 distances
- Computes inter/intra ratio (class separation metric)
- Handles standard, hierarchical, and swap experiment formats

**Supported file formats:**
- Standard: `{'activations': tensor, 'labels': tensor}`
- Hierarchical: `{'layer1_activations': tensor, 'layer2_activations': tensor, 'labels': tensor}`
- Swap: `{'activations_before': tensor, 'activations_after': tensor, 'labels': tensor}`

---

#### `linear_probe.py`
Trains a linear classifier on frozen LPL features.

**Usage:**
```bash
python scripts/linear_probe.py
```

**What it does:**
- Loads activation files from `outputs/activations/`
- Splits data 80/20 (train/test)
- Trains LogisticRegression classifier
- Reports test accuracy, intra/inter-class distances, and separation ratio
- Excludes `swap_experiment.pt` and hierarchical files

**Output columns:**
- Test Accuracy: Linear classifier performance
- Intra-Class Dist: Average L2 distance within same class
- Inter-Class Dist: Average L2 distance between different classes
- Ratio: Inter/Intra (higher = better separation)
- Std: Standard deviation (should be > 0.1 to avoid collapse)

---

#### `analyze_swap_identity.py`
Analyzes identity preservation in swap experiments.

**Usage:**
```bash
python scripts/analyze_swap_identity.py
```

**What it does:**
- Loads `swap_experiment.pt` from `outputs/activations/`
- Normalizes activations to unit length
- Computes same-sample cosine similarity (before vs after)
- Computes same-label and different-label similarities
- Computes identity preservation score = same-sample - different-label

**Metrics:**
- Same-Sample Similarity: Individual identity preservation
- Same-Label Similarity: Class coherence
- Different-Label Similarity: Class separation
- Identity Preservation Score: Overall identity preservation metric

---

### Hierarchical Analysis

#### `analyze_hierarchical_probe.py`
Linear probe analysis for hierarchical LPL models.

**Usage:**
```bash
python scripts/analyze_hierarchical_probe.py
```

**What it does:**
- Loads `hierarchical_activations_before.pt` and `hierarchical_activations_after.pt`
- Trains linear classifiers for Layer 1 and Layer 2 separately
- Compares before vs after training metrics
- Reports accuracy, intra/inter distances, and ratios for each layer

---

#### `analyze_hierarchical_swap.py`
Swap-style analysis for hierarchical LPL activations.

**Usage:**
```bash
python scripts/analyze_hierarchical_swap.py
```

**What it does:**
- Loads hierarchical activation files
- Computes same-sample, same-label, and different-label cosine similarities
- Computes identity preservation scores for Layer 1 and Layer 2
- Compares Layer 1 vs Layer 2 metrics

**Output:**
- Layer-wise identity preservation scores
- Comparison of class separation and coherence across layers

---

#### `verify_hierarchical_activations.py`
Verifies hierarchical activations for collapse and NaN issues.

**Usage:**
```bash
python scripts/verify_hierarchical_activations.py
```

**What it does:**
- Checks for collapsed representations (std < 0.1)
- Checks for NaN values
- Prints activation statistics for both layers before and after training
- Reports PASS/FAIL status

---

### Comparison Scripts

#### `compare_byol_lpl.py`
Compares BYOL and LPL on synthetic shapes dataset.

**Usage:**
```bash
python scripts/compare_byol_lpl.py
```

**What it does:**
- Loads BYOL embeddings (`byol_shapes_embeddings_*.pt`)
- Loads LPL hierarchical Layer 1 activations
- Compares classification accuracy, class separation, and representation statistics
- Highlights key differences between contrastive (BYOL) and predictive (LPL) learning

---

#### `compare_byol_lpl_cifar10.py`
Compares BYOL and LPL on CIFAR-10 dataset.

**Usage:**
```bash
python scripts/compare_byol_lpl_cifar10.py
```

**What it does:**
- Loads BYOL embeddings (`byol_embeddings_*.pt`)
- Loads LPL activations (`activations_*.pt`)
- Compares performance and representation characteristics
- Provides interpretation of differences

---

### Utility Scripts

#### `check_pt_files.py`
Inspects `.pt` files for debugging.

**Usage:**
```bash
python scripts/check_pt_files.py
```

**What it does:**
- Iterates over all `.pt` files in `outputs/activations/`
- Prints file name, tensor keys, shapes, and basic statistics
- Handles both single tensor and dictionary formats
- Skips non-floating-point tensors (like labels) when computing stats

**Output format:**
- File name
- Dictionary keys or tensor info
- Shape, dtype, mean, std, min, max
- NaN detection

---

#### `visualize_shapes.py`
Visualizes the synthetic shapes dataset.

**Usage:**
```bash
python scripts/visualize_shapes.py
```

**What it does:**
- Loads synthetic shapes dataset
- Displays sample images for each shape type
- Shows base shapes and temporal pairs
- Saves visualizations to `outputs/`

---

#### `verify_byol_embeddings.py`
Verifies BYOL embeddings for collapse and NaN issues.

**Usage:**
```bash
python scripts/verify_byol_embeddings.py
```

**What it does:**
- Checks BYOL embeddings before and after training
- Verifies L2 normalization (should be ~1.0)
- Checks for NaN values
- Reports statistics

---

## Common Workflow

1. **Run training:**
   ```bash
   python training/train_lpl.py
   python training/train_hierarchical_lpl.py
   ```

2. **Check for issues:**
   ```bash
   python scripts/check_pt_files.py
   python scripts/verify_hierarchical_activations.py
   ```

3. **Analyze activations:**
   ```bash
   python scripts/analyze_activations.py
   python scripts/linear_probe.py
   ```

4. **Compare methods:**
   ```bash
   python scripts/compare_byol_lpl_cifar10.py
   ```

## Output Location

All scripts read from `outputs/activations/` and print results to stdout. Some scripts (like `visualize_shapes.py`) save visualizations to `outputs/`.

## Notes

- All scripts use fixed random seeds (42) for reproducibility
- Scripts handle missing files gracefully with error messages
- Hierarchical scripts require both `*_before.pt` and `*_after.pt` files
- Swap experiment scripts require `swap_experiment.pt` file

