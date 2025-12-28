# Activation Analysis Framework

This directory contains comprehensive analysis tools for evaluating LPL (Local Predictive Learning) activation representations.

## Overview

The analysis framework performs systematic evaluation of exported activation representations across:
- **Datasets**: shapes, MNIST, Fashion-MNIST, CIFAR-10, STL-10
- **Architectures**: LPL (1-layer, 2-layer), Local BYOL, Baseline
- **Training lengths**: Various step counts (500, 1k, 5k, 10k, etc.)
- **Seeds**: Multiple random seeds for statistical robustness

## Structure

- `metrics.py`: Core metrics computation (linear probe, distances, variance diagnostics)
- `plots.py`: Visualization utilities (PCA, t-SNE, UMAP, scaling curves, comparisons)
- `run_all_analyses.py`: Main orchestrator script
- `generate_summary.py`: Generate written summary reports

## Expected Input Format

Activations should be organized in one of two structures:

### Standard Structure (Preferred)
```
activations/
  dataset={shapes|mnist|fashion|cifar10|stl10}/
    model={lpl_1layer|lpl_2layer|byol_local|baseline}/
      steps={500|1k|5k|10k|...}/
        seed={0|1|2}/
          before.pt
          after.pt
```

### Legacy Structure (Also Supported)
```
outputs/grid_experiments/
  run_XXX_*/metadata.json
  run_XXX_*/activations_before.pt
  run_XXX_*/activations_after.pt
```

### Activation File Format

Each `.pt` file should contain a dictionary with:
```python
{
    "layer1": Tensor[N, D1],           # Required
    "layer2": Tensor[N, D2],           # Optional (for multi-layer models)
    "labels": Tensor[N]                # Required
}
```

**Legacy formats are automatically converted:**
- `{"activations": Tensor, "labels": Tensor}` → `{"layer1": Tensor, "labels": Tensor}`
- `{"layer1_activations": Tensor, "layer2_activations": Tensor, "labels": Tensor}` → standardized format

## Usage

### Basic Analysis

Run comprehensive analysis on all discovered activation files:

```bash
python -m analysis.run_all_analyses --base-dir . --output-dir analysis_outputs
```

### Options

- `--base-dir`: Base directory containing activations/ or outputs/ (default: `.`)
- `--output-dir`: Output directory for results (default: `analysis_outputs`)
- `--skip-plots`: Skip dimensionality reduction plots (faster, default: False)
- `--random-seed`: Random seed for reproducibility (default: 42)

### Generate Summary

After running analysis, generate a written summary:

```bash
python -m analysis.generate_summary --results-file analysis_outputs/all_results.json --output analysis_outputs/summary.md
```

## Output Structure

```
analysis_outputs/
  ├── all_results.json                 # Raw analysis results (JSON)
  ├── summary.md                       # Written summary report
  ├── figures/                         # Dimensionality reduction plots
  │   └── {dataset}/{model}/{steps}/{seed}/
  │       ├── pca_layer1_before.png
  │       ├── pca_layer1_after.png
  │       ├── tsne_layer1_after.png
  │       └── umap_layer1_after.png
  ├── scaling_curves/                  # Scaling law plots
  │   └── {dataset}/{model}/
  │       ├── accuracy_mean_layer1.png
  │       ├── ratio_layer1.png
  │       └── std_layer1.png
  ├── architecture_comparisons/        # Architecture comparison plots
  │   └── {dataset}/
  │       ├── accuracy_mean_layer1_steps_1k.png
  │       └── ratio_layer1_steps_10k.png
  └── tables/                          # Summary tables
      ├── {dataset}_summary.csv
      └── {dataset}_summary.tex
```

## Metrics Computed

### 1. Linear Probe
- **Test accuracy** with 80/20 train/test split
- **Multiple random splits** (default: 3) for mean ± std
- Uses logistic regression classifier

### 2. Intra/Inter-Class Distance Analysis
- **Average intra-class L2 distance** (within same class)
- **Average inter-class L2 distance** (between different classes)
- **Separation ratio** = inter / intra (higher = better separation)
- **Per-class breakdowns**
- **Distance distributions** (not just means)

### 3. Variance & Collapse Diagnostics
- Global and per-dimension statistics
- **Collapse detection**: std < 0.1 threshold
- **Saturation detection**: tanh at bounds
- Percentage of near-zero variance dimensions
- L2 norm distribution

### 4. Before/After Deltas
- Δ accuracy
- Δ intra/inter distances
- Δ separation ratio
- Δ variance

### 5. Layer-wise Comparison
- Separation ratio gap (layer2 vs layer1)
- Linear probe gap
- Variance gap

## Visualizations

### Dimensionality Reduction
- **PCA**: First 2 principal components
- **t-SNE**: Perplexity=30
- **UMAP**: n_neighbors=15 (requires `umap-learn` package)
- Color-coded by class labels (or time index for shapes dataset)

### Scaling Curves
- Metric vs training steps
- Mean ± std across seeds
- Metrics: accuracy, separation ratio, variance

### Architecture Comparisons
- Bar charts comparing architectures
- Mean ± std across configurations
- Multiple metrics side-by-side

### Distance vs PCA Dimension
- Shows how separation changes with dimensionality
- Intra/inter-class distance curves
- Separation ratio curve

## Dependencies

Required packages:
- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`

Optional (for UMAP):
- `umap-learn` (install with: `pip install umap-learn`)

## Notes

- The analysis framework handles both legacy and new file formats automatically
- NaN values are detected and flagged in results
- Collapsed representations (std < 0.1) are flagged with status 'COLLAPSED'
- Saturated representations (tanh at bounds) are flagged with status 'SATURATED'
- All metrics are computed with proper statistical handling (mean ± std)
- Visualizations are saved in publication-ready format (300 DPI PNG)

