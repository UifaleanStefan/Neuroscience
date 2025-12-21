# Experiments

This directory contains scripts for running controlled experiments to evaluate LPL.

## Scripts Overview

### `run_ablations.py`
Ablation study: Disable individual learning terms to understand their contribution.

**Usage:**
```bash
python experiments/run_ablations.py
```

**What it does:**
- Runs 4 different training conditions on CIFAR-10:
  1. **Control (all terms enabled)**: Hebbian + Predictive + Stabilization
  2. **No Hebbian**: Disable Hebbian term (`use_hebb=False`)
  3. **No Predictive**: Disable predictive term (`use_pred=False`)
  4. **No Stabilization**: Disable stabilization term (`use_stab=False`)
  5. **Random temporal pairing**: Shuffle temporal pairs (breaks temporal structure)

- Trains LPL model for each condition
- Exports activations after training

**Output files:**
- `outputs/activations/activations_ablation_hebb.pt`
- `outputs/activations/activations_ablation_pred.pt`
- `outputs/activations/activations_ablation_stab.pt`
- `outputs/activations/activations_ablation_shuffle.pt`

**Analysis:**
Use `scripts/linear_probe.py` to compare classification performance across conditions:
```bash
python scripts/linear_probe.py
```

**Expected findings:**
- **No Hebbian**: May show reduced class separation
- **No Predictive**: May show reduced temporal structure learning
- **No Stabilization**: May show collapsed representations (std < 0.1)
- **Random pairing**: Should show worse performance (validates temporal learning)

---

### `run_swap.py`
Swap exposure experiment: Replicates Li & DiCarlo (2008) to test identity preservation.

**Usage:**
```bash
python experiments/run_swap.py
```

**What it does:**
- Selects two CIFAR-10 classes (default: classes 0 and 1)
- Phase 1: Train LPL on temporally correlated views (normal training)
- Phase 2: Swap the classes (view of class 0 paired with view of class 1)
- Exports activations before swap and after swap exposure

**Rationale:**
This experiment tests whether LPL preserves individual sample identity even when class labels are swapped. Good identity preservation means samples should maintain their representations despite label changes.

**Output files:**
- `outputs/activations/swap_experiment.pt`

**File format:**
```python
{
    'activations_before': torch.Tensor,  # Before swap exposure
    'activations_after': torch.Tensor,   # After swap exposure
    'labels_before': torch.Tensor,       # Original labels
    'labels_after': torch.Tensor         # Swapped labels
}
```

**Analysis:**
Use `scripts/analyze_swap_identity.py` to analyze results:
```bash
python scripts/analyze_swap_identity.py
```

**Metrics computed:**
- Same-sample cosine similarity (before vs after)
- Same-label similarity (class coherence)
- Different-label similarity (class separation)
- Identity preservation score = same-sample - different-label

**Expected findings:**
- High same-sample similarity indicates good identity preservation
- Identity preservation score should be positive if identity is preserved

---

### `run_base.py`
Base experiment runner (may be used for standard training configurations).

**Usage:**
```bash
python experiments/run_base.py
```

**Note:** This script may be a placeholder or contain standard baseline experiments.

---

## Experimental Design

### Ablation Study Design

The ablation study uses a factorial design:
- **Factors**: Learning terms (Hebbian, Predictive, Stabilization) and temporal pairing
- **Levels**: Enabled/Disabled for each term, Correlated/Random for pairing
- **Control**: All terms enabled with correlated temporal pairing
- **Dependent variables**: Classification accuracy, intra/inter-class distances, activation statistics

### Swap Experiment Design

The swap experiment uses a two-phase design:
1. **Phase 1 (Baseline)**: Normal training with temporally correlated views
2. **Phase 2 (Swap)**: Continue training with swapped class pairings
3. **Comparison**: Activations before vs after swap exposure

**Hypothesis:**
If LPL learns identity-preserving representations, samples should maintain similar activations before and after swap exposure, even though class labels change.

---

## Running Experiments

### Complete Experiment Pipeline

1. **Run ablations:**
   ```bash
   python experiments/run_ablations.py
   ```

2. **Run swap experiment:**
   ```bash
   python experiments/run_swap.py
   ```

3. **Analyze results:**
   ```bash
   # Compare ablation conditions
   python scripts/linear_probe.py
   
   # Analyze swap experiment
   python scripts/analyze_swap_identity.py
   ```

### Interpreting Results

**Ablation Study:**
- Compare test accuracies across conditions
- Check activation statistics (std should be > 0.1 for non-collapsed)
- Compare inter/intra ratios (higher = better class separation)
- Identify which terms are critical for performance

**Swap Experiment:**
- High same-sample similarity → Good identity preservation
- Positive identity preservation score → Identity preserved relative to class separation
- Compare same-label vs different-label similarities to understand class structure

---

## Configuration

All experiments use the same default configuration as training scripts:
- Learning rates: `lr_hebb=0.001`, `lr_pred=0.001`, `lr_stab=0.0005`
- Input dimension: 3072 (CIFAR-10)
- Output dimension: 128
- Training steps: 5000 (ablation), variable (swap)

To modify configuration, edit the scripts directly or add command-line arguments.

---

## Notes

- All experiments use fixed random seed (42) for reproducibility
- Experiments are designed to run on CPU (LPL doesn't use GPU)
- Results are saved to `outputs/activations/` for analysis
- Some experiments take time (5000+ steps) - be patient!

---

## References

- **Li & DiCarlo (2008)**: "Unsupervised Natural Experience Rapidly Alters Invariant Object Representation in Visual Cortex" - Original swap exposure experiment
- **BYOL**: Bootstrap Your Own Latent (Grill et al., 2020) - Contrastive learning baseline

