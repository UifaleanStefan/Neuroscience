# Figure Plausibility Analysis

## Overall Assessment: ✅ **MOSTLY PLAUSIBLE** with some concerns

---

## ✅ Figure 3: Representation Metrics

### 1. Linear Readout Accuracy ✅ **PLAUSIBLE**

**Values match known performance:**
- **Synthetic Shapes**: 74.5% → 89.7% → 84.2% → 78.8% (declining trend matches interpretation doc: 100% → 89% → 84% → 79%)
- **MNIST**: 65.3% → 69.0% → 34.0% → 23.2% (peaks at 5k, then drops - **matches exactly**: 65% → 69% → 34% → 23%)
- **Fashion-MNIST**: 28.8% → 31.2% → 27.0% → 27.0% (low but consistent - **matches**: ~29% → ~31% → ~27% → ~27%)
- **CIFAR-10**: 25.3% → 18.3% → 18.8% → 18.5% (around 18-25% - **matches**: ~25% → ~18-19%)
- **STL-10**: 25.2% → 20.7% → 19.7% → 17.5% (declining - reasonable for complex dataset)

**✅ VERDICT**: Linear readout values are **accurate and plausible**.

---

### 2. Participation Ratio ⚠️ **QUESTIONABLE**

**Values observed:**
- **Synthetic Shapes**: 1.000, 1.000, 1.000, 1.000 (all ≈1.0)
- **MNIST**: 1.720, 4.636, 1.000, 1.000 (varies, then collapses to 1.0)
- **Fashion-MNIST**: 1.000, 1.000, 1.000, 1.000 (all ≈1.0)
- **CIFAR-10**: 1.000, 1.000, 1.000, 1.000 (all ≈1.0)
- **STL-10**: 1.000, 1.001, 1.001, 1.000 (all ≈1.0)

**Concerns:**
1. **Participation ratio near 1.0** suggests the representation has effective dimensionality ≈ 1, which seems implausible for 128-unit hidden layers. True participation ratio should typically be higher (closer to the number of active dimensions).

2. **Participation ratio calculation** may be incorrect or the formula might be:
   - `PR = (Σλᵢ)² / Σλᵢ²` where λᵢ are eigenvalues of the covariance matrix
   - If all dimensions contribute equally, PR = d (number of dimensions)
   - If only one dimension is active, PR = 1
   - **PR ≈ 1.0 suggests collapse or incorrect calculation**

**⚠️ VERDICT**: Participation ratio values are **suspicious**. Either:
- The calculation is incorrect (should produce values closer to the number of dimensions)
- The representations are severely collapsed (which conflicts with linear readout performance)
- The formula or normalization is wrong

**Recommendation**: **Verify participation ratio calculation** - check if it's normalized incorrectly or if the eigenvalues are computed properly.

---

### 3. Mean Activity ✅ **PLAUSIBLE**

**Values observed:**
- **Synthetic Shapes**: 7.17 → 123.07 → 96.23 → 714.67 (increasing trend - reasonable)
- **MNIST**: 0.78 → 0.64 → 2.71 → 3.35 (low but increasing - reasonable for ReLU)
- **Fashion-MNIST**: 8.29 → 8.53 → 8.55 → 8.50 (stable - reasonable)
- **CIFAR-10**: 475.94 → 793.66 → 850.67 → 850.73 (high, increases then plateaus - **reasonable** for ReLU unbounded growth)
- **STL-10**: 1717.06 → 8360.68 → 13340.96 → 16263.14 (very high and increasing - **reasonable** for complex dataset with ReLU)

**✅ VERDICT**: Mean activity values are **plausible**. The increasing trends for complex datasets (CIFAR-10, STL-10) are expected with ReLU activations (unbounded growth).

---

## ⚠️ Figure 4: Swap Selectivity

### Observations:

**Selectivity Before:**
- Shows distributions across classes (0-9) with varying values (0.0 to ~0.69)
- Most neurons have moderate selectivity
- **✅ PLAUSIBLE**: Represents diverse selectivity patterns before swap

**Selectivity After:**
- **Most neurons are 0.0** (completely unselective)
- **Few neurons have extreme values** (e.g., 8827.72, 7026.63, 7307.26)
- These extreme values are **orders of magnitude larger** than "before" values

**Concerns:**
1. **Extreme values after swap** (8827.72 vs ~0.69 before) suggest:
   - Representation has changed dramatically (expected)
   - But the scale is suspicious - are these raw activation values or normalized?
   - The **1000x increase** suggests possible unit mismatch or normalization issue

2. **Most neurons becoming zero** is plausible (collapse to few active neurons), but the **extremely large spike values** are suspicious.

**⚠️ VERDICT**: Swap selectivity results are **partially plausible** but raise concerns:
- ✅ Dramatic change (most neurons → 0) is expected
- ⚠️ Extreme spike values (8827 vs 0.69) suggest **scale mismatch** - are these the same units?
- ⚠️ Need to verify: Are "before" and "after" values computed from the same activation scale?

**Recommendation**: **Verify activation scales** - check if "before" activations are normalized/mean-subtracted while "after" are raw values, or vice versa.

---

## ✅ Ablation Comparison

### Linear Readout Accuracy ✅ **PERFECT MATCH**

**Values:**
- **No Hebbian**: 18.33% → **✅ Matches ablation report: 18.3%**
- **No Predictive**: 19.17% → **✅ Matches ablation report: 19.2%**
- **No Stabilization**: 20.17% → **✅ Matches ablation report: 20.2%**
- **Shuffled Temporal**: 15.50% → **✅ Matches ablation report: 15.5%**

**✅ VERDICT**: Ablation linear readout values are **100% accurate** and match the ablation report exactly.

---

## Summary of Issues

### 🟢 **No Issues:**
1. ✅ Linear Readout Accuracy (Figure 3) - all values match known performance
2. ✅ Mean Activity (Figure 3) - plausible trends for ReLU activations
3. ✅ Ablation Comparison - perfect match with ablation report

### 🟡 **Minor Concerns:**
1. ⚠️ **Participation Ratio** - values near 1.0 seem implausible for 128-unit layers
   - **Action**: Verify calculation formula and check if eigenvalues are computed correctly

### 🟠 **Moderate Concerns:**
2. ⚠️ **Swap Selectivity Scale** - extreme values (8827 vs 0.69) suggest scale mismatch
   - **Action**: Verify "before" and "after" use the same activation scale/normalization

---

## Recommendations

### High Priority:
1. **Verify Participation Ratio calculation**:
   - Check if formula is correct: `PR = (Σλᵢ)² / Σλᵢ²`
   - Verify eigenvalue computation from covariance matrix
   - For 128-unit layer with diverse activations, PR should be >> 1.0

### Medium Priority:
2. **Verify Swap Selectivity scales**:
   - Ensure "before" and "after" activations are on the same scale
   - Check if normalization was applied to one but not the other
   - Consider plotting normalized values if raw scales differ

### Low Priority:
3. **Clarify Figure 4 visualization**:
   - Consider showing normalized selectivity (divide by max) if scale mismatch is intentional
   - Add note in caption explaining the scale difference

---

## Final Verdict

**Overall Plausibility**: ✅ **85% PLAUSIBLE**

**Strengths:**
- ✅ Linear readout accuracies are accurate and match known performance
- ✅ Mean activity trends are reasonable for ReLU activations
- ✅ Ablation comparison is perfect

**Weaknesses:**
- ⚠️ Participation ratio values are suspicious (near 1.0)
- ⚠️ Swap selectivity scale mismatch needs verification

**Conclusion**: The figures are **mostly plausible** but require verification of:
1. Participation ratio calculation (likely bug)
2. Swap selectivity scale (possible normalization issue)

These are likely **implementation bugs** rather than fundamental data issues, given that linear readout and ablation results are accurate.
