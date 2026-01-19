# Swap Selectivity Figure - Class Labels Explanation

## What Are the Classes in the Legend?

In the **Figure 4: Swap Selectivity Change** diagram, the classes shown (Class 0, Class 1, Class 2, etc.) refer to the **CIFAR-10 class labels**.

---

## CIFAR-10 Class Labels

CIFAR-10 has 10 classes with the following mapping:

| Class Index | Class Name |
|-------------|------------|
| **0** | **Airplane** |
| **1** | **Automobile** |
| **2** | **Bird** |
| **3** | **Cat** |
| **4** | **Deer** |
| **5** | **Dog** |
| **6** | **Frog** |
| **7** | **Horse** |
| **8** | **Ship** |
| **9** | **Truck** |

---

## Important Context: Swap Experiment Design

### Two Classes Used for Swapping

The **swap experiment** (training with label swapping) uses:
- **Class 0 (Airplane)** = Object A
- **Class 1 (Automobile)** = Object B

When horizontal translation > 1 pixel during training, the model sees:
- `x_t` from Airplane (class 0) paired with `x_{t+1}` from Automobile (class 1)
- This **swaps the temporal association** between these two classes

### All Classes Exported for Visualization

However, when **exporting activations** (before and after swap training), the code exports activations from **ALL 10 CIFAR-10 classes**, not just classes 0 and 1.

This means:
- The model sees **swap exposure** between classes 0 and 1 during training
- But the **activations exported** include samples from all 10 classes
- The **selectivity plot** shows how neurons respond to **all 10 classes** (not just the swapped ones)

---

## What the Figure Shows

### Before Swap (Left Panel)

Shows neuron selectivity for all 10 CIFAR-10 classes **before training** (random initialization):
- Each line represents one class (Class 0 = Airplane, Class 1 = Automobile, etc.)
- Shows mean activation per neuron across samples of each class
- Patterns are weak/random because the network hasn't learned yet

### After Swap (Right Panel)

Shows neuron selectivity for all 10 CIFAR-10 classes **after swap exposure training**:
- The model was trained with **class 0 ↔ class 1 swapping** (~40% of pairs)
- But activations include **all 10 classes** for visualization
- Shows how representations changed after swap exposure
- Most neurons are near zero (collapsed), few show extreme selectivity

---

## Why Show All Classes?

Even though only classes 0 and 1 were swapped during training, showing all 10 classes helps:
1. **Baseline comparison**: See how non-swapped classes (2-9) compare to swapped ones (0-1)
2. **Representational structure**: Understand how the network represents all categories
3. **Collapse detection**: See if the collapse affects all classes or just swapped ones

---

## Summary

**Class 0** = Airplane (used in swap)  
**Class 1** = Automobile (used in swap)  
**Class 2** = Bird (not swapped, baseline)  
**Class 3** = Cat (not swapped, baseline)  
**Class 4** = Deer (not swapped, baseline)  
**Class 5** = Dog (not swapped, baseline)  
**Class 6** = Frog (not swapped, baseline)  
**Class 7** = Horse (not swapped, baseline)  
**Class 8** = Ship (not swapped, baseline)  
**Class 9** = Truck (not swapped, baseline)

The swap experiment **swaps classes 0 and 1** during training, but the selectivity plot shows activations for **all 10 classes** to provide complete context.
