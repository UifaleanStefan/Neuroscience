# Quick Start Guide - For New Cursor Session

## Welcome!

This project implements **Local Predictive Learning (LPL)** experiments across 5 datasets. **All 85 experiments are complete** and ready for analysis.

## What You Need to Know

### Project Status
✅ **All experiments complete**: 85 total (17 per dataset)
- Synthetic Shapes: run_001-017
- MNIST: run_018-034  
- Fashion-MNIST: run_036-052
- CIFAR-10: run_053-069
- STL-10: run_070-086

### Key Documentation Files
1. **PROJECT_DOCUMENTATION.md** - Start here! Complete overview
2. **HANDOFF_NOTES.md** - Quick reference
3. **TECHNICAL_REFERENCE.md** - Implementation details

### Project Structure
```
lpl_project/
├── data/              # Dataset loaders (mnist, fashion_mnist, cifar10, stl10)
├── experiments/       # 85 experiment scripts (run_grid_exp_XXX.py)
├── outputs/          # All experiment results
│   └── grid_experiments/  # 85 output directories
├── scripts/          # Utility scripts
└── lpl_core/         # Core LPL implementation
```

## Quick Commands

### Check Experiment Status
```bash
python scripts/check_experiment_status.py
python scripts/verify_all_experiments_complete.py
```

### Run a Single Experiment
```bash
python experiments/run_grid_exp_070.py  # Example: STL-10 first experiment
```

### View Experiment Outputs
```bash
# Example: Check run_070 output
ls outputs/grid_experiments/run_070_*/
cat outputs/grid_experiments/run_070_*/metadata.json
```

## Experiment Pattern

Each dataset has **17 experiments**:
- 4 × MLP 1-layer (1000, 5000, 10000, 50000 steps)
- 4 × MLP 2-layer (1000, 5000, 10000, 50000 steps)
- 4 × MLP 3-layer (1000, 5000, 10000, 50000 steps)
- 4 × Conv-MLP Hybrid (varies by dataset)
- 1 × Backprop baseline (10000 steps)

## Important Technical Details

### RGB vs Grayscale
- **Grayscale** (MNIST, Fashion-MNIST): `d_in = 28 * 28 = 784`
- **RGB** (CIFAR-10): `d_in = 3 * 32 * 32 = 3072`
- **RGB** (STL-10): `d_in = 3 * 96 * 96 = 27648`

### STL-10 Special Case
- Uses `split='train'` (not `train=True`)
- Translation range: 4 pixels (for 96×96 images)

### Conv-MLP RGB Handling
- Must reshape flattened input: `x.view(batch, 3, H, W)`
- CIFAR-10: `(3, 32, 32)`
- STL-10: `(3, 96, 96)`

## Output Structure

Each experiment produces:
- `metadata.json` - Configuration
- `training_logs.json` - Training stats
- `activations_before.pt` - Pre-training
- `activations_after.pt` - Post-training
- `activations_midpoint.pt` - Midpoint (long runs)

## Next Steps

The project is ready for:
1. **Analysis** - Compare LPL vs backprop across datasets
2. **Visualization** - Create plots and figures
3. **Reporting** - Generate comprehensive analysis
4. **Extension** - Add new experiments if needed

## Getting Help

- **Full documentation**: See `PROJECT_DOCUMENTATION.md`
- **Technical details**: See `TECHNICAL_REFERENCE.md`
- **Quick reference**: See `HANDOFF_NOTES.md`

## Common Tasks

### Verify All Experiments
```bash
python scripts/verify_all_experiments_complete.py
```

### Check Specific Dataset
```bash
# Check STL-10 experiments
python -c "from pathlib import Path; import json; [print(d.name) for d in Path('outputs/grid_experiments').iterdir() if 'stl10' in d.name]"
```

### View Experiment Config
```bash
# Example: View run_070 config
python -c "import json; print(json.dumps(json.load(open('outputs/grid_experiments/run_070_stl10_1000steps_mlp_1layer_128_tanh_full_lpl/metadata.json')), indent=2))"
```

---

**Everything is documented and ready to go!** 🚀

Start with `PROJECT_DOCUMENTATION.md` for the full picture.

