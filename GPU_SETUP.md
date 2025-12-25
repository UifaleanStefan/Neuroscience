# GPU Setup Instructions

## Current Status
- ✅ GPU detected: NVIDIA GeForce RTX 4060 Ti
- ✅ CUDA drivers installed: CUDA 12.6
- ✅ PyTorch: 2.6.0+cu124 (CUDA support enabled)
- ✅ CUDA available: True
- ✅ GPU Memory: 8.59 GB

## Issue
The PyTorch CUDA installation requires ~2.5 GB of disk space, but installation was interrupted due to insufficient space.

## Solution

### Option 1: Free Disk Space and Reinstall (Recommended)
1. Free up at least 3-4 GB of disk space
2. Run the following commands:

```bash
# Clean up any broken PyTorch installation
pip uninstall torch torchvision -y

# Manually remove torch directory if it still exists
# (Check: C:\Users\User\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch)

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Option 2: Install Specific CUDA Version
If you prefer a different CUDA version:

```bash
# CUDA 11.8 (smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or check latest at: https://pytorch.org/get-started/locally/
```

### Verify Installation
After installation, verify GPU support:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

## Note
The code in `run_grid_exp_016.py` is already set up to automatically use GPU when available. Once PyTorch with CUDA is installed, it will automatically detect and use your GPU.

