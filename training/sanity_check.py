"""Random-input stability test for LPL training."""

import torch
import sys
from pathlib import Path

# Add project root to path to import lpl_core
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from lpl_core.lpl_layer import LPLLayer


class MinimalConfig:
    """Minimal configuration object for LPL layer."""
    def __init__(self):
        self.lr_hebb = 0.01
        self.lr_pred = 0.01
        self.lr_stab = 0.01
        self.use_hebb = True
        self.use_pred = True
        self.use_stab = True


def main():
    """Run sanity check to verify numerical stability of LPL layer."""
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Create configuration
    cfg = MinimalConfig()
    
    # Create LPL layer: input dimension 20, output dimension 10
    layer = LPLLayer(d_in=20, d_out=10, cfg=cfg)
    
    # Run 5000 training steps with random synthetic data
    num_steps = 5000
    print_interval = 500
    
    print(f"Running sanity check for {num_steps} steps...")
    print(f"{'Step':>6} {'Var(y_t)':>12} {'||W||':>12}")
    print("-" * 32)
    
    for step in range(1, num_steps + 1):
        # Sample x_t ~ N(0, I)
        x_t = torch.randn(20)
        
        # Create x_{t+1} = x_t + ε, where ε ~ N(0, 0.01 I)
        epsilon = torch.randn(20) * 0.1  # std = sqrt(0.01) = 0.1
        x_t1 = x_t + epsilon
        
        # Update layer weights using local learning rules
        layer.update(x_t, x_t1)
        
        # Every 500 steps, print diagnostics
        if step % print_interval == 0:
            # Compute y_t to check representation variance
            y_t = layer.forward(x_t)
            var_y = torch.var(y_t).item()
            
            # Check weight norm to detect explosion
            norm_W = torch.norm(layer.W).item()
            
            print(f"{step:6d} {var_y:12.6f} {norm_W:12.6f}")
    
    print("-" * 32)
    print("Sanity check completed.")
    print("\nChecks performed:")
    print("  - Variance of y_t: Should remain stable (not collapse to zero)")
    print("  - Norm of W: Should remain bounded (not explode)")


if __name__ == "__main__":
    main()
