"""LPL layer implementation."""

import torch
from .rules import hebbian, predictive, stabilization
from .predictor import Predictor


class LPLLayer:
    """
    Single-layer Latent Predictive Learning (LPL) module.
    
    Combines Hebbian, predictive, and stabilization learning rules
    to update weights using local learning rules (no backpropagation).
    """
    
    def __init__(self, d_in: int, d_out: int, cfg):
        """
        Initialize the LPL layer.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
            cfg: Configuration object with learning rates and flags:
                - cfg.lr_hebb: Learning rate for Hebbian term
                - cfg.lr_pred: Learning rate for predictive term
                - cfg.lr_stab: Learning rate for stabilization term
                - cfg.use_hebb: Boolean flag to enable Hebbian term
                - cfg.use_pred: Boolean flag to enable predictive term
                - cfg.use_stab: Boolean flag to enable stabilization term
        """
        self.d_in = d_in
        self.d_out = d_out
        self.cfg = cfg
        
        # Initialize weight matrix with small random values
        self.W = torch.randn(d_out, d_in) * 0.01
        
        # Initialize predictor (using predictive learning rate for predictor updates)
        self.predictor = Predictor(d_out, cfg.lr_pred)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute representation: y_t = W @ x_t.
        
        Applies tanh squashing to keep activations bounded and prevent explosion.
        
        Args:
            x: Input tensor of shape (d_in,)
            
        Returns:
            Representation tensor of shape (d_out,)
        """
        assert x.dim() == 1, "x must be 1D"
        assert x.shape[0] == self.d_in, f"x dimension must match input dimension {self.d_in}"
        y = self.W @ x
        # Apply tanh squashing to keep activations bounded (range: -1 to 1)
        # Scale by 5 to allow larger range while still bounding
        y = torch.tanh(y / 5.0) * 5.0
        return y
    
    def update(self, x_t: torch.Tensor, x_t1: torch.Tensor) -> None:
        """
        Update layer weights using local learning rules.
        
        Computes ΔW = ΔW_hebb + ΔW_pred + ΔW_stab and updates W explicitly.
        Includes weight clipping and normalization to prevent numerical instability.
        Also updates the internal predictor.
        
        Args:
            x_t: Input tensor at time t of shape (d_in,)
            x_t1: Input tensor at time t+1 of shape (d_in,)
        """
        assert x_t.dim() == 1, "x_t must be 1D"
        assert x_t1.dim() == 1, "x_t1 must be 1D"
        assert x_t.shape[0] == self.d_in, f"x_t dimension must match input dimension {self.d_in}"
        assert x_t1.shape[0] == self.d_in, f"x_t1 dimension must match input dimension {self.d_in}"
        
        # Compute representations
        y_t = self.forward(x_t)
        y_t1 = self.forward(x_t1)
        
        # Check for NaN in representations before proceeding
        if torch.isnan(y_t).any():
            import warnings
            warnings.warn("NaN detected in y_t representation. Skipping update.")
            return
        if torch.isnan(y_t1).any():
            import warnings
            warnings.warn("NaN detected in y_t1 representation. Skipping update.")
            return
        
        # Get prediction
        y_hat_t1 = self.predictor.forward(y_t)
        if torch.isnan(y_hat_t1).any():
            import warnings
            warnings.warn("NaN detected in y_hat_t1 prediction. Skipping update.")
            return
        
        # Initialize weight update
        dW = torch.zeros_like(self.W)
        
        # Apply Hebbian term if enabled (use in-place addition to avoid creating new tensors)
        if self.cfg.use_hebb:
            dW_hebb = hebbian(x_t, y_t, self.cfg.lr_hebb)
            if torch.isnan(dW_hebb).any():
                import warnings
                warnings.warn("NaN detected in Hebbian update term. Skipping this term.")
            else:
                dW.add_(dW_hebb)  # In-place addition
                del dW_hebb  # Explicitly free intermediate tensor
        
        # Apply predictive term if enabled
        if self.cfg.use_pred:
            dW_pred = predictive(x_t1, y_t1, y_hat_t1, self.cfg.lr_pred)
            if torch.isnan(dW_pred).any():
                import warnings
                warnings.warn("NaN detected in predictive update term. Skipping this term.")
            else:
                dW.add_(dW_pred)  # In-place addition
                del dW_pred  # Explicitly free intermediate tensor
        
        # Apply stabilization term if enabled
        if self.cfg.use_stab:
            dW_stab = stabilization(y_t, self.W, self.cfg.lr_stab)
            if torch.isnan(dW_stab).any():
                import warnings
                warnings.warn("NaN detected in stabilization update term. Skipping this term.")
            else:
                dW.add_(dW_stab)  # In-place addition
                del dW_stab  # Explicitly free intermediate tensor
        
        # Normalize weight update to prevent large jumps
        # Clip the norm of dW to prevent explosive updates
        max_update_norm = 1.0
        dW_norm = torch.norm(dW)
        if dW_norm > max_update_norm:
            scale = max_update_norm / dW_norm
            dW = dW * scale
            del dW_norm, scale  # Free intermediate tensors
        
        # Check for NaN in combined update
        if torch.isnan(dW).any():
            import warnings
            warnings.warn("NaN detected in combined weight update. Skipping update.")
            del dW  # Free before returning
            return
        
        # Update weights explicitly (this creates a new tensor, old W stays until overwritten)
        self.W = self.W + dW
        del dW  # Free dW after it's been used
        
        # Clip weights to safe range to prevent explosion (in-place to save memory)
        self.W.clamp_(min=-5.0, max=5.0)
        
        # Final check for NaN in weights
        if torch.isnan(self.W).any():
            import warnings
            warnings.warn("NaN detected in weights after update. This should not happen.")
        
        # Update predictor after weight update
        self.predictor.update(y_t, y_t1)
        
        # Free intermediate activation tensors
        del y_t, y_t1, y_hat_t1
