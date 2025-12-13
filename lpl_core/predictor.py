"""Local predictor component for LPL."""

import torch


class Predictor:
    """
    Local linear predictor for LPL.
    
    Predicts y_hat_t1 = P @ y_t and updates P using the delta rule.
    """
    
    def __init__(self, dim: int, lr: float):
        """
        Initialize the predictor.
        
        Args:
            dim: Dimension of the input/output space
            lr: Learning rate η_P for predictor updates
        """
        self.dim = dim
        self.lr = lr
        self.P = torch.eye(dim)
    
    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction: y_hat_t1 = P @ y_t.
        
        Args:
            y_t: Current output tensor of shape (dim,)
            
        Returns:
            Predicted next output tensor of shape (dim,)
        """
        assert y_t.dim() == 1, "y_t must be 1D"
        assert y_t.shape[0] == self.dim, f"y_t dimension must match predictor dimension {self.dim}"
        return self.P @ y_t
    
    def update(self, y_t: torch.Tensor, y_t1: torch.Tensor) -> None:
        """
        Update predictor parameters using the delta rule.
        
        ΔP = η_P * (y_t1 - y_hat_t1) @ y_t.T
        
        Includes clipping to prevent numerical instability.
        
        Args:
            y_t: Current output tensor of shape (dim,)
            y_t1: Next output tensor of shape (dim,)
        """
        assert y_t.dim() == 1, "y_t must be 1D"
        assert y_t1.dim() == 1, "y_t1 must be 1D"
        assert y_t.shape[0] == self.dim, f"y_t dimension must match predictor dimension {self.dim}"
        assert y_t1.shape[0] == self.dim, f"y_t1 dimension must match predictor dimension {self.dim}"
        
        # Check for NaN in inputs
        if torch.isnan(y_t).any() or torch.isnan(y_t1).any():
            import warnings
            warnings.warn("NaN detected in predictor inputs. Skipping predictor update.")
            return
        
        y_hat_t1 = self.forward(y_t)
        
        # Check for NaN in prediction
        if torch.isnan(y_hat_t1).any():
            import warnings
            warnings.warn("NaN detected in predictor output. Skipping predictor update.")
            return
        
        prediction_error = y_t1 - y_hat_t1
        delta_P = self.lr * torch.outer(prediction_error, y_t)
        
        # Check for NaN in update
        if torch.isnan(delta_P).any():
            import warnings
            warnings.warn("NaN detected in predictor update. Skipping predictor update.")
            return
        
        # Normalize update to prevent large jumps
        max_update_norm = 1.0
        delta_P_norm = torch.norm(delta_P)
        if delta_P_norm > max_update_norm:
            delta_P = delta_P * (max_update_norm / delta_P_norm)
        
        self.P = self.P + delta_P
        
        # Clip predictor weights to safe range
        self.P = torch.clamp(self.P, min=-5.0, max=5.0)
        
        # Final check for NaN
        if torch.isnan(self.P).any():
            import warnings
            warnings.warn("NaN detected in predictor weights after update. This should not happen.")
