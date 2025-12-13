"""Learning rule terms and update mechanisms for LPL."""

import torch


def hebbian(x_t: torch.Tensor, y_t: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Compute the Hebbian learning rule update term.
    
    ΔW_hebb = η_h * y_t @ x_t.T
    
    Args:
        x_t: Input tensor of shape (d_in,)
        y_t: Output tensor of shape (d_out,)
        lr: Learning rate η_h
        
    Returns:
        Weight update tensor of shape (d_out, d_in)
    """
    assert x_t.dim() == 1, "x_t must be 1D"
    assert y_t.dim() == 1, "y_t must be 1D"
    return lr * torch.outer(y_t, x_t)


def predictive(x_t1: torch.Tensor,
               y_t1: torch.Tensor,
               y_hat_t1: torch.Tensor,
               lr: float) -> torch.Tensor:
    """
    Compute the predictive learning rule update term.
    
    ΔW_pred = η_p * (y_hat_t1 - y_t1) @ x_t1.T
    
    Args:
        x_t1: Input tensor at time t+1 of shape (d_in,)
        y_t1: Target output tensor at time t+1 of shape (d_out,)
        y_hat_t1: Predicted output tensor at time t+1 of shape (d_out,)
        lr: Learning rate η_p
        
    Returns:
        Weight update tensor of shape (d_out, d_in)
    """
    assert x_t1.dim() == 1, "x_t1 must be 1D"
    assert y_t1.dim() == 1, "y_t1 must be 1D"
    assert y_hat_t1.dim() == 1, "y_hat_t1 must be 1D"
    assert y_t1.shape == y_hat_t1.shape, "y_t1 and y_hat_t1 must have the same shape"
    return lr * torch.outer(y_hat_t1 - y_t1, x_t1)


def stabilization(y_t: torch.Tensor,
                  W: torch.Tensor,
                  lr: float) -> torch.Tensor:
    """
    Compute the stabilization/decorrelation learning rule update term.
    
    ΔW_stab = -η_s * (y_t @ y_t.T + ε*I) @ W
    
    Includes adaptive scaling to prevent explosive updates when activations are large.
    
    Args:
        y_t: Output tensor of shape (d_out,)
        W: Weight matrix of shape (d_out, d_in)
        lr: Learning rate η_s
        
    Returns:
        Weight update tensor of shape (d_out, d_in)
    """
    assert y_t.dim() == 1, "y_t must be 1D"
    assert W.dim() == 2, "W must be 2D"
    assert y_t.shape[0] == W.shape[0], "y_t first dimension must match W first dimension"
    
    d_out = y_t.shape[0]
    
    # Clip y_t to prevent extremely large activations from causing explosion
    # This bounds the outer product
    y_t_clipped = torch.clamp(y_t, min=-10.0, max=10.0)
    
    # Add epsilon regularization to prevent numerical instability
    epsilon = 1e-6
    y_outer = torch.outer(y_t_clipped, y_t_clipped)
    identity_reg = epsilon * torch.eye(d_out, device=W.device, dtype=W.dtype)
    
    # Compute stabilization update
    dW_stab = -lr * (y_outer + identity_reg) @ W
    
    # Adaptive scaling: reduce update magnitude when y_t norm is large
    y_norm = torch.norm(y_t_clipped)
    if y_norm > 5.0:
        # Scale down when activations are large to prevent explosion
        scale_factor = 5.0 / y_norm
        dW_stab = dW_stab * scale_factor
    
    return dW_stab
