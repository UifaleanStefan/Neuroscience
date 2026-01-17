"""Learning rule terms and update mechanisms for LPL."""

import torch
import torch.nn.functional as F


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
        del scale_factor  # Free intermediate tensor
    del y_norm, y_t_clipped, y_outer, identity_reg  # Free intermediate tensors
    
    return dW_stab


def hebbian_conv(x_t: torch.Tensor, y_t: torch.Tensor, lr: float,
                 kernel_size: int, stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    Compute the Hebbian learning rule update term for convolutional layers.
    
    For conv layers: ΔW_hebb = η_h * Σ_patches (y_t[patch] @ x_t[patch].T)
    
    Memory-efficient implementation using unfold to extract patches.
    
    Args:
        x_t: Input tensor of shape (in_channels, H, W)
        y_t: Output tensor of shape (out_channels, H_out, W_out)
        lr: Learning rate η_h
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding size
        
    Returns:
        Weight update tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
    """
    assert x_t.dim() == 3, "x_t must be 3D (in_channels, H, W)"
    assert y_t.dim() == 3, "y_t must be 3D (out_channels, H_out, W_out)"
    
    in_channels, H, W = x_t.shape
    out_channels, H_out, W_out = y_t.shape
    
    # Add batch dimension for unfold: (1, in_channels, H, W)
    x_t_4d = x_t.unsqueeze(0)
    
    # Extract patches from input using unfold
    # This creates patches of size (kernel_size, kernel_size) for each output position
    patches = F.unfold(x_t_4d, kernel_size=kernel_size, stride=stride, padding=padding)
    # patches shape: (1, in_channels * kernel_size * kernel_size, num_patches)
    
    num_patches = patches.shape[2]
    patches = patches.squeeze(0)  # (in_channels * kernel_size * kernel_size, num_patches)
    
    # Reshape patches: (num_patches, in_channels, kernel_size, kernel_size)
    patches = patches.view(num_patches, in_channels, kernel_size, kernel_size)
    
    # Reshape y_t for outer product: (out_channels, num_patches)
    # y_t is (out_channels, H_out, W_out), flatten spatial dims
    y_t_flat = y_t.view(out_channels, num_patches)  # (out_channels, num_patches)
    
    # Compute outer product for each patch: y_t[c, p] * x_t[p, :, :, :]
    # For efficiency, we compute: Σ_p (y_t[:, p] @ patches[p].T) for each output channel
    # Shape: (out_channels, in_channels, kernel_size, kernel_size)
    dW_hebb = torch.zeros(out_channels, in_channels, kernel_size, kernel_size, 
                         device=x_t.device, dtype=x_t.dtype)
    
    # Vectorized computation: for each output channel, sum over patches
    for c in range(out_channels):
        # y_t_weights shape: (num_patches,)
        y_weights = y_t_flat[c, :]  # (num_patches,)
        # Weighted sum of patches: (in_channels, kernel_size, kernel_size)
        weighted_patches = torch.einsum('p,pijk->ijk', y_weights, patches)
        dW_hebb[c] = weighted_patches
    
    # Scale by learning rate
    dW_hebb.mul_(lr)
    
    # Clean up intermediate tensors
    del x_t_4d, patches, y_t_flat, y_weights, weighted_patches
    
    return dW_hebb


def predictive_conv(x_t1: torch.Tensor, y_t1: torch.Tensor,
                   y_hat_t1_channel_means: torch.Tensor, lr: float,
                   kernel_size: int, stride: int = 1, padding: int = 0,
                   H_out: int = None, W_out: int = None) -> torch.Tensor:
    """
    Compute the predictive learning rule update term for convolutional layers.
    
    For conv layers: ΔW_pred = η_p * Σ_patches ((y_hat_t1 - y_t1)[patch] @ x_t1[patch].T)
    
    Uses channel-wise prediction error and applies it spatially.
    
    Args:
        x_t1: Input tensor at time t+1 of shape (in_channels, H, W)
        y_t1: Output tensor at time t+1 of shape (out_channels, H_out, W_out)
        y_hat_t1_channel_means: Predicted channel means of shape (out_channels,)
        lr: Learning rate η_p
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding size
        H_out: Output height (if None, inferred from y_t1)
        W_out: Output width (if None, inferred from y_t1)
        
    Returns:
        Weight update tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
    """
    assert x_t1.dim() == 3, "x_t1 must be 3D (in_channels, H, W)"
    assert y_t1.dim() == 3, "y_t1 must be 3D (out_channels, H_out, W_out)"
    assert y_hat_t1_channel_means.dim() == 1, "y_hat_t1_channel_means must be 1D"
    
    in_channels, H, W = x_t1.shape
    out_channels, H_out_actual, W_out_actual = y_t1.shape
    
    if H_out is None:
        H_out = H_out_actual
    if W_out is None:
        W_out = W_out_actual
    
    # Compute channel-wise prediction error
    y_t1_channel_means = y_t1.view(out_channels, -1).mean(dim=1)  # (out_channels,)
    prediction_error = y_hat_t1_channel_means - y_t1_channel_means  # (out_channels,)
    
    # Expand prediction error spatially: (out_channels, H_out, W_out)
    # Broadcast channel error across spatial dimensions
    prediction_error_spatial = prediction_error.view(out_channels, 1, 1).expand(-1, H_out, W_out)
    
    # Extract patches from x_t1
    x_t1_4d = x_t1.unsqueeze(0)  # (1, in_channels, H, W)
    patches = F.unfold(x_t1_4d, kernel_size=kernel_size, stride=stride, padding=padding)
    # patches shape: (1, in_channels * kernel_size * kernel_size, num_patches)
    
    num_patches = patches.shape[2]
    patches = patches.squeeze(0)  # (in_channels * kernel_size * kernel_size, num_patches)
    
    # Reshape patches: (num_patches, in_channels, kernel_size, kernel_size)
    patches = patches.view(num_patches, in_channels, kernel_size, kernel_size)
    
    # Flatten prediction error: (out_channels, num_patches)
    pred_error_flat = prediction_error_spatial.view(out_channels, num_patches)
    
    # Compute weighted sum of patches for each output channel
    dW_pred = torch.zeros(out_channels, in_channels, kernel_size, kernel_size,
                         device=x_t1.device, dtype=x_t1.dtype)
    
    for c in range(out_channels):
        # pred_weights shape: (num_patches,)
        pred_weights = pred_error_flat[c, :]  # (num_patches,)
        # Weighted sum of patches: (in_channels, kernel_size, kernel_size)
        weighted_patches = torch.einsum('p,pijk->ijk', pred_weights, patches)
        dW_pred[c] = weighted_patches
    
    # Scale by learning rate
    dW_pred.mul_(lr)
    
    # Clean up intermediate tensors
    del x_t1_4d, patches, prediction_error, y_t1_channel_means
    del prediction_error_spatial, pred_error_flat, pred_weights, weighted_patches
    
    return dW_pred


def stabilization_conv(y_t: torch.Tensor, W: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Compute the stabilization/decorrelation learning rule update term for conv layers.
    
    For conv layers: ΔW_stab = -η_s * (y_t_means @ y_t_means.T + ε*I) @ W_reshaped
    Applied channel-wise for memory efficiency.
    
    Args:
        y_t: Output tensor of shape (out_channels, H_out, W_out)
        W: Weight tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
        lr: Learning rate η_s
        
    Returns:
        Weight update tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
    """
    assert y_t.dim() == 3, "y_t must be 3D (out_channels, H_out, W_out)"
    assert W.dim() == 4, "W must be 4D (out_channels, in_channels, kernel_size, kernel_size)"
    
    out_channels, H_out, W_out = y_t.shape
    
    # Compute channel-wise means for efficiency
    # y_t shape: (out_channels, H_out, W_out)
    # Channel means: (out_channels,)
    y_t_channel_means = y_t.view(out_channels, -1).mean(dim=1)  # (out_channels,)
    
    # Clip channel means to prevent explosion
    y_t_clipped = torch.clamp(y_t_channel_means, min=-10.0, max=10.0)
    
    # Compute outer product: (out_channels, out_channels)
    y_outer = torch.outer(y_t_clipped, y_t_clipped)
    
    # Add epsilon regularization
    epsilon = 1e-6
    identity_reg = epsilon * torch.eye(out_channels, device=W.device, dtype=W.dtype)
    
    # For conv layers, apply stabilization per output channel
    # Reshape W: (out_channels, in_channels * kernel_size * kernel_size)
    kernel_size = W.shape[2]
    W_reshaped = W.view(out_channels, -1)  # (out_channels, in_channels * k * k)
    
    # Compute stabilization: -lr * (y_outer + identity_reg) @ W_reshaped
    # Shape: (out_channels, in_channels * kernel_size * kernel_size)
    dW_stab_reshaped = -lr * (y_outer + identity_reg) @ W_reshaped
    
    # Reshape back: (out_channels, in_channels, kernel_size, kernel_size)
    dW_stab = dW_stab_reshaped.view(out_channels, -1, kernel_size, kernel_size)
    
    # Adaptive scaling when channel means are large
    y_norm = torch.norm(y_t_clipped)
    if y_norm > 5.0:
        scale_factor = 5.0 / y_norm
        dW_stab.mul_(scale_factor)
    
    # Clean up intermediate tensors
    del y_t_channel_means, y_t_clipped, y_outer, identity_reg
    del W_reshaped, dW_stab_reshaped, y_norm
    
    return dW_stab