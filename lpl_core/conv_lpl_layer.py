"""Convolutional LPL layer implementation."""

import torch
import torch.nn.functional as F
from .rules import hebbian_conv, predictive_conv, stabilization_conv
from .predictor import Predictor


class ConvLPLLayer:
    """
    Single convolutional layer trained with Local Predictive Learning (LPL) rules.
    
    This layer learns spatial features from images using local learning rules
    (Hebbian, predictive, stabilization) without backpropagation or autograd.
    
    Memory-safe design:
    - Processes patches efficiently to avoid large intermediate tensors
    - Clears intermediate activations explicitly
    - Uses in-place operations where possible
    - Avoids autograd tracking
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, cfg=None):
        """
        Initialize convolutional LPL layer.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output feature maps (filters)
            kernel_size: Convolution kernel size (assumed square, e.g., 3 for 3x3)
            stride: Convolution stride (default: 1)
            padding: Padding size (default: 0)
            cfg: Configuration object with learning rates and flags:
                - cfg.lr_hebb: Learning rate for Hebbian term
                - cfg.lr_pred: Learning rate for predictive term
                - cfg.lr_stab: Learning rate for stabilization term
                - cfg.use_hebb: Boolean flag to enable Hebbian term
                - cfg.use_pred: Boolean flag to enable predictive term
                - cfg.use_stab: Boolean flag to enable stabilization term
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cfg = cfg
        
        # Initialize conv weights: (out_channels, in_channels, kernel_size, kernel_size)
        # Small random initialization (no autograd tracking)
        self.W = torch.randn(out_channels, in_channels, kernel_size, kernel_size, 
                            requires_grad=False) * 0.01
        
        # Initialize predictor for each output channel
        # Predictor dimension = out_channels (predicts future activations)
        self.predictor = Predictor(out_channels, cfg.lr_pred if cfg else 0.001)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional layer.
        
        Args:
            x: Input tensor of shape (in_channels, H, W) or (H, W)
            
        Returns:
            Output tensor of shape (out_channels, H_out, W_out) with ReLU activation
        """
        # Ensure x is 3D: (in_channels, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add channel dimension: (1, H, W)
        elif x.dim() == 1:
            # If flattened, reshape to 2D then add channel
            size = int(x.shape[0] ** 0.5)
            x = x.reshape(1, size, size)
        
        assert x.dim() == 3, f"x must be 2D or 3D, got {x.dim()}D"
        assert x.shape[0] == self.in_channels, \
            f"Input channels {x.shape[0]} must match layer in_channels {self.in_channels}"
        
        # Manual convolution for memory efficiency and to avoid autograd
        # We'll use F.conv2d with requires_grad=False to prevent autograd tracking
        x_4d = x.unsqueeze(0)  # Add batch dimension: (1, in_channels, H, W)
        
        with torch.no_grad():
            # Perform convolution
            conv_out = F.conv2d(
                x_4d,
                self.W,
                stride=self.stride,
                padding=self.padding
            )  # Shape: (1, out_channels, H_out, W_out)
        
        # Remove batch dimension
        conv_out = conv_out.squeeze(0)  # Shape: (out_channels, H_out, W_out)
        
        # Apply ReLU activation (non-negative)
        conv_out = torch.relu(conv_out)
        
        # Clean up intermediate tensor
        del x_4d
        
        return conv_out
    
    def update(self, x_t: torch.Tensor, x_t1: torch.Tensor) -> None:
        """
        Update convolutional layer weights using local learning rules.
        
        Implements LPL rules adapted for convolutional layers:
        - Hebbian: Strengthens connections between co-active input/output patches
        - Predictive: Learns to predict future activations
        - Stabilization: Prevents representational collapse
        
        Memory-safe implementation:
        - Processes spatial dimensions efficiently
        - Clears intermediate tensors explicitly
        - Uses in-place operations where possible
        
        Args:
            x_t: Input tensor at time t of shape (in_channels, H, W) or (H, W)
            x_t1: Input tensor at time t+1 of shape (in_channels, H, W) or (H, W)
        """
        # Ensure inputs are 3D: (in_channels, H, W)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        if x_t1.dim() == 2:
            x_t1 = x_t1.unsqueeze(0)
        
        # Compute activations (before weight update)
        y_t = self.forward(x_t)  # Shape: (out_channels, H_out, W_out)
        y_t1 = self.forward(x_t1)
        
        # Check for NaN
        if torch.isnan(y_t).any() or torch.isnan(y_t1).any():
            import warnings
            warnings.warn("NaN detected in conv activations. Skipping update.")
            return
        
        # Get prediction: y_hat_t1 = P @ y_t_flat
        # Flatten spatial dimensions for predictor
        y_t_flat = y_t.flatten()  # Shape: (out_channels * H_out * W_out,)
        
        # For predictor, use channel-wise aggregation (mean pooling across spatial dims)
        # This reduces predictor dimension to out_channels
        H_out, W_out = y_t.shape[1], y_t.shape[2]
        y_t_channel_means = y_t.view(self.out_channels, -1).mean(dim=1)  # (out_channels,)
        y_t1_channel_means = y_t1.view(self.out_channels, -1).mean(dim=1)  # (out_channels,)
        
        y_hat_t1_channel_means = self.predictor.forward(y_t_channel_means)
        
        # Check for NaN in prediction
        if torch.isnan(y_hat_t1_channel_means).any():
            import warnings
            warnings.warn("NaN detected in conv prediction. Skipping update.")
            del y_t, y_t1, y_t_flat, y_t_channel_means, y_t1_channel_means, y_hat_t1_channel_means
            return
        
        # Initialize weight update tensor
        dW = torch.zeros_like(self.W)
        
        # Apply Hebbian term if enabled
        if self.cfg.use_hebb:
            dW_hebb = hebbian_conv(x_t, y_t, self.cfg.lr_hebb, 
                                  self.kernel_size, self.stride, self.padding)
            if torch.isnan(dW_hebb).any():
                import warnings
                warnings.warn("NaN detected in Hebbian conv update. Skipping this term.")
            else:
                dW.add_(dW_hebb)
                del dW_hebb
        
        # Apply predictive term if enabled
        if self.cfg.use_pred:
            dW_pred = predictive_conv(x_t1, y_t1, y_hat_t1_channel_means, 
                                     self.cfg.lr_pred, self.kernel_size, 
                                     self.stride, self.padding, H_out, W_out)
            if torch.isnan(dW_pred).any():
                import warnings
                warnings.warn("NaN detected in predictive conv update. Skipping this term.")
            else:
                dW.add_(dW_pred)
                del dW_pred
        
        # Apply stabilization term if enabled
        if self.cfg.use_stab:
            dW_stab = stabilization_conv(y_t, self.W, self.cfg.lr_stab)
            if torch.isnan(dW_stab).any():
                import warnings
                warnings.warn("NaN detected in stabilization conv update. Skipping this term.")
            else:
                dW.add_(dW_stab)
                del dW_stab
        
        # Normalize update to prevent large jumps
        max_update_norm = 1.0
        dW_norm = torch.norm(dW)
        if dW_norm > max_update_norm:
            dW.mul_(max_update_norm / dW_norm)
        
        # Check for NaN in combined update
        if torch.isnan(dW).any():
            import warnings
            warnings.warn("NaN detected in combined conv weight update. Skipping update.")
            del dW, y_t, y_t1, y_t_flat, y_t_channel_means, y_t1_channel_means, y_hat_t1_channel_means
            return
        
        # Update weights
        self.W.add_(dW)
        del dW
        
        # Clip weights to safe range (in-place)
        self.W.clamp_(min=-5.0, max=5.0)
        
        # Update predictor
        self.predictor.update(y_t_channel_means, y_t1_channel_means)
        
        # Clean up intermediate tensors
        del y_t, y_t1, y_t_flat, y_t_channel_means, y_t1_channel_means, y_hat_t1_channel_means
        
        # Final NaN check
        if torch.isnan(self.W).any():
            import warnings
            warnings.warn("NaN detected in conv weights after update.")
