"""Conv-MLP Hybrid LPL model implementation."""

import torch
import torch.nn as nn
from .lpl_layer import LPLLayer


class ConvMLPHybrid:
    """
    Conv-MLP Hybrid LPL model.
    
    Architecture: Conv layer → Flatten → LPL Linear layer(s)
    The convolutional layer extracts spatial features from 2D images.
    The linear LPL layer(s) process the flattened conv features.
    
    For this implementation, we use a single conv layer followed by a single LPL linear layer.
    The conv layer can be treated as feature extraction, and the LPL layer is trained with
    local learning rules.
    """
    
    def __init__(self, input_channels: int, input_size: int, 
                 conv_out_channels: int, conv_kernel_size: int,
                 d_out: int, cfg_linear, use_pooling=True):
        """
        Initialize Conv-MLP Hybrid model.
        
        Args:
            input_channels: Number of input channels (1 for grayscale)
            input_size: Input image size (e.g., 32 for 32x32)
            conv_out_channels: Number of output channels from conv layer
            conv_kernel_size: Kernel size for conv layer
            d_out: Output dimension of linear LPL layer (final representation)
            cfg_linear: Configuration for the linear LPL layer
            use_pooling: If True, use max pooling to reduce spatial dimensions (default: True)
        """
        self.input_channels = input_channels
        self.input_size = input_size
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.use_pooling = use_pooling
        
        # Create convolutional layer
        # Using standard conv with padding to preserve spatial dimensions
        padding = conv_kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding=padding
        )
        
        # Add pooling to reduce spatial dimensions (if enabled)
        if use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dim by 2
            pooled_size = input_size // 2
        else:
            self.pool = None
            pooled_size = input_size
        
        # Calculate flattened conv output size
        conv_output_size = conv_out_channels * pooled_size * pooled_size
        
        # Create linear LPL layer
        self.linear = LPLLayer(d_in=conv_output_size, d_out=d_out, cfg=cfg_linear)
        
        self.d_out = d_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv and linear layers.
        
        Args:
            x: Input tensor of shape (H, W) for grayscale image, or (C, H, W)
            
        Returns:
            Output tensor from linear LPL layer of shape (d_out,)
        """
        # Ensure x is 3D: (C, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add channel dimension: (1, H, W)
        elif x.dim() == 1:
            # If flattened, reshape to 2D then add channel
            size = int(x.shape[0] ** 0.5)
            x = x.reshape(1, size, size)
        
        # Conv forward
        conv_out = self.conv(x)  # (C_out, H, W)
        
        # Apply ReLU activation to conv output (standard for conv layers)
        conv_out = torch.relu(conv_out)
        
        # Apply pooling if enabled
        if self.pool is not None:
            conv_out = self.pool(conv_out)
        
        # Flatten conv output
        conv_flat = conv_out.flatten()  # (C_out * H * W,)
        
        # Linear LPL layer forward (includes tanh activation)
        output = self.linear.forward(conv_flat)
        
        return output
    
    def update(self, x_t: torch.Tensor, x_t1: torch.Tensor):
        """
        Update model using local learning rules.
        
        Updates the linear LPL layer using Full LPL rules.
        The conv layer acts as feature extraction (not updated with LPL rules).
        
        Args:
            x_t: Input tensor at time t (2D image of shape (H, W) or flattened)
            x_t1: Input tensor at time t+1 (2D image of shape (H, W) or flattened)
        """
        # Process inputs through conv layer to get features
        # Handle different input formats
        if x_t.dim() == 1:
            # Flattened input - reshape to 2D
            size = int(x_t.shape[0] ** 0.5)
            x_t = x_t.reshape(size, size)
            x_t1 = x_t1.reshape(size, size)
        
        # Ensure x_t and x_t1 are 3D: (C, H, W)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)  # (1, H, W)
            x_t1 = x_t1.unsqueeze(0)  # (1, H, W)
        
        # Forward through conv (with ReLU)
        conv_t = torch.relu(self.conv(x_t))
        conv_t1 = torch.relu(self.conv(x_t1))
        
        # Apply pooling if enabled
        if self.pool is not None:
            conv_t = self.pool(conv_t)
            conv_t1 = self.pool(conv_t1)
        
        # Flatten conv outputs
        conv_flat_t = conv_t.flatten()
        conv_flat_t1 = conv_t1.flatten()
        
        # Clear intermediate conv tensors to save GPU memory (after flattening)
        del conv_t, conv_t1
        
        # Update linear LPL layer using local learning rules (Full LPL)
        self.linear.update(conv_flat_t, conv_flat_t1)
        
        # Clear flattened tensors (after update is complete)
        del conv_flat_t, conv_flat_t1

