"""Single convolutional layer LPL model."""

import torch
from .conv_lpl_layer import ConvLPLLayer


class ConvLPLModel:
    """
    Single convolutional layer LPL model.
    
    This model contains only a single convolutional layer trained with LPL rules.
    No MLP head - just the convolutional layer outputs are used as representations.
    
    Memory-safe implementation:
    - Uses ConvLPLLayer which handles memory-efficient updates
    - No autograd tracking
    - Explicit tensor cleanup
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 input_size: int, stride: int = 1, padding: int = 0, cfg=None,
                 use_pooling: bool = False, pool_kernel_size: int = 2):
        """
        Initialize single conv layer LPL model.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output feature maps
            kernel_size: Convolution kernel size (assumed square)
            input_size: Input image size (e.g., 32 for 32x32)
            stride: Convolution stride (default: 1)
            padding: Padding size (default: 0)
            cfg: Configuration object with learning rates and flags
            use_pooling: If True, apply max pooling after conv (default: False)
            pool_kernel_size: Max pooling kernel size (default: 2)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.stride = stride
        self.padding = padding
        self.use_pooling = use_pooling
        self.pool_kernel_size = pool_kernel_size
        
        # Create convolutional LPL layer
        self.conv = ConvLPLLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            cfg=cfg
        )
        
        # Compute output spatial dimensions
        H_out = (input_size + 2 * padding - kernel_size) // stride + 1
        W_out = (input_size + 2 * padding - kernel_size) // stride + 1
        
        if use_pooling:
            H_out = H_out // pool_kernel_size
            W_out = W_out // pool_kernel_size
        
        self.output_shape = (out_channels, H_out, W_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional layer.
        
        Args:
            x: Input tensor of shape (in_channels, H, W) or (H, W) or flattened
            
        Returns:
            Output tensor of shape (out_channels, H_out, W_out) after ReLU (and pooling if enabled)
        """
        # Ensure x is 3D: (in_channels, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add channel dimension
        elif x.dim() == 1:
            # If flattened, reshape to 2D then add channel
            size = int(x.shape[0] ** 0.5)
            x = x.reshape(1, size, size)
        
        # Forward through conv layer (includes ReLU)
        y = self.conv.forward(x)  # Shape: (out_channels, H_out, W_out)
        
        # Apply max pooling if enabled
        if self.use_pooling:
            # Add batch dimension for pooling: (1, out_channels, H_out, W_out)
            y_4d = y.unsqueeze(0)
            y_pooled = torch.nn.functional.max_pool2d(
                y_4d, 
                kernel_size=self.pool_kernel_size, 
                stride=self.pool_kernel_size
            )
            y = y_pooled.squeeze(0)  # Remove batch dimension
            del y_4d, y_pooled  # Clean up intermediate tensors
        
        return y
    
    def update(self, x_t: torch.Tensor, x_t1: torch.Tensor) -> None:
        """
        Update convolutional layer using local learning rules.
        
        Args:
            x_t: Input tensor at time t of shape (in_channels, H, W) or (H, W) or flattened
            x_t1: Input tensor at time t+1 of shape (in_channels, H, W) or (H, W) or flattened
        """
        # Ensure inputs are 3D: (in_channels, H, W)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        if x_t1.dim() == 2:
            x_t1 = x_t1.unsqueeze(0)
        elif x_t.dim() == 1:
            size = int(x_t.shape[0] ** 0.5)
            x_t = x_t.reshape(1, size, size)
            x_t1 = x_t1.reshape(1, size, size)
        
        # Update conv layer using LPL rules
        self.conv.update(x_t, x_t1)
    
    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get activations (outputs) for a given input.
        
        Args:
            x: Input tensor of shape (in_channels, H, W) or (H, W) or flattened
            
        Returns:
            Output tensor of shape (out_channels, H_out, W_out)
        """
        return self.forward(x)
