"""Hierarchical 3-layer LPL model implementation."""

import torch
from .lpl_layer import LPLLayer


class HierarchicalLPL3Layer:
    """
    3-layer hierarchical LPL model.
    
    Layer 1: LPL layer receiving flattened images (d_in -> d_hidden1)
    Layer 2: LPL layer receiving first layer's activations (d_hidden1 -> d_hidden2)
    Layer 3: LPL layer receiving second layer's activations (d_hidden2 -> d_out)
    All layers trained using local learning rules (no backpropagation).
    """
    
    def __init__(self, d_in: int, d_hidden1: int, d_hidden2: int, d_out: int, 
                 cfg_layer1, cfg_layer2, cfg_layer3):
        """
        Initialize hierarchical 3-layer LPL model.
        
        Args:
            d_in: Input dimension (e.g., 32*32 = 1024 for flattened images)
            d_hidden1: First layer output dimension
            d_hidden2: Second layer output dimension
            d_out: Third layer output dimension (final representation)
            cfg_layer1: Configuration for first layer
            cfg_layer2: Configuration for second layer
            cfg_layer3: Configuration for third layer
        """
        self.d_in = d_in
        self.d_hidden1 = d_hidden1
        self.d_hidden2 = d_hidden2
        self.d_out = d_out
        
        # Create layers
        self.layer1 = LPLLayer(d_in=d_in, d_out=d_hidden1, cfg=cfg_layer1)
        self.layer2 = LPLLayer(d_in=d_hidden1, d_out=d_hidden2, cfg=cfg_layer2)
        self.layer3 = LPLLayer(d_in=d_hidden2, d_out=d_out, cfg=cfg_layer3)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through all three layers.
        
        Args:
            x: Input tensor of shape (d_in,)
            
        Returns:
            y3 (third layer output) of shape (d_out,)
        """
        assert x.dim() == 1, "x must be 1D"
        assert x.shape[0] == self.d_in, f"x dimension must match input dimension {self.d_in}"
        
        # Forward through all layers
        y1 = self.layer1.forward(x)
        y2 = self.layer2.forward(y1)
        y3 = self.layer3.forward(y2)
        
        return y3
    
    def update(self, x_t: torch.Tensor, x_t1: torch.Tensor):
        """
        Update all three layers using local learning rules.
        
        Layer 1 updates using x_t, x_t1 (input images).
        Layer 2 updates using y1_t, y1_t1 (first layer activations).
        Layer 3 updates using y2_t, y2_t1 (second layer activations).
        Updates are independent - no backpropagation.
        
        Args:
            x_t: Input tensor at time t of shape (d_in,)
            x_t1: Input tensor at time t+1 of shape (d_in,)
        """
        assert x_t.dim() == 1, "x_t must be 1D"
        assert x_t1.dim() == 1, "x_t1 must be 1D"
        assert x_t.shape[0] == self.d_in, f"x_t dimension must match input dimension {self.d_in}"
        assert x_t1.shape[0] == self.d_in, f"x_t1 dimension must match input dimension {self.d_in}"
        
        # Compute activations from each layer using current weights
        y1_t = self.layer1.forward(x_t)
        y1_t1 = self.layer1.forward(x_t1)
        
        y2_t = self.layer2.forward(y1_t)
        y2_t1 = self.layer2.forward(y1_t1)
        
        # Update layers sequentially
        self.layer1.update(x_t, x_t1)
        self.layer2.update(y1_t, y1_t1)
        self.layer3.update(y2_t, y2_t1)




