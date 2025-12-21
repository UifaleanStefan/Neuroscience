"""Hierarchical 2-layer LPL model implementation."""

import torch
from .lpl_layer import LPLLayer


class HierarchicalLPL:
    """
    2-layer hierarchical LPL model.
    
    First layer: LPL layer receiving flattened images
    Second layer: LPL layer receiving first layer's activations
    Both layers trained using local learning rules (no backpropagation).
    """
    
    def __init__(self, d_in: int, d_hidden: int, d_out: int, cfg_layer1, cfg_layer2):
        """
        Initialize hierarchical LPL model.
        
        Args:
            d_in: Input dimension (e.g., 32*32 = 1024 for flattened images)
            d_hidden: First layer output dimension (hidden representation)
            d_out: Second layer output dimension (final representation)
            cfg_layer1: Configuration for first layer
            cfg_layer2: Configuration for second layer
        """
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        
        # Create first layer
        self.layer1 = LPLLayer(d_in=d_in, d_out=d_hidden, cfg=cfg_layer1)
        
        # Create second layer
        self.layer2 = LPLLayer(d_in=d_hidden, d_out=d_out, cfg=cfg_layer2)
    
    def forward(self, x: torch.Tensor, return_all_layers=False):
        """
        Forward pass through both layers.
        
        Args:
            x: Input tensor of shape (d_in,)
            return_all_layers: If True, returns (y1, y2), else returns y2
            
        Returns:
            If return_all_layers=False: y2 (second layer output)
            If return_all_layers=True: (y1, y2) tuple
        """
        assert x.dim() == 1, "x must be 1D"
        assert x.shape[0] == self.d_in, f"x dimension must match input dimension {self.d_in}"
        
        # First layer forward
        y1 = self.layer1.forward(x)
        
        # Second layer forward (takes first layer activations as input)
        y2 = self.layer2.forward(y1)
        
        if return_all_layers:
            return y1, y2
        else:
            return y2
    
    def update(self, x_t: torch.Tensor, x_t1: torch.Tensor):
        """
        Update both layers using local learning rules.
        
        First layer updates using x_t, x_t1 (input images).
        Second layer updates using y1_t, y1_t1 (first layer activations).
        Updates are independent - no backpropagation.
        
        The key insight: Layer2 updates using activations computed from Layer1's
        CURRENT weights (before Layer1 update), maintaining local learning principles.
        
        Args:
            x_t: Input tensor at time t of shape (d_in,)
            x_t1: Input tensor at time t+1 of shape (d_in,)
        """
        assert x_t.dim() == 1, "x_t must be 1D"
        assert x_t1.dim() == 1, "x_t1 must be 1D"
        assert x_t.shape[0] == self.d_in, f"x_t dimension must match input dimension {self.d_in}"
        assert x_t1.shape[0] == self.d_in, f"x_t1 dimension must match input dimension {self.d_in}"
        
        # Compute first layer activations using current weights
        y1_t = self.layer1.forward(x_t)
        y1_t1 = self.layer1.forward(x_t1)
        
        # Update first layer using input images
        # Note: layer1.update() will compute activations again internally,
        # but we need y1_t and y1_t1 for layer2 update
        self.layer1.update(x_t, x_t1)
        
        # Update second layer using first layer activations
        # These activations were computed from layer1's weights before the update,
        # which is correct for local learning (layer2 learns from what it received)
        self.layer2.update(y1_t, y1_t1)
    
    def get_activations(self, x: torch.Tensor):
        """
        Get activations from both layers for a given input.
        
        Args:
            x: Input tensor of shape (d_in,)
            
        Returns:
            Tuple of (y1, y2) where:
            - y1: First layer activations of shape (d_hidden,)
            - y2: Second layer activations of shape (d_out,)
        """
        return self.forward(x, return_all_layers=True)

