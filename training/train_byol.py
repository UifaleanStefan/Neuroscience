"""Minimal BYOL implementation with local learning (first layer only)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SimpleConvNet(nn.Module):
    """
    Small convolutional network with projection head for BYOL.
    
    Architecture:
    - Conv layers: 3x3 conv -> 5x5 conv -> 3x3 conv
    - Global average pooling
    - Projection head: Linear -> BN -> ReLU -> Linear (to 128 dims)
    """
    
    def __init__(self, projection_dim=128):
        super().__init__()
        
        # Convolutional layers (only first layer will be updated)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # This layer gets updated
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # Frozen
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Frozen
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )
        
        # Freeze all layers except conv1 (for local learning)
        self._freeze_layers_except_first()
    
    def _freeze_layers_except_first(self):
        """Freeze all layers except the first convolutional layer."""
        # Freeze conv2, conv3, and projection head
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.proj_head.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.bn3.parameters():
            param.requires_grad = False
    
    def forward(self, x, return_projection=True):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
            return_projection: If True, return projection; else return conv features
            
        Returns:
            If return_projection=True: projection of shape (B, projection_dim)
            If return_projection=False: features after conv3 of shape (B, 256)
        """
        # First conv layer (this is the only one that updates)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv layer (frozen)
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Third conv layer (frozen)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        if return_projection:
            # Projection head (frozen)
            x = self.proj_head(x)
            # L2 normalize
            x = F.normalize(x, p=2, dim=1)
        
        return x


class BYOLAugmentation:
    """
    Simple augmentation for BYOL training.
    
    Applies:
    - Random crop with padding
    - Random horizontal flip
    - Color jitter (brightness, contrast, saturation)
    - Random translation
    """
    
    def __init__(self, image_size=32):
        self.image_size = image_size
    
    def __call__(self, image):
        """
        Apply augmentations to an image.
        
        Args:
            image: Tensor of shape (C, 32, 32) in range [0, 1]
            
        Returns:
            Augmented image tensor
        """
        # Random crop with padding
        padding = 4
        # Pad: (pad_left, pad_right, pad_top, pad_bottom) for 2D
        padded = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        
        # Random crop using slicing
        top = torch.randint(0, padding * 2, (1,)).item()
        left = torch.randint(0, padding * 2, (1,)).item()
        image = padded[:, top:top+self.image_size, left:left+self.image_size]
        
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, [2])
        
        # Color jitter (simple version)
        # Brightness
        brightness_factor = 0.8 + 0.4 * torch.rand(1).item()  # [0.8, 1.2]
        image = image * brightness_factor
        
        # Contrast
        contrast_factor = 0.8 + 0.4 * torch.rand(1).item()  # [0.8, 1.2]
        mean = image.mean()
        image = (image - mean) * contrast_factor + mean
        
        # Clamp to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)
        
        return image


def byol_loss(q, z):
    """
    BYOL loss function (MSE between normalized projections).
    
    Args:
        q: Online network projection of shape (B, D)
        z: Target network projection of shape (B, D) (stop gradients)
        
    Returns:
        Loss value
    """
    # Both should already be L2 normalized
    # Compute MSE loss
    loss = F.mse_loss(q, z.detach())  # Stop gradients on target
    return loss


def update_target_network(online_net, target_net, tau=0.99):
    """
    Update target network using exponential moving average.
    
    Args:
        online_net: Online network
        target_net: Target network
        tau: EMA coefficient (0.99 typical for BYOL)
    """
    with torch.no_grad():
        # Only update frozen layers (for BYOL, we update all)
        # But since we're doing local learning, only conv1 should change
        # So we only copy conv1 parameters
        for online_param, target_param in zip(online_net.conv1.parameters(), target_net.conv1.parameters()):
            target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)


def export_embeddings(model, dataset, num_samples=1000, batch_size=32, device='cpu', is_grayscale=False):
    """
    Export embeddings from the model.
    
    Args:
        model: The network model
        dataset: Dataset to extract embeddings from
        num_samples: Number of samples to export
        batch_size: Batch size for processing
        device: Device to use
        is_grayscale: If True, expects grayscale images (1 channel)
        
    Returns:
        Dictionary with 'embeddings' and 'labels' tensors
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            if hasattr(dataset, 'return_temporal_pair') and dataset.return_temporal_pair:
                image, _, label = dataset[i]
            else:
                image, label = dataset[i]
            
            # Convert to tensor if needed
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, dtype=torch.float32)
            
            # Handle different image formats
            if is_grayscale:
                # Grayscale: should be (H, W) or (1, H, W)
                if image.dim() == 2:
                    image = image.unsqueeze(0)  # (1, H, W)
                elif image.dim() == 3 and image.shape[0] != 1:
                    # Assume (H, W, 1) or need to adjust
                    if image.shape[2] == 1:
                        image = image.permute(2, 0, 1)
                    else:
                        image = image[0:1, :, :]  # Take first channel
            else:
                # RGB: should be (3, H, W)
                if image.dim() == 2:  # Grayscale, convert to RGB
                    image = image.unsqueeze(0).repeat(3, 1, 1)
                elif image.dim() == 3:
                    if image.shape[0] == 1:  # Single channel
                        image = image.repeat(3, 1, 1)
                    elif image.shape[0] != 3:  # (H, W, C) format
                        if image.shape[2] == 3:
                            image = image.permute(2, 0, 1)
                        else:
                            image = image[0:3, :, :]
            
            # Ensure correct size and range
            if image.shape[-2:] != (32, 32):
                image = F.interpolate(image.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
            
            image = torch.clamp(image, 0.0, 1.0)
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Forward pass (get projection, not features)
            emb = model(image, return_projection=True)
            embeddings.append(emb.cpu())
            labels.append(label)
    
    model.train()
    
    return {
        'embeddings': torch.cat(embeddings, dim=0),
        'labels': torch.tensor(labels)
    }


def train_byol_cifar10(num_steps=1000, batch_size=32, projection_dim=128, device='cpu'):
    """
    Train BYOL on CIFAR-10 with local learning (first layer only).
    
    Args:
        num_steps: Number of training steps
        batch_size: Batch size
        projection_dim: Dimension of projection head output
        device: Device to use
    """
    import torchvision
    import torchvision.transforms as transforms
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create networks
    online_net = SimpleConvNet(projection_dim=projection_dim).to(device)
    target_net = SimpleConvNet(projection_dim=projection_dim).to(device)
    
    # Initialize target network with online network weights
    target_net.load_state_dict(online_net.state_dict())
    
    # Freeze target network completely
    for param in target_net.parameters():
        param.requires_grad = False
    
    # Optimizer (only updates conv1 parameters due to freezing)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, online_net.parameters()),
        lr=1e-3
    )
    
    # Load CIFAR-10
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Create augmentation
    augment = BYOLAugmentation(image_size=32)
    
    # Training logs
    logs = {'loss': [], 'step': []}
    
    print(f"Training BYOL for {num_steps} steps...")
    print(f"Only first conv layer will be updated (local learning)")
    
    # Export embeddings before training
    print("\nExporting embeddings before training...")
    embeddings_before = export_embeddings(online_net, trainset, num_samples=1000, device=device, is_grayscale=False)
    
    output_dir = Path('outputs/activations')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_before, output_dir / 'byol_embeddings_before.pt')
    print(f"Saved embeddings to {output_dir / 'byol_embeddings_before.pt'}")
    
    # Training loop
    online_net.train()
    target_net.train()  # Set to train mode for BN, but parameters are frozen
    
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        
        # Sample batch
        indices = torch.randint(0, len(trainset), (batch_size,))
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            image, label = trainset[idx.item()]
            batch_images.append(image)
            batch_labels.append(label)
        
        batch_images = torch.stack(batch_images).to(device)
        
        # Create two augmented views
        view1 = torch.stack([augment(img) for img in batch_images]).to(device)
        view2 = torch.stack([augment(img) for img in batch_images]).to(device)
        
        # Forward pass through online network
        q1 = online_net(view1, return_projection=True)
        q2 = online_net(view2, return_projection=True)
        
        # Forward pass through target network (no gradients)
        with torch.no_grad():
            z1 = target_net(view1, return_projection=True)
            z2 = target_net(view2, return_projection=True)
        
        # BYOL loss (symmetric)
        loss1 = byol_loss(q1, z2)
        loss2 = byol_loss(q2, z1)
        loss = (loss1 + loss2) / 2.0
        
        # Backward pass (only updates conv1)
        loss.backward()
        optimizer.step()
        
        # Update target network (EMA on conv1 only)
        update_target_network(online_net, target_net, tau=0.99)
        
        # Log loss
        logs['loss'].append(loss.item())
        logs['step'].append(step)
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f}")
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"WARNING: NaN loss at step {step}!")
            break
    
    print("Training completed.")
    
    # Save training logs
    logs_file = output_dir / 'byol_training_logs.json'
    with open(logs_file, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"Saved training logs to {logs_file}")
    
    # Export embeddings after training
    print("\nExporting embeddings after training...")
    embeddings_after = export_embeddings(online_net, trainset, num_samples=1000, device=device, is_grayscale=False)
    torch.save(embeddings_after, output_dir / 'byol_embeddings_after.pt')
    print(f"Saved embeddings to {output_dir / 'byol_embeddings_after.pt'}")
    
    # Print statistics
    print(f"\nEmbedding stats after training:")
    print(f"  Mean: {embeddings_after['embeddings'].mean().item():.6f}")
    print(f"  Std: {embeddings_after['embeddings'].std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(embeddings_after['embeddings']).any().item()}")


def train_byol_synthetic_shapes(num_steps=1000, batch_size=32, projection_dim=128, device='cpu'):
    """
    Train BYOL on synthetic shapes with local learning (first layer only).
    
    Args:
        num_steps: Number of training steps
        batch_size: Batch size
        projection_dim: Dimension of projection head output
        device: Device to use
    """
    from data.synthetic_shapes import SyntheticShapesDataset
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create networks
    online_net = SimpleConvNet(projection_dim=projection_dim).to(device)
    target_net = SimpleConvNet(projection_dim=projection_dim).to(device)
    
    # Initialize target network with online network weights
    target_net.load_state_dict(online_net.state_dict())
    
    # Freeze target network completely
    for param in target_net.parameters():
        param.requires_grad = False
    
    # Modify network for grayscale input (synthetic shapes are grayscale)
    # Replace first conv to accept 1 channel instead of 3
    online_net.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1).to(device)
    online_net.bn1 = nn.BatchNorm2d(64).to(device)
    target_net.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1).to(device)
    target_net.bn1 = nn.BatchNorm2d(64).to(device)
    
    # Re-freeze layers
    online_net._freeze_layers_except_first()
    
    # Optimizer (only updates conv1 parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, online_net.parameters()),
        lr=1e-3
    )
    
    # Load synthetic shapes
    trainset = SyntheticShapesDataset(num_samples=5000, seed=42, return_temporal_pair=False)
    
    # Create augmentation
    augment = BYOLAugmentation(image_size=32)
    
    # Training logs
    logs = {'loss': [], 'step': []}
    
    print(f"Training BYOL on synthetic shapes for {num_steps} steps...")
    print(f"Only first conv layer will be updated (local learning)")
    
    # Export embeddings before training
    print("\nExporting embeddings before training...")
    embeddings_before = export_embeddings(online_net, trainset, num_samples=1000, device=device, is_grayscale=True)
    
    output_dir = Path('outputs/activations')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_before, output_dir / 'byol_shapes_embeddings_before.pt')
    print(f"Saved embeddings to {output_dir / 'byol_shapes_embeddings_before.pt'}")
    
    # Training loop
    online_net.train()
    target_net.train()
    
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        
        # Sample batch
        indices = torch.randint(0, len(trainset), (batch_size,))
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            image, label = trainset[idx.item()]
            # Convert grayscale to 3-channel (for augmentation compatibility)
            if image.dim() == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)
            batch_images.append(image)
            batch_labels.append(label)
        
        batch_images = torch.stack(batch_images).to(device)
        
        # Create two augmented views
        view1 = torch.stack([augment(img) for img in batch_images]).to(device)
        view2 = torch.stack([augment(img) for img in batch_images]).to(device)
        
        # Convert back to grayscale for network input (take first channel)
        view1_gray = view1[:, 0:1, :, :]  # (B, 1, H, W)
        view2_gray = view2[:, 0:1, :, :]
        
        # Forward pass through online network
        q1 = online_net(view1_gray, return_projection=True)
        q2 = online_net(view2_gray, return_projection=True)
        
        # Forward pass through target network (no gradients)
        with torch.no_grad():
            z1 = target_net(view1_gray, return_projection=True)
            z2 = target_net(view2_gray, return_projection=True)
        
        # BYOL loss (symmetric)
        loss1 = byol_loss(q1, z2)
        loss2 = byol_loss(q2, z1)
        loss = (loss1 + loss2) / 2.0
        
        # Backward pass (only updates conv1)
        loss.backward()
        optimizer.step()
        
        # Update target network (EMA on conv1 only)
        update_target_network(online_net, target_net, tau=0.99)
        
        # Log loss
        logs['loss'].append(loss.item())
        logs['step'].append(step)
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f}")
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"WARNING: NaN loss at step {step}!")
            break
    
    print("Training completed.")
    
    # Save training logs
    logs_file = output_dir / 'byol_shapes_training_logs.json'
    with open(logs_file, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"Saved training logs to {logs_file}")
    
    # Export embeddings after training
    print("\nExporting embeddings after training...")
    embeddings_after = export_embeddings(online_net, trainset, num_samples=1000, device=device, is_grayscale=True)
    torch.save(embeddings_after, output_dir / 'byol_shapes_embeddings_after.pt')
    print(f"Saved embeddings to {output_dir / 'byol_shapes_embeddings_after.pt'}")
    
    # Print statistics
    print(f"\nEmbedding stats after training:")
    print(f"  Mean: {embeddings_after['embeddings'].mean().item():.6f}")
    print(f"  Std: {embeddings_after['embeddings'].std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(embeddings_after['embeddings']).any().item()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BYOL with local learning')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'shapes'],
                       help='Dataset to use')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    if args.dataset == 'cifar10':
        train_byol_cifar10(num_steps=args.steps, batch_size=args.batch_size, device=args.device)
    else:
        train_byol_synthetic_shapes(num_steps=args.steps, batch_size=args.batch_size, device=args.device)

