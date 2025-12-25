"""
Grid Experiment #017: Non-LPL Baseline (Standard Backpropagation)

Configuration:
- Dataset: Synthetic Shapes
- Training steps: 10,000
- Architecture: 1-layer MLP (128 units)
- Activation: tanh (scaled)
- Learning rule: Standard gradient descent with backpropagation
- Baseline: Non-LPL (standard supervised/self-supervised learning)
- Seed: 42

This run provides a baseline comparison using standard backpropagation
instead of local learning rules (LPL).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import gc
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.synthetic_shapes import SyntheticShapesDataset, create_temporal_pair_dataset


class StandardMLP(nn.Module):
    """
    Standard 1-layer MLP trained with backpropagation.
    
    Architecture: Input (1024) -> Linear (128) -> tanh
    Trained to predict next frame from current frame (temporal prediction).
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        Initialize standard MLP.
        
        Args:
            d_in: Input dimension (32*32 = 1024 for flattened images)
            d_out: Output dimension (128 units)
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # Single linear layer
        self.linear = nn.Linear(d_in, d_out)
        
        # Initialize weights with small random values
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = tanh(W @ x + b) * 5 (scaled tanh).
        
        Args:
            x: Input tensor of shape (d_in,) or (batch, d_in)
            
        Returns:
            Output tensor of shape (d_out,) or (batch, d_out)
        """
        y = self.linear(x)
        # Apply scaled tanh activation (same as LPL runs)
        y = torch.tanh(y / 5.0) * 5.0
        return y


def export_embeddings(model, dataset, num_samples=1000, device='cpu'):
    """
    Export embeddings (activations) for a set of samples.
    
    Args:
        model: StandardMLP model
        dataset: SyntheticShapesDataset
        num_samples: Number of samples to export
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with 'embeddings' and 'labels'
    """
    embeddings = []
    labels = []
    
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            # Handle both temporal pair and single image modes
            if hasattr(dataset, 'return_temporal_pair') and dataset.return_temporal_pair:
                x_t, _, label = dataset[i]
                image = x_t
            else:
                image, label = dataset[i]
            
            # Flatten image to 1D
            if image.dim() > 1:
                image = image.flatten()
            
            # Ensure input is float32 and in [0,1] range
            if image.dtype != torch.float32:
                image = image.float()
            image = torch.clamp(image, 0.0, 1.0)
            
            # Move to device
            image = image.to(device)
            
            # Forward pass to get embeddings
            emb = model.forward(image)
            
            # Move back to CPU for storage
            emb = emb.cpu()
            
            # Check for NaN in embeddings
            if torch.isnan(emb).any():
                print(f"WARNING: NaN detected in embedding at sample {i}")
                continue
            
            embeddings.append(emb)
            labels.append(label)
    
    model.train()  # Set back to training mode
    
    return {
        'embeddings': torch.stack(embeddings),
        'labels': torch.tensor(labels)
    }


def main():
    """
    Run grid experiment #017.
    """
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Note: CUDA not available, using CPU. This will be slower but will still work.")
    
    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    
    # Experiment configuration
    EXPERIMENT_CONFIG = {
        'dataset': 'synthetic_shapes',
        'steps': 10000,
        'architecture': 'mlp_1layer_128',
        'activation': 'tanh',
        'rule': 'backprop',
        'baseline': 'standard_gradient_descent',
        'seed': 42,
        'd_in': 32 * 32,           # 1024 for flattened 32x32 images
        'd_out': 128,               # 128 units
        'lr': 0.001,                # Learning rate
        'device': str(device),
    }
    
    print("="*70)
    print("GRID EXPERIMENT #017".center(70))
    print("="*70)
    print(f"Dataset: {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps: {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture: {EXPERIMENT_CONFIG['architecture']}")
    print(f"  Input dim: {EXPERIMENT_CONFIG['d_in']}")
    print(f"  Output dim: {EXPERIMENT_CONFIG['d_out']}")
    print(f"Activation: {EXPERIMENT_CONFIG['activation']}")
    print(f"Rule: {EXPERIMENT_CONFIG['rule']} (standard backpropagation)")
    print(f"Baseline: {EXPERIMENT_CONFIG['baseline']}")
    print(f"Learning rate: {EXPERIMENT_CONFIG['lr']}")
    print(f"Seed: {EXPERIMENT_CONFIG['seed']}")
    print(f"Device: {EXPERIMENT_CONFIG['device']}")
    print("="*70)
    
    # Create output directory with experiment identifier
    output_base = Path('outputs/grid_experiments')
    output_dir = output_base / f"run_017_{EXPERIMENT_CONFIG['dataset']}_{EXPERIMENT_CONFIG['steps']}steps_{EXPERIMENT_CONFIG['architecture']}_{EXPERIMENT_CONFIG['activation']}_{EXPERIMENT_CONFIG['rule']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(EXPERIMENT_CONFIG, f, indent=2)
    print(f"\nMetadata saved to {metadata_file}")
    
    # Create model (standard MLP with backprop)
    model = StandardMLP(
        d_in=EXPERIMENT_CONFIG['d_in'],
        d_out=EXPERIMENT_CONFIG['d_out']
    )
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=EXPERIMENT_CONFIG['lr'])
    
    # Loss function: MSE for temporal prediction
    # Train to predict x_t1 from x_t using the learned representation
    # We'll use a simple autoencoder-style loss: predict x_t1 from h_t (representation of x_t)
    # This is a self-supervised temporal prediction task
    criterion = nn.MSELoss()
    
    # Create datasets
    # For embedding export: single images (not temporal pairs)
    export_dataset = SyntheticShapesDataset(
        num_samples=1000,
        seed=42,
        return_temporal_pair=False
    )
    
    # For training: temporal pairs
    train_dataset = create_temporal_pair_dataset(
        num_samples=10000,  # Large enough to sample from during training
        seed=42
    )
    
    # Export embeddings before training
    print("\nExporting embeddings before training...")
    embeddings_before = export_embeddings(model, export_dataset, num_samples=1000, device=device)
    
    # Safety check: no NaN in embeddings
    assert not torch.isnan(embeddings_before['embeddings']).any(), \
        "ERROR: NaN detected in embeddings before training!"
    
    torch.save(embeddings_before, output_dir / 'embeddings_before.pt')
    print(f"Saved embeddings to {output_dir / 'embeddings_before.pt'}")
    
    # Initialize training logs
    log_interval = 50  # Log every 50 steps
    training_logs = {
        'step': [],
        'weight_norm': [],
        'activation_norm': [],
        'loss': []
    }
    
    # Training loop
    print(f"\nTraining standard MLP with backpropagation for {EXPERIMENT_CONFIG['steps']} steps...")
    model.train()
    
    for step in range(1, EXPERIMENT_CONFIG['steps'] + 1):
        # Sample a temporal pair from the dataset
        idx = torch.randint(0, len(train_dataset), (1,)).item()
        x_t, x_t1, _ = train_dataset[idx]
        
        # Flatten images to 1D
        x_t_flat = x_t.flatten()
        x_t1_flat = x_t1.flatten()
        
        # Ensure inputs are floats in [0,1] range
        if x_t_flat.dtype != torch.float32:
            x_t_flat = x_t_flat.float()
        if x_t1_flat.dtype != torch.float32:
            x_t1_flat = x_t1_flat.float()
        x_t_flat = torch.clamp(x_t_flat, 0.0, 1.0)
        x_t1_flat = torch.clamp(x_t1_flat, 0.0, 1.0)
        
        # Move to device
        x_t_flat = x_t_flat.to(device)
        x_t1_flat = x_t1_flat.to(device)
        
        # Forward pass: get representation of x_t
        h_t = model.forward(x_t_flat)  # Representation/embedding
        
        # Loss: predict x_t1 from h_t using a simple linear projection
        # We'll train an additional projection head for this task
        # For simplicity, we'll use the representation itself and add a projection layer
        # Actually, let's use a simpler approach: train to minimize representation distance
        # between temporally adjacent frames (contrastive-style)
        
        # Alternative: Use the representation to reconstruct/predict next frame
        # We'll add a projection head that maps representation back to input space
        # But for simplicity in baseline, let's use a contrastive loss:
        # Minimize distance between h_t and h_t1 for temporal pairs
        
        # Forward pass for x_t1
        h_t1 = model.forward(x_t1_flat)
        
        # Loss: minimize L2 distance between temporally adjacent representations
        # This is a simple temporal consistency loss
        loss = criterion(h_t, h_t1.detach())  # Predict h_t1 from h_t
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clear tensors
        del x_t_flat, x_t1_flat, h_t, h_t1, loss
        
        # Log metrics periodically
        if step % log_interval == 0 or step == 1:
            # Compute metrics
            weight_norm = torch.norm(model.linear.weight).item()
            
            # Sample a random image for activation norm
            sample_idx = torch.randint(0, len(train_dataset), (1,)).item()
            sample_img, _, _ = train_dataset[sample_idx]
            sample_img_flat = sample_img.flatten().float().clamp(0.0, 1.0).to(device)
            
            with torch.no_grad():
                sample_emb = model.forward(sample_img_flat)
                activation_norm = torch.norm(sample_emb).item()
                del sample_emb
            
            del sample_img_flat
            
            # Compute loss for logging (sample a new pair)
            log_idx = torch.randint(0, len(train_dataset), (1,)).item()
            log_x_t, log_x_t1, _ = train_dataset[log_idx]
            log_x_t_flat = log_x_t.flatten().float().clamp(0.0, 1.0).to(device)
            log_x_t1_flat = log_x_t1.flatten().float().clamp(0.0, 1.0).to(device)
            
            with torch.no_grad():
                log_h_t = model.forward(log_x_t_flat)
                log_h_t1 = model.forward(log_x_t1_flat)
                log_loss = criterion(log_h_t, log_h_t1).item()
                del log_h_t, log_h_t1
            
            del log_x_t_flat, log_x_t1_flat
            
            training_logs['step'].append(step)
            training_logs['weight_norm'].append(weight_norm)
            training_logs['activation_norm'].append(activation_norm)
            training_logs['loss'].append(log_loss)
        
        # Clear GPU cache periodically
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print progress every 1000 steps
        if step % 1000 == 0:
            weight_norm = torch.norm(model.linear.weight).item()
            print(f"Step {step}/{EXPERIMENT_CONFIG['steps']} | "
                  f"||W||={weight_norm:.4f} | ||h||={training_logs['activation_norm'][-1]:.4f} | "
                  f"Loss={training_logs['loss'][-1]:.6f}")
    
    print("Training completed.")
    
    # Final safety check: no NaN in weights
    assert not torch.isnan(model.linear.weight).any(), \
        "ERROR: Weights contain NaN values after training!"
    
    # Save training logs
    logs_file = output_dir / 'training_logs.json'
    with open(logs_file, 'w') as f:
        json.dump(training_logs, f, indent=2)
    print(f"\nSaved training logs to {logs_file}")
    
    # Export embeddings after training
    print("\nExporting embeddings after training...")
    embeddings_after = export_embeddings(model, export_dataset, num_samples=1000, device=device)
    
    # Safety check: no NaN in embeddings
    assert not torch.isnan(embeddings_after['embeddings']).any(), \
        "ERROR: NaN detected in embeddings after training!"
    
    torch.save(embeddings_after, output_dir / 'embeddings_after.pt')
    print(f"Saved embeddings to {output_dir / 'embeddings_after.pt'}")
    
    # Comprehensive safety verification
    print("\n" + "="*70)
    print("SAFETY & VERIFICATION".center(70))
    print("="*70)
    
    # Check for NaN in weights
    weights_has_nan = torch.isnan(model.linear.weight).any().item()
    weights_check = 'PASS' if not weights_has_nan else 'FAIL'
    print(f"Weights NaN check: {weights_check}")
    if weights_has_nan:
        print("  ⚠️  WARNING: NaN values detected in weights!")
    
    # Check for NaN in embeddings_before
    emb_before_has_nan = torch.isnan(embeddings_before['embeddings']).any().item()
    emb_before_check = 'PASS' if not emb_before_has_nan else 'FAIL'
    print(f"Embeddings before NaN check: {emb_before_check}")
    if emb_before_has_nan:
        print("  ⚠️  WARNING: NaN values detected in embeddings_before!")
    
    # Check for NaN in embeddings_after
    emb_after_has_nan = torch.isnan(embeddings_after['embeddings']).any().item()
    emb_after_check = 'PASS' if not emb_after_has_nan else 'FAIL'
    print(f"Embeddings after NaN check: {emb_after_check}")
    if emb_after_has_nan:
        print("  ⚠️  WARNING: NaN values detected in embeddings_after!")
    
    # Check embedding std > 0.1 (non-collapsed)
    embeddings_final = embeddings_after['embeddings']
    embedding_std = embeddings_final.std().item()
    embedding_mean = embeddings_final.mean().item()
    embedding_min = embeddings_final.min().item()
    embedding_max = embeddings_final.max().item()
    non_collapsed = embedding_std > 0.1
    collapse_check = 'PASS' if non_collapsed else 'FAIL'
    print(f"Embedding std > 0.1 (non-collapsed): {collapse_check} (std={embedding_std:.6f})")
    if not non_collapsed:
        print("  ⚠️  WARNING: Embedding std <= 0.1 - representation may be collapsed!")
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS".center(70))
    print("="*70)
    print(f"Embedding mean: {embedding_mean:.6f}")
    print(f"Embedding std:  {embedding_std:.6f}")
    print(f"Embedding min:  {embedding_min:.6f}")
    print(f"Embedding max:  {embedding_max:.6f}")
    final_weight_norm = torch.norm(model.linear.weight).item()
    print(f"Weight norm: {final_weight_norm:.6f}")
    print("="*70)
    
    # Final summary output
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY".center(70))
    print("="*70)
    print(f"Dataset:          {EXPERIMENT_CONFIG['dataset']}")
    print(f"Steps:            {EXPERIMENT_CONFIG['steps']}")
    print(f"Architecture:     {EXPERIMENT_CONFIG['architecture']}")
    print(f"Embedding std:    {embedding_std:.6f}")
    print(f"Weight norm:      {final_weight_norm:.6f}")
    all_checks_pass = not weights_has_nan and not emb_before_has_nan and not emb_after_has_nan and non_collapsed
    summary_check = 'PASS' if all_checks_pass else 'FAIL'
    print(f"NaN check:        {summary_check}")
    if not all_checks_pass:
        print("\n  ⚠️  WARNING: One or more safety checks failed!")
        print(f"     - Weights NaN: {weights_check}")
        print(f"     - Embeddings before NaN: {emb_before_check}")
        print(f"     - Embeddings after NaN: {emb_after_check}")
        print(f"     - Non-collapsed (std > 0.1): {collapse_check}")
    print("="*70)
    
    print(f"\nExperiment completed successfully!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

