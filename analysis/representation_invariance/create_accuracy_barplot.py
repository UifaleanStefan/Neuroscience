"""
Create a bar plot comparing MLP vs Conv-MLP accuracy across all datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from ANALYSIS_INTERPRETATION_COMPLETE.md
datasets = ['Synthetic\nShapes', 'MNIST', 'Fashion-\nMNIST', 'CIFAR-10', 'STL-10']

# Best accuracy values (linear probe)
mlp_accuracy = [
    100.0,   # Synthetic Shapes - 1k steps (best overall)
    69.0,    # MNIST - 5k steps
    31.2,    # Fashion-MNIST - 5k steps
    25.3,    # CIFAR-10 - 1k steps
    25.2     # STL-10 - 1k steps
]

conv_mlp_accuracy = [
    99.8,    # Synthetic Shapes - 10k steps
    36.3,    # MNIST - 1k steps (best for Conv-MLP, but worse than MLP)
    50.0,    # Fashion-MNIST - 5k steps
    27.7,    # CIFAR-10 - 5k steps
    31.3     # STL-10 - 10k steps
]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Set width of bars
x = np.arange(len(datasets))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, mlp_accuracy, width, label='MLP', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, conv_mlp_accuracy, width, label='Conv-MLP', color='#A23B72', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Customize plot
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Linear Probe Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('MLP vs Conv-MLP: Best Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)

# Add horizontal line at chance level (10% for 10-class datasets, but varies)
# For simplicity, not adding chance line since it varies by dataset

plt.tight_layout()

# Save figure
output_dir = Path('analysis/representation_invariance')
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / 'mlp_vs_conv_mlp_accuracy_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_path}")

# Also save as PDF
pdf_path = output_dir / 'mlp_vs_conv_mlp_accuracy_comparison.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
print(f"Saved PDF to {pdf_path}")

plt.close()
