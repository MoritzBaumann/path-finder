# Geospatial imports
from collections import defaultdict
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from skimage.transform import resize
from scipy.ndimage import binary_dilation
import cv2

# ML/Data handling imports
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall

import os
import matplotlib.pyplot as plt
import glob


# ========================================
# STEP 11: PLOT TRAINING RESULTS
# ========================================
print("\nGenerating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
epochs_range = range(1, num_epochs + 1)

# Plot 1: Loss
axes[0, 0].plot(epochs_range, history['train_loss'],
                'b-o', label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs_range, history['val_loss'],
                'r-o', label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training and Validation Loss',
                     fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: IoU and F1 Score
axes[0, 1].plot(epochs_range, history['val_iou'],
                'g-o', label='IoU', linewidth=2)
axes[0, 1].plot(epochs_range, history['val_f1'],
                'm-o', label='F1 Score', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Score', fontsize=12)
axes[0, 1].set_title('Validation IoU and F1 Score',
                     fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# Plot 3: Precision and Recall
axes[1, 0].plot(epochs_range, history['val_precision'],
                'c-o', label='Precision', linewidth=2)
axes[1, 0].plot(epochs_range, history['val_recall'], 'orange',
                marker='o', label='Recall', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_title('Validation Precision and Recall',
                     fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1])

# Plot 4: Summary metrics table
axes[1, 1].axis('off')
summary_text = f"""
Final Training Results (Epoch {num_epochs})
{'='*40}

Loss:
  Train Loss:      {history['train_loss'][-1]:.4f}
  Val Loss:        {history['val_loss'][-1]:.4f}

Performance Metrics:
  IoU:             {history['val_iou'][-1]:.4f}
  F1 Score:        {history['val_f1'][-1]:.4f}
  Precision:       {history['val_precision'][-1]:.4f}
  Recall:          {history['val_recall'][-1]:.4f}

Best Performance:
  Best IoU:        {max(history['val_iou']):.4f} (Epoch {history['val_iou'].index(max(history['val_iou']))+1})
  Best F1:         {max(history['val_f1']):.4f} (Epoch {history['val_f1'].index(max(history['val_f1']))+1})
  Best Val Loss:   {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss']))+1})
"""

axes[1, 1].text(0.1, 0.5, summary_text,
                fontsize=11,
                family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('../training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training plots saved to '../training_results.png'")


# ========================================
# STEP 12: MODEL ARCHITECTURE SUMMARY
# ========================================

def print_model_summary(model, optimizer, scheduler, device):
    """
    Print a comprehensive summary of the model architecture and training configuration
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE & TRAINING CONFIGURATION".center(70))
    print("="*70)

    # Model Architecture
    print("\n📐 MODEL ARCHITECTURE")
    print("-" * 70)
    print(f"  Architecture:          {model.__class__.__name__}")
    print(f"  Encoder Weights:       ImageNet (pretrained)")
    print(f"  Input Channels:        4 (RGB + NIR)")
    print(f"  Output Classes:        1 (binary segmentation)")
    print(f"  Activation:            Sigmoid")
    print(f"  Device:                {device.upper()}")

    # Model Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable_params:,}")
    print(
        f"  Model Size:            {total_params * 4 / (1024**2):.2f} MB (FP32)")

    # Loss Function
    print("\n📊 LOSS FUNCTION")
    print("-" * 70)
    print(f"  Loss Type:             Combined Loss")
    print(f"    └─ Dice Loss:        Weight = 1.0")
    print(f"    └─ Focal Loss:       Weight = 1.0, α=0.25, γ=2.0")
    print(f"  Purpose:               Handle severe class imbalance")

    # Optimizer
    print("\n⚙️  OPTIMIZER")
    print("-" * 70)
    print(f"  Optimizer:             {optimizer.__class__.__name__}")
    print(f"  Learning Rate:         {optimizer.param_groups[0]['lr']}")
    if 'weight_decay' in optimizer.param_groups[0]:
        print(
            f"  Weight Decay:          {optimizer.param_groups[0]['weight_decay']}")
    if 'betas' in optimizer.param_groups[0]:
        print(f"  Betas:                 {optimizer.param_groups[0]['betas']}")
    print(f"  Gradient Clipping:     max_norm=1.0")

    # Learning Rate Scheduler
    print("\n📉 LEARNING RATE SCHEDULER")
    print("-" * 70)
    print(f"  Scheduler:             {scheduler.__class__.__name__}")
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        print(f"  Mode:                  {scheduler.mode}")
        print(f"  Factor:                {scheduler.factor}")
        print(f"  Patience:              {scheduler.patience}")

    # Data Configuration
    print("\n🗂️  DATA CONFIGURATION")
    print("-" * 70)
    print(
        f"  Tile Size:             {TILE_SIZE}x{TILE_SIZE} → {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  Batch Size:            {BATCH_SIZE}")
    print(f"  Path Dilation:         {PATH_PIXEL_DILATION} pixels")
    print(f"  Train/Val Split:       80% / 20%")
    print(f"  Normalization:         ImageNet (RGB) + [0,1] (NIR)")

    # Metrics
    print("\n📈 EVALUATION METRICS")
    print("-" * 70)
    print(f"  Primary Metric:        IoU (Intersection over Union)")
    print(f"  Additional Metrics:    F1-Score, Precision, Recall")
    print(f"  Threshold:             0.5")

    # Training Configuration
    print("\n🎯 TRAINING CONFIGURATION")
    print("-" * 70)
    print(f"  Number of Epochs:      {num_epochs}")
    print(
        f"  Early Stopping:        Enabled (patience={scheduler.patience if hasattr(scheduler, 'patience') else 'N/A'})")
    print(f"  Model Checkpoint:      Best IoU")
    print(
        f"  Mixed Precision:       {'Enabled' if torch.cuda.is_available() else 'Disabled'}")

    print("\n" + "="*70 + "\n")


# Call the function after setting up your model
print_model_summary(model, optimizer, scheduler, device)

# Optional: Save summary to text file for easy reference
with open('../model_summary.txt', 'w') as f:
    import sys
    old_stdout = sys.stdout
    sys.stdout = f
    print_model_summary(model, optimizer, scheduler, device)
    sys.stdout = old_stdout

print("✓ Model summary saved to '../model_summary.txt'")
