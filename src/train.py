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
# STEP 9: MODEL SETUP
# ========================================
print("\nSetting up model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = smp.Unet(
    encoder_name="efficientnet-b0",  # or resnet34 -> more power, slower
    encoder_weights="imagenet",
    in_channels=4,
    classes=1,
    activation=None
).to(device)

# Loss functions (Solution 1: Weighted/Focal Loss)
dice_loss = smp.losses.DiceLoss(mode="binary")
focal_loss = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)


def combined_loss(preds, targets):
    return dice_loss(preds, targets) + focal_loss(preds, targets)


print("Using Dice + Focal Loss to handle class imbalance")


# Optimizer & Learning Rate
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Metrics
iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)
f1_metric = BinaryF1Score(threshold=0.5).to(device)
precision_metric = BinaryPrecision(threshold=0.5).to(device)
recall_metric = BinaryRecall(threshold=0.5).to(device)

# Learning rate scheduler (optional)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)


# ========================================
# STEP 10: TRAINING LOOP
# ========================================
print("\nStarting training...")

num_epochs = 20

# Initialize tracking lists
history = {
    'train_loss': [],
    'val_loss': [],
    'val_iou': [],
    'val_f1': [],
    'val_precision': [],
    'val_recall': []
}

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    # ---- TRAIN ----
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()

        # Add gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- VALIDATION ----
    model.eval()
    val_loss, val_iou, val_f1, val_precision, val_recall = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)

            loss = combined_loss(preds, masks)
            val_loss += loss.item() * images.size(0)

            preds_sigmoid = torch.sigmoid(preds)
            val_iou += iou_metric(preds_sigmoid, masks.int()
                                  ).item() * images.size(0)
            val_f1 += f1_metric(preds_sigmoid, masks.int()
                                ).item() * images.size(0)
            val_precision += precision_metric(preds_sigmoid,
                                              masks.int()).item() * images.size(0)
            val_recall += recall_metric(preds_sigmoid,
                                        masks.int()).item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)
    val_f1 /= len(val_loader.dataset)
    val_precision /= len(val_loader.dataset)
    val_recall /= len(val_loader.dataset)

    # Store metrics in history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    history['val_f1'].append(val_f1)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)

    # Learning rate scheduling
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f}")
    print(
        f"Val Loss:   {val_loss:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f}")
    print(f"Precision:  {val_precision:.4f} | Recall: {val_recall:.4f}")

    # Warning if model is predicting all zeros
    if val_recall < 0.1:
        print("⚠️  WARNING: Very low recall - model may be predicting mostly zeros!")

print("\n✓ Training complete!")

# Save model
torch.save(model.state_dict(),
           "../trained_models/path_segmentation_model_lr0.0005.pth")
print("Model saved!")
