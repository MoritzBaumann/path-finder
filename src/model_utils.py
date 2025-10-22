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
# STEP 6: CUSTOM DATASET
# ========================================
class TileDataset(Dataset):
    def __init__(self, img_paths, mask_paths, normalize=True, preload=True):
        self.normalize = normalize
        self.mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        if preload:
            print("Preloading all tiles into memory...")
            self.images = [np.load(p) for p in img_paths]
            self.masks = [np.load(p) for p in mask_paths]
            self.img_paths = None
            self.mask_paths = None
            print(f"✓ Loaded {len(self.images)} tiles")
        else:
            self.images = None
            self.masks = None
            self.img_paths = img_paths
            self.mask_paths = mask_paths

    def __len__(self):
        return len(self.images) if self.images else len(self.img_paths)

    def __getitem__(self, idx):
        if self.images is not None:
            img = self.images[idx]
            mask = self.masks[idx]
        else:
            img = np.load(self.img_paths[idx])
            mask = np.load(self.mask_paths[idx])

        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.normalize:
            img_tensor = img_tensor / 255.0
            # Normalize RGB channels (0-2) with ImageNet stats
            img_tensor[:3] = (img_tensor[:3] - self.mean_rgb) / self.std_rgb
            # NIR channel (3) stays as-is (already 0-1 after division)

        return img_tensor, mask_tensor


# ========================================
# STEP 7: SPLIT DATA AND CREATE DATALOADERS
# ========================================
print("\nCreating train/val split...")

# Get file paths (NOT the actual data!)
img_files = sorted(glob.glob(os.path.join(tile_dir, "*img_*.npy")))
mask_files = [f.replace("img_", "mask_") for f in img_files]

# Split paths only
img_train, img_val, mask_train, mask_val = train_test_split(
    img_files, mask_files, test_size=0.2, random_state=42
)

print(f"Train tiles: {len(img_train)}")
print(f"Validation tiles: {len(img_val)}")

# Create datasets
train_dataset = TileDataset(img_train, mask_train, normalize=True)
val_dataset = TileDataset(img_val, mask_val, normalize=True)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# Test dataloader
print("\nTesting dataloader...")
for images, masks in train_loader:
    print(f"Batch shape: {images.shape}, {masks.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Mask range: [{masks.min():.2f}, {masks.max():.2f}]")
    break


# ========================================
# STEP 8: CALCULATE CLASS WEIGHTS
# ========================================
print("\nCalculating class weights from training data...")

total_pixels = 0
positive_pixels = 0

# Sample masks to calculate imbalance (use all or subset for speed)
sample_size = min(100, len(mask_train))
for mask_path in mask_train[:sample_size]:
    mask = np.load(mask_path)
    total_pixels += mask.size
    positive_pixels += mask.sum()

pos_ratio = positive_pixels / total_pixels
neg_ratio = 1 - pos_ratio

print(f"Positive pixels: {pos_ratio*100:.3f}%")
print(f"Negative pixels: {neg_ratio*100:.3f}%")

# Calculate pos_weight for BCE loss
pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0
print(f"Positive class weight: {pos_weight:.2f}")
