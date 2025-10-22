"""
data_utils.py

Utilities for reading raster/vector data, rasterizing vector labels, creating
tiles, and visualizing samples for coastal path detection preprocessing.

Key features:
- Use relatively large excerpts (4096x4096) of GeoTIFFs to preserve context
- Downsample to manageable size (256x256) for model training
- Keep images and masks aligned through all processing steps
- Use label dilation to thicken thin paths and decrease class imbalance
- Perform label dilation at original resolution before any downsampling
- Export image and mask tiles as NumPy arrays for downstream training
"""

from collections import defaultdict
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import cv2

import os
import glob

import numpy as np
import matplotlib.pyplot as plt


# ========================================
# STEP 1: LOAD VECTOR DATA ONCE
# ========================================
print("Loading vector data...")
labels = gpd.read_file(vector_path)
print(f"✓ Loaded {len(labels)} path features")

# ========================================
# STEP 2: GET ALL TIFF FILES
# ========================================
tiff_files = sorted(glob.glob(os.path.join(raster_dir, "*.tif")))
print(f"\nFound {len(tiff_files)} TIFF files:")
for tiff_file in tiff_files:
    print(f"  - {os.path.basename(tiff_file)}")


# ========================================
# STEP 3: PROCESS EACH TIFF FILE
# ========================================


def rasterize_paths(paths_gdf, out_meta):
    shapes = ((geom, 1) for geom in paths_gdf.geometry)
    mask = rasterize(
        shapes=shapes,
        out_shape=(out_meta['height'], out_meta['width']),
        transform=out_meta['transform'],
        fill=0,
        dtype='uint8'
    )

    return mask


def generate_tiles_from_tiff(image_path, mask, tile_size=TILE_SIZE,
                             target_size=TARGET_SIZE, out_dir=tile_dir,
                             file_prefix=""):
    """Generate tiles from a single TIFF file"""

    # CRITICAL: Dilate at ORIGINAL resolution before downscaling
    # This ensures the path doesn't disappear during downscaling
    if PATH_PIXEL_DILATION > 0:
        kernel_original = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (PATH_PIXEL_DILATION*2+1, PATH_PIXEL_DILATION*2+1)
        )
        print(
            f"  Dilating paths by {PATH_PIXEL_DILATION} pixels at original resolution...")
        mask = cv2.dilate(mask, kernel_original)

    with rasterio.open(image_path) as src:
        img_width, img_height = src.width, src.height
        tile_count = 0

        for i in range(0, img_width, tile_size):
            for j in range(0, img_height, tile_size):
                window = Window(i, j, tile_size, tile_size)
                img_tile = src.read(window=window)
                mask_tile = mask[j:j+tile_size, i:i+tile_size]

                if img_tile.shape != (4, tile_size, tile_size):
                    continue
                if mask_tile.shape != (tile_size, tile_size):
                    continue

                # Downscale image using cv2 (much faster than skimage)
                img_downscaled = np.zeros(
                    (4, target_size, target_size), dtype=img_tile.dtype)
                for band in range(4):
                    img_downscaled[band] = cv2.resize(
                        img_tile[band],
                        (target_size, target_size),
                        interpolation=cv2.INTER_LINEAR
                    )

                # Downscale mask using cv2
                # Use INTER_LINEAR for masks to preserve partial coverage, then threshold
                mask_downscaled = cv2.resize(
                    mask_tile.astype(np.float32),
                    (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR
                )
                # Threshold to binary (anything >0.3 becomes 1)
                mask_downscaled = (mask_downscaled > 0.3).astype(np.uint8)

                # Save with file prefix to avoid overwrites
                np.save(os.path.join(
                    out_dir, f"{file_prefix}_img_{i}_{j}.npy"), img_downscaled)
                np.save(os.path.join(
                    out_dir, f"{file_prefix}_mask_{i}_{j}.npy"), mask_downscaled)
                tile_count += 1

        return tile_count


# Create output directory
os.makedirs(tile_dir, exist_ok=True)

# Process each TIFF file
total_tiles = 0
for tiff_idx, tiff_path in enumerate(tiff_files):
    print(f"\n{'='*60}")
    print(
        f"Processing TIFF {tiff_idx+1}/{len(tiff_files)}: {os.path.basename(tiff_path)}")
    print(f"{'='*60}")

    # Create a unique prefix for this TIFF's tiles
    file_prefix = f"tiff{tiff_idx:02d}"

    # Load raster and rasterize vector data for this TIFF
    try:
        with rasterio.open(tiff_path) as src:
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  CRS: {src.crs}")
            print(f"  Rasterizing vector data...")
            mask = rasterize_paths(labels, src.meta)
            print(
                f"  Mask shape: {mask.shape}, unique values: {np.unique(mask)}")

        # Generate tiles
        print(f"  Generating tiles...")
        tile_count = generate_tiles_from_tiff(
            tiff_path, mask,
            tile_size=TILE_SIZE,
            target_size=TARGET_SIZE,
            out_dir=tile_dir,
            file_prefix=file_prefix
        )

        print(f"  ✓ Generated {tile_count} tiles from this TIFF")
        total_tiles += tile_count

    except Exception as e:
        print(f"  ✗ Error processing {os.path.basename(tiff_path)}: {str(e)}")
        continue

print(f"\n{'='*60}")
print(
    f"✓ COMPLETE: Generated {total_tiles} tiles from {len(tiff_files)} TIFF files")
print(f"{'='*60}")


# ========================================
# STEP 4: VERIFY TILES
# ========================================
print("\nVerifying generated tiles...")
img_files = sorted(glob.glob(os.path.join(tile_dir, "*_img_*.npy")))
mask_files = sorted(glob.glob(os.path.join(tile_dir, "*_mask_*.npy")))
print(f"Found {len(img_files)} image tiles and {len(mask_files)} mask tiles")


# ========================================
# STEP 5: VISUALIZE SAMPLES FROM DIFFERENT TIFFS
# ========================================
print("\nVisualizing sample tiles from each TIFF...")

# Group tiles by TIFF source
tiles_by_tiff = defaultdict(list)
for img_file in img_files:
    # Extract tiff prefix (e.g., "tiff00" from "tiff00_img_0_0.npy")
    prefix = os.path.basename(img_file).split('_img_')[0]
    tiles_by_tiff[prefix].append(img_file)

# Show one tile from each TIFF
for prefix in sorted(tiles_by_tiff.keys())[:10]:  # Show first 5 TIFFs
    img_file = tiles_by_tiff[prefix][0]
    mask_file = img_file.replace("_img_", "_mask_")

    img = np.load(img_file)
    mask = np.load(mask_file)

    print(f"Dimensions of raster '{prefix}': {img.shape}")
    print(f"Dimensions of mask '{prefix}': {mask.shape}")

    # Convert to (H, W, bands) for display
    if img.shape[0] < img.shape[-1]:
        img = np.moveaxis(img, 0, -1)

    # Normalize for display
    img_disp = img[:, :, :3]
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Sample from {prefix}", fontsize=14, fontweight='bold')

    plt.subplot(1, 2, 1)
    plt.imshow(img_disp)
    plt.title(os.path.basename(img_file))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="Reds")
    plt.title("Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

print("\n✓ All TIFF files processed and tiles generated!")
