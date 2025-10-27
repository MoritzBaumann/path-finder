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

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import cv2

import os
import glob
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_labels(labels_path: Path) -> gpd.GeoDataFrame:
    """Load vector path labels from GeoPackage files and combine into a single GeoDataFrame."""
    print("Loading labels...")
    labels_list = sorted(glob.glob(os.path.join(labels_path, "*.gpkg")))
    print(f"\nFound {len(labels_list)} labels (geopackage files):")
    for label in labels_list:
        print(f"  - {os.path.basename(label)}")

    labels = []
    for label in labels_list:
        gdf = gpd.read_file(label)
        labels.append(gdf)
    labels_gdf = gpd.GeoDataFrame(pd.concat(
        [gdf for gdf in labels], ignore_index=True))
    print(
        f"✓ Loaded {len(labels_list)} path features and combined into one GeoDataFrame")

    return labels_gdf


def rasterize_labels_for_tiff(labels_gdf: gpd.GeoDataFrame, tiff_path: Path) -> np.ndarray:
    """
    Rasterize vector path labels to match a specific GeoTIFF's spatial extent and resolution.

    Creates a binary mask where paths are 1 and background is 0, aligned to the 
    GeoTIFF's coordinate system, dimensions, and transform.

    Args:
        labels_gdf: GeoDataFrame containing path geometries (may span larger area than TIFF)
        tiff_path: Path to GeoTIFF file that defines the target raster space

    Returns:
        Binary mask array (uint8) with shape (height, width) matching the GeoTIFF
    """
    try:
        with rasterio.open(tiff_path) as src:
            tiff_metadata = src.meta.copy()

            print(
                f"  📐 Raster: {tiff_metadata['width']}W × {tiff_metadata['height']}H pixels")
            print(f"  🗺️  CRS: {tiff_metadata['crs']}")
            print(f"  🖊️  Rasterizing {len(labels_gdf)} path geometries...")

            shapes = ((geom, 1) for geom in labels_gdf.geometry)
            label_mask = rasterize(
                shapes=shapes,
                out_shape=(tiff_metadata['height'], tiff_metadata['width']),
                transform=tiff_metadata['transform'],
                fill=0,
                dtype='uint8'
            )

            path_pixels = np.sum(label_mask)
            coverage_pct = (path_pixels / label_mask.size) * 100
            print(
                f"  ✓ Mask: {label_mask.shape}, {path_pixels:,} path pixels ({coverage_pct:.2f}% coverage)")

    except Exception as e:
        raise ValueError(
            f"Failed to rasterize labels for {tiff_path.name}: {str(e)}")

    return label_mask


def dilate_path_pixels(mask: np.ndarray, pixel_dilation: int) -> np.ndarray:
    """
    Dilate path pixels in binary mask to increase path width.

    This method computes the Euclidean distance from each background pixel
    to the nearest path pixel and marks all pixels within the specified radius
    as path.
    Applied at original resolution before downscaling to prevent paths from
    disappearing during resize. Uses elliptical structuring element for smooth edges.


    Args:
        mask: Binary mask array (uint8 or bool) where 1=path, 0=background.
        pixel_dilation: Dilation radius in pixels (0=no dilation).

    Returns:
        Dilated binary mask of the same shape as the input.
    """
    if pixel_dilation <= 0:
        print("  ⊘ No dilation applied")
        return mask

    print(f"  🛤 Broadening path using distance-based dilation...")
    # Compute Euclidean distance from each background pixel to nearest foreground pixel
    dist = cv2.distanceTransform(
        1 - mask, distanceType=cv2.DIST_L2, maskSize=5)
    # Pixels within radius `pixel_dilation` of a path become 1
    mask_dilated = (dist <= pixel_dilation).astype(np.uint8)
    print(
        f"  ✅ Dilated paths: {pixel_dilation}px radius using distance transform")
    return mask_dilated


def downscale_image_tile(img_tile: np.ndarray, target_size: int) -> np.ndarray:
    """Downscale multi-band image tile using bilinear interpolation."""
    img_downscaled = np.zeros(
        (3, target_size, target_size), dtype=img_tile.dtype
    )
    for band in range(3):
        img_downscaled[band] = cv2.resize(
            img_tile[band],
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR
        )
    return img_downscaled


def downscale_mask_tile(mask_tile: np.ndarray, target_size: int, threshold: float = 0.3) -> np.ndarray:
    """Downscale binary mask using bilinear interpolation with thresholding to preserve paths."""
    mask_downscaled = cv2.resize(
        mask_tile.astype(np.float32),
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR
    )
    mask_downscaled = (mask_downscaled > threshold).astype(np.uint8)
    return mask_downscaled


def extract_and_save_tiles(
    geotiff_path: Path,
    mask: np.ndarray,
    tile_size: int,
    target_size: int,
    tile_dir: Path,
    file_prefix: str = "tile"
) -> int:
    """
    Extract non-overlapping tiles from GeoTIFF and aligned mask, downscale, and save.

    Tiles are extracted at original resolution, downscaled to target size, and saved
    as paired numpy arrays. Incomplete tiles at image boundaries are skipped.

    Args:
        geotiff_path: Path to source GeoTIFF file
        mask: Binary mask array aligned to GeoTIFF spatial extent (H×W), already dilated
        tile_size: Extraction window size at original resolution (pixels)
        target_size: Downscaled output size (pixels)
        tile_dir: Directory for saved tile arrays
        file_prefix: Unique prefix for filenames (prevents overwrites across GeoTIFFs)

    Returns:
        Number of valid tiles extracted and saved

    Output Format:
        {file_prefix}_img_{col}_{row}.npy  -> shape (3, target_size, target_size), RGB
        {file_prefix}_mask_{col}_{row}.npy -> shape (target_size, target_size), binary
    """
    with rasterio.open(geotiff_path) as src:
        img_width, img_height = src.width, src.height

        print(
            f"  📊 Source: {img_width}W × {img_height}H, {src.count} bands, CRS={src.crs}")
        print(
            f"  ✂️  Extracting {tile_size}×{tile_size} tiles → downscaling to {target_size}×{target_size}")

        tile_count = 0

        for col in range(0, img_width, tile_size):
            for row in range(0, img_height, tile_size):
                # Define extraction window
                window = Window(col_off=col, row_off=row,  # type: ignore
                                width=tile_size, height=tile_size)  # type: ignore

                # Extract aligned image and mask tiles
                img_tile = src.read([1, 2, 3], window=window)  # RGB only
                mask_tile = mask[row:row + tile_size, col:col + tile_size]

                # Skip incomplete tiles at boundaries
                if img_tile.shape != (3, tile_size, tile_size):
                    continue
                if mask_tile.shape != (tile_size, tile_size):
                    continue

                # Downscale both tiles
                img_downscaled = downscale_image_tile(img_tile, target_size)
                mask_downscaled = downscale_mask_tile(mask_tile, target_size)

                # Save as numpy arrays
                np.save(tile_dir /
                        f"{file_prefix}_img_{col}_{row}.npy", img_downscaled)
                np.save(tile_dir /
                        f"{file_prefix}_mask_{col}_{row}.npy", mask_downscaled)

                tile_count += 1

        print(f"  ✓ Extracted {tile_count} complete tiles")

    return tile_count


def process_geotiffs_to_tiles(
    geotiffs_path: Path,
    labels_path: Path,
    output_dir: Path,
    tile_size: int,
    target_size: int,
    path_pixel_dilation: int
) -> dict:
    """
    Process all GeoTIFF files into training tiles with aligned masks.

    For each GeoTIFF:
    1. Rasterize vector labels to create aligned mask
    2. Dilate paths to increase path width
    3. Extract and downscale tiles
    4. Save as numpy arrays

    Args:
        geotiffs_path: Directory containing GeoTIFF files (.tiff files)
        labels_path: Directory containing label geometries (.gpkg files)
        output_dir: Directory for saved tile arrays
        tile_size: Extraction window size at original resolution (pixels)
        target_size: Downscaled output size (pixels)
        path_pixel_dilation: Radius in pixels to dilate paths

    Returns:
        Dictionary with processing statistics:
        - 'total_tiles': Total number of tiles created
        - 'processed_tiffs': Number of successfully processed GeoTIFFs
        - 'failed_tiffs': Number of GeoTIFFs that failed
    """
    # Load path labels
    labels_gdf = load_labels(labels_path)

    tiff_files = sorted(glob.glob(os.path.join(geotiffs_path, "*.tif")))
    print(f"\n{'='*60}")
    print(f"Found {len(tiff_files)} GeoTIFF files to process")
    print(f"{'='*60}")

    # Create output directory
    tile_dir = output_dir / f"tiles_{tile_size//1000}k_to_{target_size}p"
    tile_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    processed_count = 0
    failed_count = 0

    for tiff_idx, tiff_path in enumerate(tiff_files):
        tiff_name = os.path.basename(tiff_path)
        print(f"\n[{tiff_idx + 1}/{len(tiff_files)}] Processing {tiff_name}")
        print(f"{'-'*60}")

        # Generate unique prefix for this TIFF's tiles
        file_prefix = f"tiff{tiff_idx+1:02d}"

        try:
            # Step 1: Rasterize labels for this specific TIFF
            mask = rasterize_labels_for_tiff(labels_gdf, Path(tiff_path))

            # Step 2: Dilate paths at original resolution
            mask_dilated = dilate_path_pixels(mask, path_pixel_dilation)

            # Step 3: Extract and save tiles
            tile_count = extract_and_save_tiles(
                geotiff_path=Path(tiff_path),
                mask=mask_dilated,
                tile_size=tile_size,
                target_size=target_size,
                tile_dir=tile_dir,
                file_prefix=file_prefix
            )

            total_tiles += tile_count
            processed_count += 1

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed_count += 1
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {processed_count}/{len(tiff_files)} GeoTIFFs")
    print(f"  Failed: {failed_count}")
    print(f"  Total tiles created: {total_tiles:,}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

    return {
        'total_tiles': total_tiles,
        'processed_tiffs': processed_count,
        'failed_tiffs': failed_count,
        'tile_dir': tile_dir
    }


def visualize_tiles(tile_dir: Path, num_tiles_to_visualize: int = 5):
    """
    Visualizes a sample of image and corresponding mask tiles in a single figure.

    The function randomly selects a specified number of image tiles from the
    given directory, loads the image and its associated mask, and plots them
    side-by-side as subplots within a single, large figure.

    Args:
        tile_dir (Path): The directory containing the image and mask tiles.
                         Image files should contain '_img_' in their name
                         and masks should contain '_mask_', and both should
                         be loadable with numpy.load (e.g., .npy files).
        num_tiles_to_visualize (int, optional): The number of image/mask pairs
                                             to randomly select and visualize.
                                             Defaults to 5.

    Raises:
        ValueError: If no image files are found in the specified directory.
    """
    # File Selection and Validation
    img_files = sorted(tile_dir.glob("*_img_*"))
    if not img_files:
        raise ValueError(
            f"No image files containing '_img_' found in {tile_dir}")

    # Randomly select the specified number of files
    num_to_sample = min(num_tiles_to_visualize, len(img_files))
    rnd_img_files = random.sample(img_files, num_to_sample)

    print(
        f"\nVisualizing {num_to_sample} sample tiles from '{tile_dir.name}'...")

    # Setup Single Figure for All Plots
    # Each pair (img + mask) needs 2 columns.
    n_rows = num_to_sample
    n_cols = 2
    # Determine the figure size dynamically for better visual balance
    fig_width = 10
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_width, fig_height)
    )

    # Flatten axes array for easy indexing, especially if n_rows=1
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    # Load and Plot Tiles
    for i, img_filepath in enumerate(rnd_img_files):
        try:
            # Determine mask path and load data
            mask_filepath = Path(str(img_filepath).replace("_img_", "_mask_"))
            img = np.load(img_filepath)
            mask = np.load(mask_filepath)
        except Exception as e:
            print(f"Skipping file {img_filepath.name} due to load error: {e}")
            continue

        # Convert to (H, W, bands) for display if necessary (assuming channel-first)
        if img.ndim == 3 and img.shape[0] < img.shape[-1]:
            img = np.moveaxis(img, 0, -1)

        # Plot Image
        ax_img = axes[i, 0]
        ax_img.imshow(img)
        ax_img.set_title(f"Image: {img_filepath.name}", fontsize=10)
        ax_img.axis("off")

        # Plot Mask
        ax_mask = axes[i, 1]
        ax_mask.imshow(mask, cmap="Reds")
        ax_mask.set_title("Mask", fontsize=10)
        ax_mask.axis("off")

    # Final Display Cleanup
    fig.suptitle(
        f"Sample Visualization of {num_to_sample} Image/Mask Tile Pairs",
        fontsize=16,
        fontweight='bold',
        y=1.00  # Adjust position for better title visibility
    )
    plt.tight_layout()
    plt.show()
