# Geospatial imports

import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_tiles(
    tile_dir: Path,
    num_tiles_to_visualize: int = 5,
    output_dir: Path | None = None,
) -> Path:
    """
    Visualizes a sample of image and corresponding mask tiles and saves to disk.

    The function randomly selects a specified number of image tiles from the
    given directory, loads the image and its associated mask, and plots them
    side-by-side as subplots within a single figure. The resulting visualization
    is saved as a PNG file.

    Args:
        tile_dir: The directory containing the image and mask tiles.
            Image files should contain '_img_' in their name and masks should
            contain '_mask_', and both should be loadable with numpy.load
            (e.g., .npy files).
        num_tiles_to_visualize: The number of image/mask pairs to randomly
            select and visualize. Defaults to 5.
        output_dir: Directory where the visualization will be saved. If None,
            saves to 'outputs/visualizations/' by default.
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("outputs/dataset_visualizations")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # File Selection and Validation
    img_files = sorted(tile_dir.glob("*_img_*"))
    if not img_files:
        raise ValueError(f"No image files containing '_img_' found in {tile_dir}")

    # Randomly select the specified number of files
    num_to_sample = min(num_tiles_to_visualize, len(img_files))
    rnd_img_files = random.sample(img_files, num_to_sample)
    print(f"\nVisualizing {num_to_sample} sample tiles from '{tile_dir.name}'...")

    # Setup Single Figure for All Plots
    # Each pair (img + mask) needs 2 columns.
    n_rows = num_to_sample
    n_cols = 2

    # Determine the figure size dynamically for better visual balance
    fig_width = 10
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_height)
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
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()

    # Save figure with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_tile_visualization_{tile_dir.name}_n{num_to_sample}.png"
    filepath = output_dir / filename

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory

    print(f"✓ Visualization saved to: {filepath}")

    return filepath


def plot_training_curves(history: dict, save_path: Path, run_name: str) -> None:
    """
    Create comprehensive training visualization with loss and metrics.

    Args:
        history: Training history dictionary with metrics per epoch.
        save_path: Directory to save the plot.
        run_name: Short name for this training run (for title/filename).
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)

    total_epochs = len(history["train_loss"])
    epochs = range(1, total_epochs + 1)

    ax[0, 0].plot(
        epochs, history["train_loss"], "b-", label="Train loss", linewidth=2.5
    )
    ax[0, 0].plot(epochs, history["val_loss"], "r-", label="Val loss", linewidth=2.5)
    ax[0, 0].set_xlabel("Epochs")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].set_title("Training and Validation Loss")
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)

    ax[1, 0].plot(epochs, history["val_iou"], "green", label="IoU", linewidth=2.5)
    ax[1, 0].plot(epochs, history["val_f1"], "purple", label="F1", linewidth=2.5)
    ax[1, 0].set_xlabel("Epochs")
    ax[1, 0].set_ylabel("Score")
    ax[1, 0].set_title("Validation IoU and F1 Score")
    ax[1, 0].legend()
    ax[1, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(
        epochs, history["val_precision"], "turquoise", label="Precision", linewidth=2.5
    )
    ax[0, 1].plot(
        epochs, history["val_recall"], "orange", label="Recall", linewidth=2.5
    )
    ax[0, 1].set_xlabel("Epochs")
    ax[0, 1].set_ylabel("Score")
    ax[0, 1].set_title("Validation Precision and Recall")
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 1].axis("off")
    summary_text = f"""
    Final Training Results (Epoch {total_epochs})
    {"=" * 40}

    Loss:
      Train Loss:      {history["train_loss"][-1]:.4f}
      Val Loss:        {history["val_loss"][-1]:.4f}

    Performance Metrics:
      IoU:             {history["val_iou"][-1]:.4f}
      F1 Score:        {history["val_f1"][-1]:.4f}
      Precision:       {history["val_precision"][-1]:.4f}
      Recall:          {history["val_recall"][-1]:.4f}

    Best Performance:
      Best IoU:        {max(history["val_iou"]):.4f} (Epoch {history["val_iou"].index(max(history["val_iou"])) + 1})
      Best F1:         {max(history["val_f1"]):.4f} (Epoch {history["val_f1"].index(max(history["val_f1"])) + 1})
      Best Val Loss:   {min(history["val_loss"]):.4f} (Epoch {history["val_loss"].index(min(history["val_loss"])) + 1})
    """

    ax[1, 1].text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.suptitle(f"Training Results for model run:\n{run_name}", fontsize=14)

    save_path.mkdir(parents=True, exist_ok=True)
    filename = f"{run_name}_training_curves.png"
    filepath = save_path / filename

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Training curves saved to: {filepath}")
