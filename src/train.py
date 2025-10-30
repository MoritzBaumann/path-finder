from datetime import datetime
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import yaml
from tqdm import tqdm

from src.data_utils import create_dataloaders
from src.model_utils import (
    create_model,
    get_losses,
    get_metrics,
    get_optimizer,
    get_scheduler,
)


def save_model_and_history_to_disk(
    model: smp.Unet,
    history: dict,
    output_dir: Path,
    model_type: str,
    encoder_name: str,
    dataset_name: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
    encoder_weights: str = "random",
) -> None:
    """Save trained model state dict and training history to disk.

    1. Model state dict (.pth) - learnable parameters (weights and biases)
    2. Training history (.yaml) - loss and metric values across epochs

    Both files are saved with timestamped filenames that include hyperparameter metadata and
    architecture details needed for easy identification and reproducibility.

    Args:
        model: Trained segmentation model (e.g., U-Net with specified encoder).
        history: Dictionary containing training metrics per epoch. Expected keys:
            'train_loss', 'val_loss', 'val_iou', 'val_f1', 'val_precision', 'val_recall'.
        output_dir: Directory path where model artifacts will be saved.
        model_type: Model architecture type (e.g., 'unet', 'fpn').
        encoder_name: Name of the encoder backbone (e.g., 'resnet34', 'efficientnet-b0').
        dataset_name: Name/identifier of the training dataset.
        num_epochs: Total number of training epochs completed.
        batch_size: Batch size used during training.
        lr: Learning rate used for optimization.
        encoder_weights: Pretrained weights used for encoder initialization
            (e.g., 'imagenet'). Defaults to "random" for random initialization.

    Returns:
        None. Files are written to disk at the specified output_dir.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create formatted strings for filenames
    lr_str = f"{lr:.0e}".replace("+0", "")  # e.g., '1e-4'
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_name = f"{date_str}_{model_type}_{encoder_name}_{encoder_weights}_{dataset_name}_ep{num_epochs}_lr{lr_str}_bs{batch_size}"

    model_name = f"{base_name}.pth"
    history_name = f"{base_name}_history.yaml"

    # Save model state dict
    torch.save(model.state_dict(), output_dir / model_name)

    # Save training history
    with open(output_dir / history_name, "w") as f:
        yaml.dump(history, f, default_flow_style=False)

    print(f"\n✓ Model artifacts saved to {output_dir}")
    print(f"  • Model state dict: {model_name}")
    print(f"  • Training history: {history_name}")


def train_model(
    tile_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    encoder: str,
    weights: str,
) -> tuple[smp.Unet, dict]:
    """Train a binary segmentation model using U-Net architecture.

    This function implements a complete training loop with validation, including:
    - Data loading from preprocessed tiles
    - Model training with backpropagation
    - Validation with multiple metrics (IoU, F1, precision, recall)
    - Learning rate scheduling based on validation loss
    - Progress tracking with tqdm

    Args:
        tile_dir: Path to directory containing preprocessed image tiles stored as .npy files.
            Expected structure: separate subdirectories or naming convention for train/val splits.
        batch_size: Number of samples per batch for both training and validation dataloaders.
        epochs: Total number of training epochs to run.
        lr: Initial learning rate for the optimizer.
        encoder: Name of the encoder backbone architecture (e.g., 'resnet34', 'efficientnet-b0').
            Must be compatible with segmentation_models_pytorch.
        weights: Pretrained weight specification for encoder initialization.
            Use 'imagenet' for ImageNet pretrained weights, or None for random initialization.

    Returns:
        A tuple containing:
        - model (smp.Unet): Trained segmentation model on CPU/GPU depending on availability.
        - history (dict): Training history with keys 'train_loss', 'val_loss', 'val_iou',
            'val_f1', 'val_precision', 'val_recall'. Each key maps to a list of values per epoch.

    Notes:
        - Automatically detects and uses CUDA if available, otherwise falls back to CPU.
        - Uses binary cross-entropy loss for training.
        - Applies sigmoid activation + 0.5 threshold for binary predictions during validation.
        - Learning rate scheduler reduces LR when validation loss plateaus.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize training components
    train_loader, val_loader = create_dataloaders(tile_dir, batch_size)
    model = create_model(encoder, weights).to(device)
    loss_fn = get_losses()
    optimizer = get_optimizer(model, lr)
    scheduler = get_scheduler(optimizer)
    metrics = get_metrics(device)

    # Initialize history tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
    }

    # Training loop across epochs
    for epoch in range(1, epochs + 1):
        # ===== Training Phase =====
        model.train()
        train_loss_accum = 0.0
        n_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            preds = model(images)
            loss = loss_fn(preds, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            train_loss_accum += loss.item()
            n_batches += 1
            progress_bar.set_postfix(
                {"train_loss": f"{train_loss_accum / n_batches:.4f}"}
            )

        train_loss = train_loss_accum / max(1, n_batches)

        # ===== Validation Phase =====
        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0

        # Reset metrics from previous epoch
        for m in metrics.values():
            if hasattr(m, "reset"):
                m.reset()

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                # Forward pass
                preds = model(images)
                loss = loss_fn(preds, masks)
                val_loss_accum += loss.item()
                n_val_batches += 1

                # Convert logits to binary predictions
                probs = torch.sigmoid(preds)
                preds_bin = (probs > 0.5).int()

                # Update metrics (handle different tensor shapes)
                for name, metric in metrics.items():
                    try:
                        metric(preds_bin, masks.int())
                    except Exception:
                        # Handle case where metric expects shape without channel dimension
                        metric(preds_bin.squeeze(1), masks.squeeze(1).int())

        # Compute epoch metrics
        val_loss = val_loss_accum / max(1, n_val_batches)
        val_iou = metrics["iou"].compute().item()
        val_f1 = metrics["f1"].compute().item()
        val_precision = metrics["precision"].compute().item()
        val_recall = metrics["recall"].compute().item()

        # Reset metrics after computation
        for m in metrics.values():
            if hasattr(m, "reset"):
                m.reset()

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)

        # Print epoch summary
        print(
            f"Epoch {epoch}/{epochs} — "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"F1: {val_f1:.4f}"
        )

    print(f"\n✓ Training complete! ({epochs} epochs)")

    return model, history
