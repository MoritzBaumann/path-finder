import os
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from src.data_utils import create_dataloaders
from src.model_utils import (
    create_model,
    get_losses,
    get_metrics,
    get_optimizer,
    get_scheduler,
)


def create_model_name_and_path(
    model_type: str,
    encoder_name: str,
    dataset_name: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
    encoder_weights: str | None = None,
) -> tuple[str, Path]:
    """Creates a name for an ML model including the date and various metadata."""
    lr_str = f"{lr:.0e}".replace("+0", "")  # e.g. 1e-4
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"{date_str}_{model_type}_{encoder_name}_{encoder_weights}_{dataset_name}_ep{num_epochs}_lr{lr_str}_bs{batch_size}.pth"
    return (model_name, output_dir / model_name)


def train_model(
    tile_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    encoder: str,
    weights: str,
    save_path: Path,
) -> dict:
    """Train loop for binary segmentation.

    Args:
        tile_dir: Path to directory with tiles (npys) produced by data preprocessing.
        batch_size: Batch size for loaders.
        epochs: Number of epochs.
        lr: Learning rate.
        encoder: Encoder name for segmentation model.
        weights: Encoder weights specification (e.g. "imagenet" or None).
        save_path: Where to save final model state_dict.

    Returns:
        History dict with lists for losses and metrics per epoch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = create_dataloaders(tile_dir, batch_size)
    model = create_model(encoder, weights).to(device)
    loss_fn = get_losses()
    optimizer = get_optimizer(model, lr)
    scheduler = get_scheduler(optimizer)
    metrics = get_metrics(device)

    # Bookkeeping
    history = {
        k: []
        for k in [
            "train_loss",
            "val_loss",
            "val_iou",
            "val_f1",
            "val_precision",
            "val_recall",
        ]
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Epoch loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        # Training loop
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            n_batches += 1
            progress_bar.set_postfix(
                {"train_loss": f"{train_loss_accum / n_batches:.4f}"}
            )

        train_loss = train_loss_accum / max(1, n_batches)

        # Validation loop
        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0

        for m in metrics.values():
            if hasattr(m, "reset:"):
                m.reset()  # Clear old metric states

        val_loss = val_iou = val_f1 = val_precision = val_recall = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                preds = model(images)
                loss = loss_fn(preds, masks)
                val_loss_accum += loss.item()
                n_val_batches += 1

                # Convert logits -> binary predictions
                probs = torch.sigmoid(preds)
                preds_bin = (probs > 0.5).int()

                # Update metrics safely (handles different tensor shapes)
                for name, metric in metrics.items():
                    try:
                        metric(preds_bin, masks.int())
                    except Exception:
                        metric(preds_bin.squeeze(1), masks.squeeze(1).int())

        # Metrics computation
        val_loss = val_loss_accum / max(1, n_val_batches)
        val_iou = metrics["iou"].compute().item()
        val_f1 = metrics["f1"].compute().item()
        val_precision = metrics["precision"].compute().item()
        val_recall = metrics["recall"].compute().item()

        for m in metrics.values():
            if hasattr(m, "reset"):
                m.reset()

        scheduler.step(val_loss)

        # Logging
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)

        print(
            f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Training complete! \nModel saved under {save_path}")
    return history
