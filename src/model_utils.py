import segmentation_models_pytorch as smp
import torch
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)


def create_model(encoder: str, weights: str | None, in_channels: int = 3):
    """Create a Unet model from segmentation_models_pytorch.

    Args:
        encoder (str): Encoder name (e.g. "resnet34").
        weights (str | None): Pretrained weights name or None.
        in_channels (int, optional): Number of input channels. Defaults to 3.

    Returns:
        An `nn.Module` instance.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=1,
        activation=None,  # return logits
    )
    return model


def get_losses():
    """Return a loss function for binary segmentation.

    Combines Dice loss and Focal loss (from smp) to stabilize training.
    """
    dice_loss = smp.losses.DiceLoss(mode="binary")
    focal_loss = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)

    def combined_loss(preds, targets):
        # preds: logits, targets: binary {0,1}
        return dice_loss(preds, targets) + focal_loss(preds, targets)

    return combined_loss


def get_optimizer(model: torch.nn.Module, lr: float = 5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr)


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )


def get_metrics(device) -> dict:
    """Return instantiated torchmetrics metrics (moved to `device`).

    The returned dict contains metric objects and the caller should use the
    metrics like: `metric(preds, target)` then `metric.compute()` and `metric.reset()`.
    """
    return {
        "iou": BinaryJaccardIndex(threshold=0.5).to(device),
        "f1": BinaryF1Score(threshold=0.5).to(device),
        "precision": BinaryPrecision(threshold=0.5).to(device),
        "recall": BinaryRecall(threshold=0.5).to(device),
    }
