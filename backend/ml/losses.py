import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Specifically suited for class-imbalanced datasets like solar filaments,
    where filament pixels are a small fraction of the solar disc.
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred_probs: (B, 1, H, W) in [0, 1]
        # target: (B, 1, H, W) in {0, 1}
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Combined BCE + Dice Loss for robust segmentation optimization.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_bce = self.bce(logits, targets)
        pred_probs = torch.sigmoid(logits)
        loss_dice = self.dice(pred_probs, targets)

        total_loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_bce": loss_bce.item(),
            "loss_dice": loss_dice.item()
        }


def compute_segmentation_metrics(
    pred_probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Computes key segmentation metrics:
    - IoU (Jaccard Index)
    - Dice (F1-score)
    - Precision
    - Recall / Sensitivity
    - Specificity
    - Pixel Accuracy
    """
    preds = (pred_probs > threshold).float()
    targets = (targets > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + smooth)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_acc)
    }
