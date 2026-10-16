import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FilamentBasicCNN(nn.Module):
    """
    Stage 1: Basic Convolutional Neural Network for Solar Filament Detection.
    - Input: Grayscale solar image (B, 1, H, W)
    - Outputs:
      - presence_logits: (B, 1) probability of solar filament existing in image
      - category_logits: (B, 5) classification for [None, Left, Right, Unidentifiable, Ambiguous]
      - features: (B, 256) pooled deep visual features
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5, base_filters: int = 32):
        super(FilamentBasicCNN, self).__init__()

        # Conv Block 1: 1 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )

        # Conv Block 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )

        # Conv Block 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )

        # Conv Block 4: 128 -> 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling -> (B, 256, 1, 1)
        )

        self.flatten_dim = base_filters * 8

        # Classification Heads
        self.presence_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.category_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        features = torch.flatten(x4, 1)
        presence_logits = self.presence_head(features)
        category_logits = self.category_head(features)

        return {
            "presence_logits": presence_logits,
            "presence_prob": torch.sigmoid(presence_logits),
            "category_logits": category_logits,
            "category_probs": F.softmax(category_logits, dim=-1),
            "features": features
        }


def compute_cnn_loss(
    outputs: Dict[str, torch.Tensor],
    has_filament_target: torch.Tensor,
    category_target: torch.Tensor,
    presence_weight: float = 1.0,
    category_weight: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined Loss for Stage 1 CNN:
    BCEWithLogitsLoss for presence detection + CrossEntropyLoss for dominant category.
    """
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    loss_presence = bce(outputs["presence_logits"].squeeze(1), has_filament_target)
    loss_category = ce(outputs["category_logits"], category_target)

    total_loss = presence_weight * loss_presence + category_weight * loss_category

    return total_loss, {
        "loss_total": total_loss.item(),
        "loss_presence": loss_presence.item(),
        "loss_category": loss_category.item()
    }
