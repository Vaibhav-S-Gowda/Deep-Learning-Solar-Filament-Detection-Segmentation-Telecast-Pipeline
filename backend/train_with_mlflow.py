#!/usr/bin/env python3
"""
Train Solar Filament Segmentation Model with Full MLflow Tracking.

Logs:
- Model architecture hyperparameters & training settings
- Epoch-by-epoch loss and evaluation metrics (Dice, IoU, Precision, Recall)
- Saved model checkpoint (.pth) and JSON report artifacts
- Registered MLflow PyTorch model flavor

Usage:
    python backend/train_with_mlflow.py --epochs 3 --batch-size 8 --max-samples 60
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.dataset import SolarFilamentDataset
from backend.ml.unet_model import SolarFilamentUNet
from backend.ml.losses import DiceBCELoss, compute_segmentation_metrics
from backend.ml.mlflow_tracker import MLflowTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train SolarFilamentUNet with MLflow Tracking")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=40, help="Max dataset samples to use (for rapid verification)")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    parser.add_argument("--experiment-name", type=str, default="Solar-Filament-Segmentation", help="MLflow experiment name")
    parser.add_argument("--checkpoint-dir", type=str, default="backend/models", help="Directory to save checkpoint")
    return parser.parse_args()


def train_with_mlflow(
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 5e-4,
    max_samples: int = 40,
    run_name: str = None,
    experiment_name: str = "Solar-Filament-Segmentation",
    checkpoint_dir: str = "backend/models"
):
    json_path = os.path.join(
        PROJECT_ROOT,
        "MAGFiLO_1.0_Kaggle_2026",
        "train",
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    img_dir = os.path.join(
        PROJECT_ROOT,
        "MAGFiLO_1.0_Kaggle_2026",
        "train",
        "train_images"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[MLflow Training] Initializing on device: {device}")

    # Load dataset
    full_ds = SolarFilamentDataset(
        json_path=json_path,
        image_dir=img_dir,
        target_size=(256, 256),
        is_train=True,
        max_samples=max_samples
    )

    val_size = max(1, int(len(full_ds) * 0.2))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[Dataset] Train samples: {train_size} | Val samples: {val_size}")

    # Initialize tracker
    tracker = MLflowTracker(experiment_name=experiment_name)
    run_title = run_name or f"unet_run_e{epochs}_b{batch_size}_{int(time.time())}"
    tracker.start_run(run_name=run_title)

    # Log parameters
    tracker.log_params({
        "architecture": "SolarFilamentUNet",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "loss_function": "DiceBCELoss",
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "input_resolution": "256x256",
        "train_samples": train_size,
        "val_samples": val_size,
        "device": str(device)
    })

    # Initialize model
    model = SolarFilamentUNet().to(device)
    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_dice = -1.0
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, "unet_segmenter_mlflow.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = criterion(outputs["logits"], masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / max(1, len(train_loader.dataset))

        # Evaluation
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                loss, _ = criterion(outputs["logits"], masks)
                running_val_loss += loss.item() * images.size(0)

                all_preds.append(outputs["probs"].cpu())
                all_targets.append(masks.cpu())

        val_loss = running_val_loss / max(1, len(val_loader.dataset))
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        metrics = compute_segmentation_metrics(cat_preds, cat_targets, threshold=0.4)

        ep_duration = time.time() - ep_t0

        # Log epoch metrics to MLflow
        tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": metrics["dice"],
            "val_iou": metrics["iou"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_pixel_acc": metrics["pixel_accuracy"],
            "epoch_duration_sec": ep_duration
        }, step=epoch)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"({ep_duration:.2f}s)"
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_dice": best_dice
            }, checkpoint_path)

    total_time = time.time() - start_time

    # Log summary metrics
    tracker.log_metrics({
        "best_dice": best_dice,
        "best_val_loss": best_val_loss,
        "total_training_time_sec": total_time
    })

    # Log checkpoint artifact
    tracker.log_artifact(checkpoint_path, artifact_path="checkpoints")

    # Log PyTorch Model flavor
    tracker.log_model(model, artifact_path="model")

    tracker.end_run()

    print("\n" + "=" * 65)
    print(f"MLflow Training Run Complete!")
    print(f"Total Time: {total_time:.2f}s | Best Dice: {best_dice:.4f}")
    print(f"Local MLflow runs stored in: {os.path.abspath('mlruns')}")
    print(f"Run 'mlflow ui' to visualize in browser on http://localhost:5000")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    args = parse_args()
    train_with_mlflow(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir
    )
