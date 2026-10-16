import os
import json
import time
from typing import Dict, Any, Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from backend.ml.cnn_model import FilamentBasicCNN, compute_cnn_loss
from backend.ml.unet_model import SolarFilamentUNet
from backend.ml.losses import DiceBCELoss, compute_segmentation_metrics


class PipelineTrainer:
    """
    Unified Trainer for:
      - Stage 1: Basic CNN Classifier
      - Stage 2: U-Net Segmenter
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        save_dir: str = "backend/models"
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train_basic_cnn(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        lr: float = 1e-3,
        checkpoint_name: str = "basic_cnn.pth"
    ) -> Dict[str, Any]:
        """Trains Stage 1 Basic CNN Classifier."""
        model = FilamentBasicCNN().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_presence_acc": [],
            "val_category_acc": []
        }

        best_val_loss = float("inf")
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        print(f"--- Starting Stage 1 Basic CNN Training on {self.device} for {epochs} epochs ---")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            running_train_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                has_filaments = batch["has_filament"].to(self.device)
                categories = batch["category_id"].to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss, _ = compute_cnn_loss(outputs, has_filaments, categories)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * images.size(0)

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            scheduler.step()

            # Validation
            model.eval()
            running_val_loss = 0.0
            correct_presence = 0
            correct_category = 0
            total_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    has_filaments = batch["has_filament"].to(self.device)
                    categories = batch["category_id"].to(self.device)

                    outputs = model(images)
                    loss, _ = compute_cnn_loss(outputs, has_filaments, categories)
                    running_val_loss += loss.item() * images.size(0)

                    # Presence accuracy
                    preds_presence = (outputs["presence_prob"] > 0.5).squeeze(1).float()
                    correct_presence += (preds_presence == has_filaments).sum().item()

                    # Category accuracy
                    preds_cat = torch.argmax(outputs["category_probs"], dim=-1)
                    correct_category += (preds_cat == categories).sum().item()

                    total_samples += images.size(0)

            epoch_val_loss = running_val_loss / max(1, total_samples)
            presence_acc = correct_presence / max(1, total_samples)
            category_acc = correct_category / max(1, total_samples)

            history["epoch"].append(epoch)
            history["train_loss"].append(round(epoch_train_loss, 4))
            history["val_loss"].append(round(epoch_val_loss, 4))
            history["val_presence_acc"].append(round(presence_acc, 4))
            history["val_category_acc"].append(round(category_acc, 4))

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Presence Acc: {presence_acc * 100:.2f}% | "
                f"Cat Acc: {category_acc * 100:.2f}%"
            )

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "presence_acc": presence_acc,
                    "category_acc": category_acc
                }, checkpoint_path)

        elapsed = time.time() - start_time
        print(f"Basic CNN training completed in {elapsed:.2f}s. Checkpoint saved to: {checkpoint_path}")

        # Save training report
        report_path = os.path.join(self.save_dir, "cnn_training_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "model": "FilamentBasicCNN",
                "epochs": epochs,
                "best_val_loss": best_val_loss,
                "history": history,
                "elapsed_seconds": round(elapsed, 2)
            }, f, indent=2)

        return {"model": model, "checkpoint_path": checkpoint_path, "history": history}

    def train_unet_segmenter(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        lr: float = 5e-4,
        checkpoint_name: str = "unet_segmenter.pth"
    ) -> Dict[str, Any]:
        """Trains Stage 2 U-Net Segmenter."""
        model = SolarFilamentUNet().to(self.device)
        criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_precision": [],
            "val_recall": []
        }

        best_dice = -1.0
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        print(f"--- Starting Stage 2 U-Net Segmenter Training on {self.device} for {epochs} epochs ---")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            running_train_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss, _ = criterion(outputs["logits"], masks)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * images.size(0)

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            scheduler.step()

            # Validation
            model.eval()
            running_val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)

                    outputs = model(images)
                    loss, _ = criterion(outputs["logits"], masks)
                    running_val_loss += loss.item() * images.size(0)

                    all_preds.append(outputs["probs"].cpu())
                    all_targets.append(masks.cpu())

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            cat_preds = torch.cat(all_preds, dim=0)
            cat_targets = torch.cat(all_targets, dim=0)
            metrics = compute_segmentation_metrics(cat_preds, cat_targets, threshold=0.4)

            history["epoch"].append(epoch)
            history["train_loss"].append(round(epoch_train_loss, 4))
            history["val_loss"].append(round(epoch_val_loss, 4))
            history["val_iou"].append(round(metrics["iou"], 4))
            history["val_dice"].append(round(metrics["dice"], 4))
            history["val_precision"].append(round(metrics["precision"], 4))
            history["val_recall"].append(round(metrics["recall"], 4))

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"IoU: {metrics['iou']:.4f} | "
                f"Dice: {metrics['dice']:.4f} | "
                f"Recall: {metrics['recall']:.4f}"
            )

            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": epoch_val_loss,
                    "dice": best_dice,
                    "iou": metrics["iou"],
                    "metrics": metrics
                }, checkpoint_path)

        elapsed = time.time() - start_time
        print(f"U-Net training completed in {elapsed:.2f}s. Checkpoint saved to: {checkpoint_path}")

        # Save training report
        report_path = os.path.join(self.save_dir, "unet_training_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "model": "SolarFilamentUNet",
                "epochs": epochs,
                "best_dice": best_dice,
                "history": history,
                "elapsed_seconds": round(elapsed, 2)
            }, f, indent=2)

        return {"model": model, "checkpoint_path": checkpoint_path, "history": history}
