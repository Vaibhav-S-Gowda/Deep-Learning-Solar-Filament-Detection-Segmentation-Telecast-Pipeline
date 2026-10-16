import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.dataset import get_dataloaders
from backend.ml.trainer import PipelineTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Solar Filament Detection & Segmentation Pipeline")
    parser.add_argument("--json_path", type=str, default=r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    parser.add_argument("--image_dir", type=str, default=r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\train_images")
    parser.add_argument("--stage", type=str, default="both", choices=["cnn", "unet", "both"], help="Stage to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=120, help="Max samples for fast baseline training")
    parser.add_argument("--save_dir", type=str, default=r"c:\Users\vaibh\Desktop\Solar-Filament\backend\models")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow experiment tracking")
    parser.add_argument("--experiment_name", type=str, default="Solar-Filament-Segmentation", help="MLflow experiment name")

    args = parser.parse_args()

    tracker = None
    if args.mlflow:
        from backend.ml.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker(experiment_name=args.experiment_name)
        tracker.start_run(run_name=f"pipeline_{args.stage}_e{args.epochs}_b{args.batch_size}")
        tracker.log_params({
            "stage": args.stage,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "save_dir": args.save_dir
        })

    print(f"Loading MAGFiLO dataset (Max samples: {args.max_samples})...")
    train_loader, val_loader, full_ds = get_dataloaders(
        json_path=args.json_path,
        image_dir=args.image_dir,
        target_size=(256, 256),
        batch_size=args.batch_size,
        val_split=0.2,
        max_samples=args.max_samples
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")

    trainer = PipelineTrainer(save_dir=args.save_dir)

    if args.stage in ["cnn", "both"]:
        print("\n=== Training Stage 1: Basic CNN Classifier ===")
        cnn_res = trainer.train_basic_cnn(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=1e-3,
            checkpoint_name="basic_cnn.pth"
        )
        if tracker:
            tracker.log_artifact(cnn_res["checkpoint_path"], artifact_path="checkpoints")
            tracker.log_model(cnn_res["model"], artifact_path="cnn_model")

    if args.stage in ["unet", "both"]:
        print("\n=== Training Stage 2: Solar Filament U-Net Segmenter ===")
        unet_res = trainer.train_unet_segmenter(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=5e-4,
            checkpoint_name="unet_segmenter.pth"
        )
        if tracker:
            tracker.log_artifact(unet_res["checkpoint_path"], artifact_path="checkpoints")
            tracker.log_model(unet_res["model"], artifact_path="unet_model")

    if tracker:
        tracker.end_run()

    print("\nAll requested training stages completed successfully!")


if __name__ == "__main__":
    main()
