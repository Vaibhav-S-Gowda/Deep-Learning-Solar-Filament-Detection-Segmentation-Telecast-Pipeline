import os
import json
import threading
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.app.schemas import TrainRequest, TrainStatusResponse
from backend.ml.dataset import get_dataloaders
from backend.ml.trainer import PipelineTrainer
from backend.app.routes.inference import get_pipeline_service

router = APIRouter(prefix="/api/v1/train", tags=["Model Training & Fine-Tuning"])

# In-memory training state
_training_state: Dict[str, Any] = {
    "is_training": False,
    "current_stage": None,
    "progress_percentage": 0.0,
    "status_message": "Idle",
    "latest_metrics": None
}

MODELS_DIR = "backend/models"


def run_training_task(req: TrainRequest):
    global _training_state
    try:
        _training_state["is_training"] = True
        _training_state["status_message"] = "Loading dataset..."
        _training_state["current_stage"] = req.stage

        json_path = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        img_dir = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\train_images"

        train_loader, val_loader, _ = get_dataloaders(
            json_path=json_path,
            image_dir=img_dir,
            target_size=(256, 256),
            batch_size=req.batch_size,
            val_split=0.2,
            max_samples=req.max_samples
        )

        trainer = PipelineTrainer(save_dir=MODELS_DIR)

        if req.stage in ["cnn", "both"]:
            _training_state["status_message"] = "Training Stage 1 Basic CNN..."
            _training_state["current_stage"] = "Stage 1: Basic CNN"
            res = trainer.train_basic_cnn(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=req.epochs,
                checkpoint_name="basic_cnn.pth"
            )
            _training_state["latest_metrics"] = res.get("history")

        if req.stage in ["unet", "both"]:
            _training_state["status_message"] = "Training Stage 2 Solar Filament U-Net..."
            _training_state["current_stage"] = "Stage 2: U-Net Segmenter"
            res = trainer.train_unet_segmenter(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=req.epochs,
                checkpoint_name="unet_segmenter.pth"
            )
            _training_state["latest_metrics"] = res.get("history")

        # Reload updated model weights in pipeline service
        service = get_pipeline_service()
        service.load_checkpoints()

        _training_state["is_training"] = False
        _training_state["status_message"] = "Training completed successfully!"
        _training_state["progress_percentage"] = 100.0

    except Exception as e:
        _training_state["is_training"] = False
        _training_state["status_message"] = f"Training failed: {str(e)}"
        print(f"[Training Error] {e}")


@router.post("/start", response_model=TrainStatusResponse, summary="Start Pipeline Training")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Triggers Stage 1 CNN or Stage 2 U-Net training asynchronously."""
    global _training_state
    if _training_state["is_training"]:
        raise HTTPException(status_code=409, detail="A training session is already in progress.")

    _training_state["is_training"] = True
    _training_state["status_message"] = "Queued training task..."
    _training_state["progress_percentage"] = 0.0

    background_tasks.add_task(run_training_task, req)

    return TrainStatusResponse(
        is_training=True,
        current_stage=req.stage,
        progress_percentage=0.0,
        status_message="Training initialized in background.",
        latest_metrics=None
    )


@router.get("/status", response_model=TrainStatusResponse, summary="Get Training Status")
async def get_training_status():
    """Returns real-time status and metrics of active/recent training sessions."""
    global _training_state
    return TrainStatusResponse(
        is_training=_training_state["is_training"],
        current_stage=_training_state["current_stage"],
        progress_percentage=_training_state["progress_percentage"],
        status_message=_training_state["status_message"],
        latest_metrics=_training_state["latest_metrics"]
    )


@router.get("/history", summary="Get Historical Training Curves")
async def get_training_history():
    """Loads saved loss & metric reports for charting in Next.js UI."""
    cnn_report_path = os.path.join(MODELS_DIR, "cnn_training_report.json")
    unet_report_path = os.path.join(MODELS_DIR, "unet_training_report.json")

    cnn_history = None
    unet_history = None

    if os.path.exists(cnn_report_path):
        with open(cnn_report_path, "r") as f:
            cnn_history = json.load(f)

    if os.path.exists(unet_report_path):
        with open(unet_report_path, "r") as f:
            unet_history = json.load(f)

    return {
        "stage1_cnn": cnn_history,
        "stage2_unet": unet_history
    }
