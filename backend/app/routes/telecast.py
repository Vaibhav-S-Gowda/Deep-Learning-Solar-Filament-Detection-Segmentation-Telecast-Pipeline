import os
import random
import base64
import cv2
from fastapi import APIRouter, HTTPException, Depends
from backend.app.schemas import TelecastFrameRequest, TelecastFrameResponse
from backend.app.services.pipeline_service import SolarPipelineService
from backend.app.routes.inference import get_pipeline_service

router = APIRouter(prefix="/api/v1/telecast", tags=["Live Telecast Processing"])

SAMPLE_IMAGES_DIR = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\train_images"


@router.post("/process-frame", response_model=TelecastFrameResponse, summary="Process Live Telecast Frame")
async def process_telecast_frame(
    req: TelecastFrameRequest,
    service: SolarPipelineService = Depends(get_pipeline_service)
):
    """
    Real-time frame ingestion endpoint for live telescope video streams.
    Accepts base64-encoded solar images, segments active filaments, and returns
    annotated overlay with solar activity status.
    """
    try:
        resp = service.process_telecast_frame(
            frame_base64=req.frame_base64,
            threshold=req.detection_threshold,
            return_overlay=req.return_overlay
        )
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame processing error: {str(e)}")


@router.get("/sample-frame", summary="Get Sample Live Telecast Frame for Next.js Simulation")
async def get_sample_telecast_frame():
    """
    Fetches a real H-Alpha solar observation image from the dataset as a base64 frame.
    Allows Next.js frontend developers to simulate a live telescope telecast loop immediately.
    """
    if not os.path.exists(SAMPLE_IMAGES_DIR):
        raise HTTPException(status_code=404, detail="Dataset images directory not found")

    files = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith((".jpeg", ".jpg"))]
    if not files:
        raise HTTPException(status_code=404, detail="No images found in dataset directory")

    chosen_file = random.choice(files)
    file_path = os.path.join(SAMPLE_IMAGES_DIR, chosen_file)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read sample image")

    # Resize to 512x512 for responsive live simulation transmission
    img_sim = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img_sim, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return {
        "success": True,
        "file_name": chosen_file,
        "width": 512,
        "height": 512,
        "frame_base64": frame_b64,
        "telescope_station": chosen_file[-7:-5] if len(chosen_file) >= 7 else "GONG"
    }


@router.get("/status", summary="Telecast Stream Status")
async def get_telecast_status(service: SolarPipelineService = Depends(get_pipeline_service)):
    """Returns streaming health, accelerator device, and model readiness."""
    return {
        "status": "ready",
        "device": str(service.device),
        "stage1_cnn_ready": service.cnn_loaded,
        "stage2_unet_ready": service.unet_loaded,
        "recommended_fps": 10 if service.device.type == "cpu" else 30
    }
