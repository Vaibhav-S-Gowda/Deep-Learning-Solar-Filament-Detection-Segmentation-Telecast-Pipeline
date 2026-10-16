import os
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from backend.app.schemas import (
    CNNDetectionResponse,
    SegmentationResponse,
    FullAnalysisResponse
)
from backend.app.services.pipeline_service import SolarPipelineService

router = APIRouter(prefix="/api/v1", tags=["Solar Filament Inference"])

# Service singleton
_pipeline_service: Optional[SolarPipelineService] = None

def get_pipeline_service() -> SolarPipelineService:
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = SolarPipelineService()
    return _pipeline_service


@router.post("/detect", response_model=CNNDetectionResponse, summary="Stage 1: Basic CNN Presence Detection")
async def detect_filament(
    file: UploadFile = File(..., description="Solar H-Alpha grayscale image (JPEG, PNG)"),
    service: SolarPipelineService = Depends(get_pipeline_service)
):
    """
    Runs Stage 1 Basic Convolutional Neural Network.
    Determines if solar filaments are present on the solar disc and predicts dominant orientation category.
    """
    try:
        content = await file.read()
        _, image_tensor, _ = service.preprocess_image(content)
        result = service.detect_presence(image_tensor)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@router.post("/segment", response_model=SegmentationResponse, summary="Stage 2: U-Net Solar Filament Segmentation")
async def segment_filament(
    file: UploadFile = File(..., description="Solar H-Alpha grayscale image (JPEG, PNG)"),
    threshold: float = Form(0.4, description="Binary segmentation threshold (0.1 - 0.9)"),
    service: SolarPipelineService = Depends(get_pipeline_service)
):
    """
    Runs Stage 2 U-Net Segmentation Network.
    Extracts fine-grained pixel segmentation masks, COCO-style polygons, and bounding boxes.
    """
    try:
        if threshold <= 0.0 or threshold >= 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be strictly between 0.0 and 1.0 (e.g. 0.4).")
        content = await file.read()
        _, image_tensor, orig_shape = service.preprocess_image(content)
        result, _ = service.segment_filaments(image_tensor, orig_shape, threshold=threshold)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Segmentation error: {str(e)}")


@router.post("/analyze", response_model=FullAnalysisResponse, summary="Full End-to-End Pipeline Analysis")
async def analyze_solar_image(
    file: UploadFile = File(..., description="Solar H-Alpha grayscale image (JPEG, PNG)"),
    threshold: float = Form(0.4, description="Binary segmentation threshold (0.1 - 0.9)"),
    service: SolarPipelineService = Depends(get_pipeline_service)
):
    """
    Complete ML Pipeline execution:
    1. Runs Stage 1 Basic CNN for screening and classification.
    2. Runs Stage 2 U-Net for pixel-level segmentation.
    3. Generates high-contrast visual overlay (base64) ready for direct Next.js `<img />` rendering.
    """
    try:
        if threshold <= 0.0 or threshold >= 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be strictly between 0.0 and 1.0 (e.g. 0.4).")
        content = await file.read()
        result = service.analyze_image(content, filename=file.filename or "observation.jpg", threshold=threshold)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis error: {str(e)}")
