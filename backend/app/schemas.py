from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class FilamentPolygonAnnotation(BaseModel):
    id: str = Field(..., description="Unique filament identifier")
    segmentation: List[List[float]] = Field(..., description="List of [x0, y0, x1, y1, ...] polygon coordinates")
    bbox: List[float] = Field(..., description="[x, y, width, height] in pixel units")
    area: float = Field(..., description="Filament segmentation area in pixels")
    confidence: float = Field(..., description="Detection confidence score (0.0 - 1.0)")
    category_id: int = Field(default=1, description="Category ID (1: Left, 2: Right, 3: Unidentifiable, 4: Ambiguous)")
    category_name: str = Field(default="Filament", description="Category display label")


class CategoryScore(BaseModel):
    category_id: int
    name: str
    probability: float


class CNNDetectionResponse(BaseModel):
    success: bool = True
    has_filament: bool = Field(..., description="Whether a solar filament was detected by Stage 1 CNN")
    filament_probability: float = Field(..., description="Confidence probability of filament presence")
    dominant_category: str = Field(..., description="Dominant filament orientation or class")
    category_scores: List[CategoryScore] = Field(default_factory=list, description="Class probabilities")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")


class SegmentationResponse(BaseModel):
    success: bool = True
    num_filaments_found: int = Field(..., description="Number of separated filament structures identified")
    total_filament_area: float = Field(..., description="Total pixel area covered by filaments")
    coverage_percentage: float = Field(0.0, description="Percentage of solar disk covered by filaments")
    has_high_solar_activity: bool = Field(False, description="Flagged if total filament area exceeds 1000px")
    filaments: List[FilamentPolygonAnnotation] = Field(default_factory=list, description="Extracted filament polygons")
    rle_mask: Optional[Dict[str, Any]] = Field(default=None, description="Lossless COCO-compliant RLE mask")
    binary_mask_base64: Optional[str] = Field(None, description="Base64 encoded PNG of binary segmentation mask")
    inference_time_ms: float = Field(..., description="Segmentation latency in milliseconds")


class FullAnalysisResponse(BaseModel):
    success: bool = True
    image_metadata: Dict[str, Any] = Field(default_factory=dict, description="Image dimensions and capture info")
    stage1_cnn: CNNDetectionResponse = Field(..., description="Stage 1 CNN presence & classification output")
    stage2_segmentation: SegmentationResponse = Field(..., description="Stage 2 U-Net segmentation output")
    overlay_image_base64: Optional[str] = Field(None, description="JPEG/PNG Base64 string of annotated solar image")
    total_processing_time_ms: float = Field(..., description="Total pipeline latency")


class TelecastFrameRequest(BaseModel):
    frame_base64: str = Field(..., description="Base64-encoded solar telecast frame (JPEG or PNG)")
    timestamp: Optional[str] = Field(None, description="Optional timestamp string of the live feed")
    return_overlay: bool = Field(True, description="Whether to return annotated visual overlay frame")
    detection_threshold: float = Field(0.4, description="Segmentation probability threshold")


class TelecastFrameResponse(BaseModel):
    success: bool = True
    timestamp: str
    filaments_detected: int
    total_filament_area: float
    has_high_solar_activity: bool
    overlay_frame_base64: Optional[str] = None
    processing_time_ms: float


class DatasetStatsResponse(BaseModel):
    dataset_name: str
    total_train_images: int
    total_test_images: int
    total_annotations: int
    category_counts: Dict[str, int]
    image_resolution: str
    instrument: str


class TrainRequest(BaseModel):
    stage: str = Field("both", description="'cnn', 'unet', or 'both'")
    epochs: int = Field(3, ge=1, le=50)
    batch_size: int = Field(8, ge=1, le=64)
    max_samples: Optional[int] = Field(150, ge=10, le=1000)


class TrainStatusResponse(BaseModel):
    is_training: bool
    current_stage: Optional[str] = None
    progress_percentage: float = 0.0
    status_message: str
    latest_metrics: Optional[Dict[str, Any]] = None
