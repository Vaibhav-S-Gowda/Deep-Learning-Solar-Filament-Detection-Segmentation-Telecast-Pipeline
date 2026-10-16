import os
import io
import time
import base64
import numpy as np
import cv2
import torch
from typing import Dict, Any, Optional, Tuple

from backend.ml.cnn_model import FilamentBasicCNN
from backend.ml.unet_model import (
    SolarFilamentUNet,
    mask_to_polygons_and_bboxes,
    binary_mask_to_rle,
    create_detection_overlay
)
from backend.app.schemas import (
    CNNDetectionResponse,
    CategoryScore,
    SegmentationResponse,
    FilamentPolygonAnnotation,
    FullAnalysisResponse,
    TelecastFrameResponse
)


class SolarPipelineService:
    """
    Central inference engine managing:
      - Model checkpoint loading (Stage 1 CNN + Stage 2 U-Net)
      - End-to-end inference on images (file, bytes, or base64)
      - Generation of segmentation masks, polygons, and overlays
    """

    CATEGORIES = {
        0: "None",
        1: "Left",
        2: "Right",
        3: "Unidentifiable",
        4: "Ambiguous"
    }

    def __init__(self, models_dir: str = "backend/models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = models_dir

        self.cnn_model = FilamentBasicCNN().to(self.device)
        self.unet_model = SolarFilamentUNet().to(self.device)

        self.cnn_loaded = False
        self.unet_loaded = False

        self.load_checkpoints()

    def load_checkpoints(self):
        """Loads weights from disk if available."""
        cnn_path = os.path.join(self.models_dir, "basic_cnn.pth")
        if os.path.exists(cnn_path):
            try:
                ckpt = torch.load(cnn_path, map_location=self.device)
                self.cnn_model.load_state_dict(ckpt["model_state_dict"])
                self.cnn_loaded = True
                print(f"[PipelineService] Stage 1 CNN loaded from {cnn_path}")
            except Exception as e:
                print(f"[PipelineService] Could not load CNN checkpoint: {e}")

        unet_path = os.path.join(self.models_dir, "unet_segmenter.pth")
        if os.path.exists(unet_path):
            try:
                ckpt = torch.load(unet_path, map_location=self.device)
                self.unet_model.load_state_dict(ckpt["model_state_dict"])
                self.unet_loaded = True
                print(f"[PipelineService] Stage 2 U-Net loaded from {unet_path}")
            except Exception as e:
                print(f"[PipelineService] Could not load U-Net checkpoint: {e}")

        self.cnn_model.eval()
        self.unet_model.eval()

    def preprocess_image(self, image_bytes: bytes, target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, torch.Tensor, Tuple[int, int]]:
        """
        Decodes image bytes to grayscale numpy array and normalizes to PyTorch tensor.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_orig = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_orig is None:
            raise ValueError("Invalid image bytes. Could not decode image as valid JPEG, PNG, or BMP.")

        orig_shape = img_orig.shape  # (H, W)
        th, tw = target_size
        img_resized = cv2.resize(img_orig, (tw, th), interpolation=cv2.INTER_AREA)

        # Normalize to [0.0, 1.0]
        tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        return img_orig, tensor, orig_shape

    def detect_presence(self, image_tensor: torch.Tensor) -> CNNDetectionResponse:
        """Stage 1: Runs basic CNN to detect presence and dominant category."""
        t0 = time.time()
        with torch.no_grad():
            outputs = self.cnn_model(image_tensor)

        prob = float(outputs["presence_prob"][0, 0].item())
        has_filament = prob >= 0.5

        cat_probs = outputs["category_probs"][0].cpu().numpy().tolist()
        dom_idx = int(np.argmax(cat_probs))
        dom_category = self.CATEGORIES.get(dom_idx, "Unknown")

        category_scores = [
            CategoryScore(category_id=i, name=self.CATEGORIES.get(i, f"Class_{i}"), probability=round(p, 4))
            for i, p in enumerate(cat_probs)
        ]

        ms = round((time.time() - t0) * 1000, 2)
        return CNNDetectionResponse(
            success=True,
            has_filament=has_filament,
            filament_probability=round(prob, 4),
            dominant_category=dom_category,
            category_scores=category_scores,
            inference_time_ms=ms
        )

    def segment_filaments(
        self,
        image_tensor: torch.Tensor,
        orig_shape: Tuple[int, int],
        threshold: float = 0.4
    ) -> Tuple[SegmentationResponse, np.ndarray]:
        """Stage 2: Runs U-Net to extract fine-grained segmentation masks and polygons."""
        t0 = time.time()
        with torch.no_grad():
            outputs = self.unet_model(image_tensor)

        probs = outputs["probs"][0, 0].cpu().numpy()
        bin_mask = (probs > threshold).astype(np.uint8)

        filaments_raw = mask_to_polygons_and_bboxes(
            bin_mask,
            min_area=15,
            approx_epsilon=1.0,
            orig_shape=orig_shape
        )

        filaments = [
            FilamentPolygonAnnotation(
                id=f["id"],
                segmentation=f["segmentation"],
                bbox=f["bbox"],
                area=f["area"],
                confidence=round(f["confidence"], 4),
                category_id=1,
                category_name="Solar Filament"
            )
            for f in filaments_raw
        ]

        total_area = sum(f.area for f in filaments)
        orig_h, orig_w = orig_shape
        coverage_pct = round((total_area / max(1.0, float(orig_h * orig_w))) * 100, 4)
        has_high_activity = bool(total_area > 1000.0)

        # Base64 binary mask
        _, mask_buf = cv2.imencode(".png", (bin_mask * 255).astype(np.uint8))
        mask_b64 = f"data:image/png;base64,{base64.b64encode(mask_buf).decode('utf-8')}"

        rle = binary_mask_to_rle(bin_mask)
        ms = round((time.time() - t0) * 1000, 2)

        resp = SegmentationResponse(
            success=True,
            num_filaments_found=len(filaments),
            total_filament_area=round(total_area, 2),
            coverage_percentage=coverage_pct,
            has_high_solar_activity=has_high_activity,
            filaments=filaments,
            rle_mask=rle,
            binary_mask_base64=mask_b64,
            inference_time_ms=ms
        )
        return resp, bin_mask

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str = "observation.jpg",
        threshold: float = 0.4
    ) -> FullAnalysisResponse:
        """Runs Stage 1 + Stage 2, generating annotated visual overlay."""
        t_start = time.time()
        img_orig, image_tensor, orig_shape = self.preprocess_image(image_bytes)

        # Stage 1: Basic CNN
        stage1_res = self.detect_presence(image_tensor)

        # Stage 2: U-Net
        stage2_res, bin_mask = self.segment_filaments(image_tensor, orig_shape, threshold=threshold)

        # Generate overlay image
        filaments_dict_list = [f.model_dump() for f in stage2_res.filaments]
        overlay_np = create_detection_overlay(img_orig, bin_mask, filaments_dict_list)

        # Encode overlay to base64 JPEG
        success, buffer = cv2.imencode(".jpg", overlay_np, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        overlay_base64 = None
        if success:
            overlay_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        total_ms = round((time.time() - t_start) * 1000, 2)

        return FullAnalysisResponse(
            success=True,
            image_metadata={
                "filename": filename,
                "original_height": orig_shape[0],
                "original_width": orig_shape[1],
                "device": str(self.device)
            },
            stage1_cnn=stage1_res,
            stage2_segmentation=stage2_res,
            overlay_image_base64=overlay_base64,
            total_processing_time_ms=total_ms
        )

    def process_telecast_frame(
        self,
        frame_base64: str,
        threshold: float = 0.4,
        return_overlay: bool = True
    ) -> TelecastFrameResponse:
        """Processes a single telecast frame in real-time."""
        t_start = time.time()

        # Clean base64 header if present
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",", 1)[1]

        frame_bytes = base64.b64decode(frame_base64)
        img_orig, image_tensor, orig_shape = self.preprocess_image(frame_bytes)

        stage2_res, bin_mask = self.segment_filaments(image_tensor, orig_shape, threshold=threshold)

        overlay_base64 = None
        if return_overlay:
            filaments_dict_list = [f.model_dump() for f in stage2_res.filaments]
            overlay_np = create_detection_overlay(img_orig, bin_mask, filaments_dict_list)
            success, buffer = cv2.imencode(".jpg", overlay_np, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                overlay_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        total_ms = round((time.time() - t_start) * 1000, 2)
        has_high_activity = stage2_res.num_filaments_found >= 3 or stage2_res.total_filament_area > 5000

        return TelecastFrameResponse(
            success=True,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            filaments_detected=stage2_res.num_filaments_found,
            total_filament_area=stage2_res.total_filament_area,
            has_high_solar_activity=has_high_activity,
            overlay_frame_base64=overlay_base64,
            processing_time_ms=total_ms
        )
