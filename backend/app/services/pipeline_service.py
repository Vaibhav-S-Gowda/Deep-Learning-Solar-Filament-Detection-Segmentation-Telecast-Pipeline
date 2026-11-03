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
    create_detection_overlay,
    create_non_solar_rejection_overlay
)
from backend.app.schemas import (
    CNNDetectionResponse,
    CategoryScore,
    SegmentationResponse,
    FilamentPolygonAnnotation,
    FilamentMorphology,
    FilamentDiagnosticQA,
    FullAnalysisResponse,
    TelecastFrameResponse
)


def validate_solar_observation(img_orig: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates whether the input image is an authentic full-disk solar observation:
    1. Aspect Ratio: Near-square (telescope field of view, aspect <= 1.32).
    2. Space Background: Outer corners/borders must be dark space (mean < 95, not bright/white background).
    3. Solar Disc Geometry: Circular body with area occupying 30% - 95% of frame,
       circular enclosing extent >= 0.82, centered within 14% of image center.
    4. Chromospheric Texture: Sensor intensity inside disk matches H-alpha telescope distribution.
    Rejects arbitrary everyday images (portraits, animals, clothes, landscapes, documents, cartoons).
    """
    if len(img_orig.shape) == 3:
        gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_orig

    h, w = gray.shape

    # 1. Aspect ratio check
    aspect = max(h, w) / max(1, min(h, w))
    if aspect > 1.32:
        return False, f"Image aspect ratio ({w}x{h}, aspect={aspect:.2f}) does not match solar telescope framing (expected ~1:1 square).", {}

    # 2. Outer border brightness check (space surrounding sun must be dark)
    bx = max(2, w // 25)
    by = max(2, h // 25)
    border_pixels = np.concatenate([
        gray[0:by, :].flatten(),
        gray[-by:, :].flatten(),
        gray[:, 0:bx].flatten(),
        gray[:, -bx:].flatten()
    ])
    border_mean = float(np.mean(border_pixels))
    center_region = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    center_mean = float(np.mean(center_region))

    if border_mean > 95 and border_mean >= center_mean * 0.85:
        return False, f"Image has a bright background (border={border_mean:.1f}, center={center_mean:.1f}). Solar observations require dark space surrounding the disk.", {}

    # 3. Detect the solar disk via contour analysis
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return False, "No distinct bright astronomical body detected against background space.", {}

    largest = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    total_area = float(h * w)
    area_ratio = area / total_area

    if area_ratio < 0.30 or area_ratio > 0.95:
        return False, f"Detected object area ratio ({area_ratio:.2f}) is outside normal solar disk range (0.30 - 0.95).", {}

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    circle_area = np.pi * (radius ** 2)
    extent = area / circle_area if circle_area > 0 else 0.0
    center_dist = np.sqrt((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) / float(min(h, w))

    if extent < 0.82:
        return False, f"Detected body is non-circular (circular extent={extent:.2f}, expected >= 0.82 for Sun).", {}

    if center_dist > 0.14:
        return False, f"Solar disk is off-center (offset={center_dist:.2f}, expected <= 0.14).", {}

    # 4. Chromospheric Texture check
    disk_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disk_mask, (int(cx), int(cy)), int(radius * 0.88), 255, -1)
    disk_pixels = gray[disk_mask == 255]
    if len(disk_pixels) > 0:
        d_std = float(np.std(disk_pixels))
        d_mean = float(np.mean(disk_pixels))
        if d_std < 4.0:
            return False, "Object lacks chromospheric surface variation (uniform solid shape).", {}
        if d_mean < 35 or d_mean > 245:
            return False, f"Mean intensity ({d_mean:.1f}) outside valid solar sensor range.", {}

    return True, "Valid H-alpha full-disk solar observation", {
        "center": (int(cx), int(cy)),
        "radius": int(radius),
        "extent": round(extent, 3),
        "area_ratio": round(area_ratio, 3)
    }


def locate_solar_disk(img_gray: np.ndarray) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    Locates the solar disk center (cx, cy) and radius R using Otsu thresholding.
    Robust to corner observatory logos, header text, and dark background space.
    """
    try:
        blurred = cv2.GaussianBlur(img_gray, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        largest = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        h, w = img_gray.shape[:2]
        if area < 0.20 * (h * w):
            return None
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        return ((float(cx), float(cy)), float(radius))
    except Exception:
        return None


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

    def detect_presence(self, image_tensor: torch.Tensor, img_orig: Optional[np.ndarray] = None) -> CNNDetectionResponse:
        """Stage 1: Runs basic CNN to detect presence and dominant category."""
        t0 = time.time()

        if img_orig is not None:
            is_solar, val_reason, _ = validate_solar_observation(img_orig)
            if not is_solar:
                ms = round((time.time() - t0) * 1000, 2)
                return CNNDetectionResponse(
                    success=True,
                    is_solar_observation=False,
                    validation_message=val_reason,
                    has_filament=False,
                    filament_probability=0.0,
                    dominant_category="None",
                    category_scores=[
                        CategoryScore(category_id=i, name=self.CATEGORIES.get(i, f"Class_{i}"), probability=1.0 if i == 0 else 0.0)
                        for i in range(5)
                    ],
                    inference_time_ms=ms
                )

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
            is_solar_observation=True,
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
        threshold: float = 0.4,
        img_orig: Optional[np.ndarray] = None
    ) -> Tuple[SegmentationResponse, np.ndarray]:
        """Stage 2: Runs U-Net to extract fine-grained segmentation masks and polygons."""
        t0 = time.time()

        if img_orig is not None:
            is_solar, val_reason, _ = validate_solar_observation(img_orig)
            if not is_solar:
                ms = round((time.time() - t0) * 1000, 2)
                return SegmentationResponse(
                    success=True,
                    is_solar_observation=False,
                    num_filaments_found=0,
                    total_filament_area=0.0,
                    coverage_percentage=0.0,
                    has_high_solar_activity=False,
                    filaments=[],
                    rle_mask=None,
                    binary_mask_base64=None,
                    inference_time_ms=ms
                ), np.zeros(orig_shape, dtype=np.uint8)

        with torch.no_grad():
            outputs = self.unet_model(image_tensor)

        probs = outputs["probs"][0, 0].cpu().numpy()
        h_mask, w_mask = probs.shape

        # 1. Solar Disk Geometry & Masking: strictly isolate the solar disk
        # and eliminate outer space, observatory logos (NSF, NOAA), timestamp text, and limb ring
        solar_disk_geom = None
        if img_orig is not None:
            solar_disk_geom = locate_solar_disk(img_orig)

        if solar_disk_geom is None:
            tensor_gray = (image_tensor[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            tg_geom = locate_solar_disk(tensor_gray)
            if tg_geom is not None:
                (tcx, tcy), trad = tg_geom
                scale_x = orig_shape[1] / float(tensor_gray.shape[1])
                scale_y = orig_shape[0] / float(tensor_gray.shape[0])
                solar_disk_geom = ((tcx * scale_x, tcy * scale_y), trad * min(scale_x, scale_y))
            else:
                solar_disk_geom = ((orig_shape[1] / 2.0, orig_shape[0] / 2.0), min(orig_shape) * 0.45)

        (cx, cy), radius = solar_disk_geom
        sx = w_mask / float(orig_shape[1])
        sy = h_mask / float(orig_shape[0])

        # Mask disk with 0.85 * radius to exclude steep limb darkening drop-off
        disk_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
        cv2.circle(disk_mask, (int(cx * sx), int(cy * sy)), int(radius * min(sx, sy) * 0.85), 1, -1)

        # 2. Adaptive Calibration for compressed sigmoid range
        p_disk = probs[disk_mask > 0]
        if len(p_disk) > 0 and float(p_disk.max()) < 0.65:
            p_max = float(p_disk.max())
            p_mean = float(p_disk.mean())
            p_std = float(p_disk.std())
            sens = (threshold - 0.1) / 0.8  # Map [0.1, 0.9] -> [0.0, 1.0]
            eff_thresh = p_mean + (0.5 + sens * 1.5) * p_std
            eff_thresh = min(eff_thresh, p_max - 0.005)
            effective_threshold = max(0.15, eff_thresh)
        else:
            effective_threshold = threshold

        bin_mask = ((probs > effective_threshold) & (disk_mask > 0)).astype(np.uint8)

        # Filter out tiny noise, giant limb circle, and off-disk artifacts
        max_area_scaled = orig_shape[0] * orig_shape[1] * 0.15
        filaments_raw, clean_bin_mask = mask_to_polygons_and_bboxes(
            bin_mask,
            min_area=15,
            max_area=max_area_scaled,
            approx_epsilon=1.0,
            orig_shape=orig_shape,
            solar_disk=solar_disk_geom,
            return_clean_mask=True
        )
        bin_mask = clean_bin_mask

        # 3. Fallback: Disabled morphological blackhat to prevent artificial limb ring artifacts.
        # Deep U-Net provides authentic physical segmentation directly.

        filaments = [
            FilamentPolygonAnnotation(
                id=f["id"],
                segmentation=f["segmentation"],
                bbox=f["bbox"],
                area=f["area"],
                confidence=round(f["confidence"], 4),
                category_id=1,
                category_name="Solar Filament",
                morphology=FilamentMorphology(**f["morphology"]) if f.get("morphology") else None,
                diagnostic_qa=FilamentDiagnosticQA(**f["diagnostic_qa"]) if f.get("diagnostic_qa") else None,
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

        solar_meta = {
            "center": [float(cx), float(cy)],
            "radius_px": float(radius),
            "km_per_px": round(696340.0 / max(1.0, float(radius)), 2),
            "sun_radius_km": 696340.0
        }

        resp = SegmentationResponse(
            success=True,
            num_filaments_found=len(filaments),
            total_filament_area=round(total_area, 2),
            coverage_percentage=coverage_pct,
            has_high_solar_activity=has_high_activity,
            filaments=filaments,
            solar_disk_metadata=solar_meta,
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

        # 0. Solar Observation Domain Validation
        is_solar, val_reason, disk_info = validate_solar_observation(img_orig)

        if not is_solar:
            total_ms = round((time.time() - t_start) * 1000, 2)
            stage1_res = CNNDetectionResponse(
                success=True,
                is_solar_observation=False,
                validation_message=val_reason,
                has_filament=False,
                filament_probability=0.0,
                dominant_category="None",
                category_scores=[
                    CategoryScore(category_id=i, name=self.CATEGORIES.get(i, f"Class_{i}"), probability=1.0 if i == 0 else 0.0)
                    for i in range(5)
                ],
                inference_time_ms=total_ms
            )
            stage2_res = SegmentationResponse(
                success=True,
                is_solar_observation=False,
                num_filaments_found=0,
                total_filament_area=0.0,
                coverage_percentage=0.0,
                has_high_solar_activity=False,
                filaments=[],
                rle_mask=None,
                binary_mask_base64=None,
                inference_time_ms=total_ms
            )

            overlay_np = create_non_solar_rejection_overlay(img_orig, val_reason)
            success, buffer = cv2.imencode(".jpg", overlay_np, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            overlay_base64 = None
            if success:
                overlay_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

            return FullAnalysisResponse(
                success=True,
                is_solar_observation=False,
                validation_message=val_reason,
                image_metadata={
                    "filename": filename,
                    "original_height": orig_shape[0],
                    "original_width": orig_shape[1],
                    "device": str(self.device),
                    "is_solar_observation": False,
                    "rejection_reason": val_reason
                },
                stage1_cnn=stage1_res,
                stage2_segmentation=stage2_res,
                overlay_image_base64=overlay_base64,
                total_processing_time_ms=total_ms
            )

        # Stage 1: Basic CNN
        stage1_res = self.detect_presence(image_tensor, img_orig=img_orig)

        # Stage 2: U-Net
        stage2_res, bin_mask = self.segment_filaments(image_tensor, orig_shape, threshold=threshold, img_orig=img_orig)

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
            is_solar_observation=True,
            validation_message="Valid H-alpha full-disk solar observation",
            image_metadata={
                "filename": filename,
                "original_height": orig_shape[0],
                "original_width": orig_shape[1],
                "device": str(self.device),
                "is_solar_observation": True
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

        stage2_res, bin_mask = self.segment_filaments(image_tensor, orig_shape, threshold=threshold, img_orig=img_orig)

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
