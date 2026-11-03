import os
import json
import random
import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from backend.app.schemas import DatasetStatsResponse

router = APIRouter(prefix="/api/v1/dataset", tags=["Dataset & Metadata"])

JSON_PATH = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG_DIR = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\train_images"
TEST_IMG_DIR = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\test\test_images"


@router.get("/stats", response_model=DatasetStatsResponse, summary="MAGFiLO Dataset Statistics")
async def get_dataset_stats():
    """Returns dataset summary, image counts, and annotation categories."""
    if not os.path.exists(JSON_PATH):
        raise HTTPException(status_code=404, detail="Dataset JSON not found")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    cat_counts = {name: 0 for name in cat_map.values()}

    for ann in coco_data.get("annotations", []):
        cid = ann.get("category_id")
        if cid in cat_map:
            cat_counts[cat_map[cid]] += 1

    train_files = len(os.listdir(TRAIN_IMG_DIR)) if os.path.exists(TRAIN_IMG_DIR) else 0
    test_files = len(os.listdir(TEST_IMG_DIR)) if os.path.exists(TEST_IMG_DIR) else 0

    return DatasetStatsResponse(
        dataset_name="MAGFiLO 1.0 (GONG H-Alpha)",
        total_train_images=train_files,
        total_test_images=test_files,
        total_annotations=len(coco_data.get("annotations", [])),
        category_counts=cat_counts,
        image_resolution="2048 x 2048",
        instrument="Global Oscillations Network Group (GONG) H-Alpha"
    )


@router.get("/sample", summary="Get Ground-Truth Sample with Annotations")
async def get_dataset_sample():
    """
    Returns a sample image from MAGFiLO with its ground-truth polygon annotations rasterized.
    """
    if not os.path.exists(JSON_PATH) or not os.path.exists(TRAIN_IMG_DIR):
        raise HTTPException(status_code=404, detail="Dataset files not found")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    img_to_anns = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    existing_images = []
    for img_info in coco_data.get("images", []):
        fn = img_info["file_name"]
        if os.path.exists(os.path.join(TRAIN_IMG_DIR, fn)):
            existing_images.append(img_info)

    if not existing_images:
        raise HTTPException(status_code=404, detail="No existing images found on disk")

    sample_img = random.choice(existing_images)
    img_id = sample_img["id"]
    file_name = sample_img["file_name"]
    anns = img_to_anns.get(img_id, [])

    img_path = os.path.join(TRAIN_IMG_DIR, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    # 1. Pure raw grayscale solar observation (clean unannotated frame)
    raw_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    _, buf_raw = cv2.imencode(".jpg", raw_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    raw_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_raw).decode('utf-8')}"

    # 2. Draw ground-truth polygons in warm Solar Amber/Gold (BGR: [20, 140, 255])
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt_mask = np.zeros_like(vis)

    for idx, ann in enumerate(anns):
        for poly in ann.get("segmentation", []):
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(gt_mask, [pts], (20, 140, 255))  # Solar Amber

    if np.any(gt_mask > 0):
        # Subtle, clean amber overlay
        cv2.addWeighted(gt_mask, 0.40, vis, 0.60, 0, vis)

        # Draw crisp golden boundary contours (BGR: [0, 190, 255])
        gt_gray = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gt_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 190, 255), 1, cv2.LINE_AA)

        # Draw subtle golden bounding boxes with filament labels
        for idx, ann in enumerate(anns):
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                bx, by, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 185, 255), 1, cv2.LINE_AA)
                tag_label = f"#{idx + 1}"
                (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                tag_y = max(th + 4, by - 4)
                cv2.rectangle(vis, (bx, tag_y - th - 3), (bx + tw + 6, tag_y + 3), (14, 20, 27), cv2.FILLED)
                cv2.putText(vis, tag_label, (bx + 3, tag_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 215, 0), 1, cv2.LINE_AA)

    # Resize annotated preview to 512x512
    vis_resized = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", vis_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return {
        "success": True,
        "image_id": img_id,
        "file_name": file_name,
        "date_captured": sample_img.get("date_captured"),
        "num_filaments_annotated": len(anns),
        "raw_base64": raw_b64,
        "overlay_base64": overlay_b64,
        "annotations": [
            {
                "id": a.get("id"),
                "category_id": a.get("category_id"),
                "bbox": a.get("bbox"),
                "area": a.get("area")
            }
            for a in anns
        ]
    }
