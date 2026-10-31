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

    # Draw ground-truth polygons with glowing bloom
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt_mask = np.zeros_like(vis)

    for idx, ann in enumerate(anns):
        for poly in ann.get("segmentation", []):
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(gt_mask, [pts], (0, 255, 140))  # Vibrant emerald/spring green

    if np.any(gt_mask > 0):
        # Radiant bloom glow
        glow_wide = cv2.GaussianBlur(gt_mask, (31, 31), 0)
        glow_tight = cv2.GaussianBlur(gt_mask, (11, 11), 0)
        bloom = cv2.addWeighted(glow_wide, 0.7, glow_tight, 0.85, 0)
        bloom = cv2.addWeighted(bloom, 1.0, gt_mask, 0.65, 0)
        vis = cv2.addWeighted(vis, 1.0, bloom, 0.95, 0)

        # Draw contour perimeter highlight
        gt_gray = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gt_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (200, 255, 230), 1, cv2.LINE_AA)

    # Integrated HUD Badge showing total filament count
    num_anns = len(anns)
    badge_label = f"FILAMENTS: {num_anns}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (b_w, b_h), _ = cv2.getTextSize(badge_label, font, font_scale, thickness)
    badge_w = b_w + 34
    badge_h = b_h + 16
    h_v, w_v = vis.shape[:2]
    bx1 = w_v - badge_w - 14
    by1 = 14
    bx2 = w_v - 14
    by2 = by1 + badge_h

    if bx1 > 0 and by2 < h_v:
        overlay_box = vis.copy()
        cv2.rectangle(overlay_box, (bx1, by1), (bx2, by2), (14, 20, 27), cv2.FILLED)
        cv2.addWeighted(overlay_box, 0.88, vis, 0.12, 0, vis)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 140), 1, cv2.LINE_AA)
        dot_x = bx1 + 12
        dot_y = by1 + badge_h // 2
        cv2.circle(vis, (dot_x, dot_y), 4, (0, 255, 140), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(vis, (dot_x, dot_y), 6, (0, 255, 140), 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            badge_label,
            (dot_x + 10, by1 + badge_h // 2 + b_h // 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    # Resize preview to 512x512
    vis_resized = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", vis_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return {
        "success": True,
        "image_id": img_id,
        "file_name": file_name,
        "date_captured": sample_img.get("date_captured"),
        "num_filaments_annotated": len(anns),
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
