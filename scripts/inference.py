#!/usr/bin/env python3
"""
Repeatable Inference Script for Solar Filament Segmentation.

Loads a trained SolarFilamentUNet checkpoint and runs inference on a single
image or a directory of solar observation images. Extracts binary masks,
COCO polygon vectors, bounding boxes, and generates annotated visual overlays.

Usage:
    python scripts/inference.py --image path/to/image.jpg
    python scripts/inference.py --image path/to/folder/ --output-dir outputs/
    python scripts/inference.py --image path/to/image.jpg --threshold 0.35 --device cpu
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.unet_model import (
    SolarFilamentUNet,
    mask_to_polygons_and_bboxes,
    binary_mask_to_rle,
    create_detection_overlay
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeatable Solar Filament U-Net Segmentation Inference"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to input image (.jpg, .jpeg, .png) or directory of images"
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "models" / "unet_segmenter.pth"),
        help="Path to trained PyTorch model checkpoint (.pth)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "inference"),
        help="Directory where output masks, overlays, and metrics will be saved"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.40,
        help="Binary segmentation probability threshold (default: 0.40)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Model input resolution (height width), default: 256 256"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device ('auto', 'cpu', 'cuda')"
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="Save colorized alpha-blended visual overlay (default: True)"
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        default=True,
        help="Save raw binary mask as PNG (default: True)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save extracted polygon geometries and metrics to JSON (default: True)"
    )
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device) -> SolarFilamentUNet:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = SolarFilamentUNet(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Support full checkpoint dict or raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.eval()
    print(f"[Model Loaded] Checkpoint: {model_path} | Device: {device}")
    return model


def preprocess_image(image_path: str, target_size: Tuple[int, int]) -> Tuple[np.ndarray, torch.Tensor, Tuple[int, int]]:
    img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise ValueError(f"Could not read image from {image_path}")

    orig_shape = img_orig.shape  # (H, W)
    th, tw = target_size
    img_resized = cv2.resize(img_orig, (tw, th), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0] and add batch & channel dims: (1, 1, H, W)
    img_norm = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return img_orig, tensor, orig_shape


def run_inference_single(
    image_path: str,
    model: SolarFilamentUNet,
    device: torch.device,
    target_size: Tuple[int, int],
    threshold: float,
    output_dir: Path,
    save_overlay: bool = True,
    save_mask: bool = True,
    save_json: bool = True
) -> Dict[str, Any]:
    file_stem = Path(image_path).stem
    t0 = time.time()

    img_orig, tensor, orig_shape = preprocess_image(image_path, target_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = outputs["probs"][0, 0].cpu().numpy()

    t_infer = (time.time() - t0) * 1000

    # Binary mask at model resolution
    binary_mask_model = (probs > threshold).astype(np.uint8)

    # Resize binary mask back to original resolution
    orig_h, orig_w = orig_shape
    binary_mask_orig = cv2.resize(
        binary_mask_model, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )

    # Extract polygon vectors, bboxes and area
    filaments_raw = mask_to_polygons_and_bboxes(
        binary_mask_model,
        min_area=15,
        approx_epsilon=1.0,
        orig_shape=orig_shape
    )

    total_area = sum(f["area"] for f in filaments_raw)
    coverage_pct = round((total_area / (orig_h * orig_w)) * 100, 4)
    rle = binary_mask_to_rle(binary_mask_model)

    results = {
        "image_file": os.path.basename(image_path),
        "original_resolution": [orig_w, orig_h],
        "inference_resolution": list(target_size),
        "threshold": threshold,
        "inference_time_ms": round(t_infer, 2),
        "num_filaments_found": len(filaments_raw),
        "total_filament_area_px": round(total_area, 2),
        "coverage_percentage": coverage_pct,
        "has_high_solar_activity": bool(total_area > 1000.0),
        "filaments": filaments_raw,
        "rle_mask": rle
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save binary mask
    if save_mask:
        mask_out_path = output_dir / f"{file_stem}_mask.png"
        cv2.imwrite(str(mask_out_path), binary_mask_orig * 255)
        results["saved_mask_path"] = str(mask_out_path)

    # Save visual overlay
    if save_overlay:
        overlay = create_detection_overlay(img_orig, binary_mask_orig, filaments_raw)
        overlay_out_path = output_dir / f"{file_stem}_overlay.jpg"
        cv2.imwrite(str(overlay_out_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        results["saved_overlay_path"] = str(overlay_out_path)

    # Save JSON metadata
    if save_json:
        json_out_path = output_dir / f"{file_stem}_metrics.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        results["saved_json_path"] = str(json_out_path)

    return results


def main():
    args = parse_args()
    device = get_device(args.device)
    model = load_model(args.model_path, device)
    output_dir = Path(args.output_dir)

    target_size = (args.target_size[0], args.target_size[1])

    # Determine input files
    input_path = Path(args.image)
    if input_path.is_dir():
        image_files = sorted([
            str(p) for p in input_path.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])
        if not image_files:
            print(f"[Error] No supported images found in directory: {input_path}")
            sys.exit(1)
        print(f"[Batch Mode] Found {len(image_files)} images to process.")
    elif input_path.is_file():
        image_files = [str(input_path)]
    else:
        print(f"[Error] Input path does not exist: {input_path}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f" SOLAR FILAMENT SEGMENTATION INFERENCE ENGINE")
    print("=" * 65)

    all_results = []
    total_t0 = time.time()

    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(img_file)} ...", end=" ")
        try:
            res = run_inference_single(
                image_path=img_file,
                model=model,
                device=device,
                target_size=target_size,
                threshold=args.threshold,
                output_dir=output_dir,
                save_overlay=args.save_overlay,
                save_mask=args.save_mask,
                save_json=args.save_json
            )
            all_results.append(res)
            print(f"Done in {res['inference_time_ms']} ms | Found: {res['num_filaments_found']} filaments | Area: {res['total_filament_area_px']} px")
        except Exception as e:
            print(f"FAILED: {e}")

    total_time = time.time() - total_t0
    print("-" * 65)
    print(f"Summary: Processed {len(all_results)}/{len(image_files)} images in {total_time:.2f} s")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
