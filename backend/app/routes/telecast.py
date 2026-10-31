import os
import time
import random
import base64
import asyncio
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from backend.app.schemas import TelecastFrameRequest, TelecastFrameResponse
from backend.app.services.pipeline_service import SolarPipelineService
from backend.app.routes.inference import get_pipeline_service
from backend.ml.unet_model import create_detection_overlay

router = APIRouter(prefix="/api/v1/telecast", tags=["Live Telecast Processing"])

SAMPLE_IMAGES_DIR = r"c:\Users\vaibh\Desktop\Solar-Filament\MAGFiLO_1.0_Kaggle_2026\train\train_images"

GONG_STATIONS = {
    "Bh": {
        "code": "Bh",
        "name": "Big Bear Solar Observatory",
        "location": "California, USA",
        "station_id": "BBSO"
    },
    "Ch": {
        "code": "Ch",
        "name": "Cerro Tololo Inter-American Observatory",
        "location": "La Serena, Chile",
        "station_id": "CTIO"
    },
    "Lh": {
        "code": "Lh",
        "name": "Learmonth Solar Observatory",
        "location": "Western Australia",
        "station_id": "LEAR"
    },
    "Mh": {
        "code": "Mh",
        "name": "Mauna Loa Solar Observatory",
        "location": "Hawaii, USA",
        "station_id": "MLSO"
    },
    "Th": {
        "code": "Th",
        "name": "Teide Observatory",
        "location": "Canary Islands, Spain",
        "station_id": "OT"
    },
    "Uh": {
        "code": "Uh",
        "name": "Udaipur Solar Observatory",
        "location": "Rajasthan, India",
        "station_id": "USO"
    }
}


def _get_sorted_station_files(station: str = "all"):
    """Returns chronologically sorted observation files for a station or all stations."""
    if not os.path.exists(SAMPLE_IMAGES_DIR):
        return []
    all_files = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith((".jpeg", ".jpg"))]
    if station and station != "all" and station in GONG_STATIONS:
        filtered = [f for f in all_files if f.endswith(f"{station}.jpeg") or f.endswith(f"{station}.jpg")]
        filtered.sort()
        return filtered
    all_files.sort()
    return all_files


def _parse_filename_date(fn: str) -> str:
    """Extracts ISO-like timestamp from filename (YYYYMMDDHHMMSS[Station]h.jpeg)."""
    try:
        # e.g. 20110109104734Ch.jpeg
        base = os.path.splitext(fn)[0]
        digits = base[:14]
        if len(digits) == 14 and digits.isdigit():
            y, m, d = digits[0:4], digits[4:6], digits[6:8]
            hh, mm, ss = digits[8:10], digits[10:12], digits[12:14]
            return f"{y}-{m}-{d} {hh}:{mm}:{ss} UTC"
    except Exception:
        pass
    return "LIVE OBSERVATION"


@router.get("/stations", summary="List GONG Telescope Stations")
async def get_telecast_stations():
    """Returns available GONG telescope stations with frame counts and observatory info."""
    result = []
    for code, info in GONG_STATIONS.items():
        files = _get_sorted_station_files(code)
        first_date = _parse_filename_date(files[0]) if files else None
        last_date = _parse_filename_date(files[-1]) if files else None
        result.append({
            **info,
            "total_frames": len(files),
            "start_date": first_date,
            "end_date": last_date
        })
    return {
        "stations": result,
        "total_dataset_frames": len(_get_sorted_station_files("all"))
    }


@router.get("/stream", summary="Real-Time Continuous Solar Telecast Video Stream (MJPEG)")
async def live_telecast_stream(
    station: str = Query("Ch", description="Telescope station ('Bh', 'Ch', 'Lh', 'Mh', 'Th', 'Uh', 'all')"),
    fps: int = Query(10, ge=1, le=30, description="Target streaming framerate (frames per second)"),
    threshold: float = Query(0.4, ge=0.1, le=0.9, description="Filament segmentation threshold"),
    overlay: bool = Query(True, description="Whether to overlay filament segmentation masks"),
    hud: bool = Query(True, description="Whether to burn observatory HUD graphics onto the video stream"),
    start_index: int = Query(0, ge=0, description="Starting frame index in sequence (supports pause/resume)"),
    service: SolarPipelineService = Depends(get_pipeline_service)
):
    """
    True real-time continuous video stream of the Sun.
    Streams chronologically ordered H-Alpha solar observations at continuous framerates (10-30 FPS)
    with real-time U-Net filament segmentation and observatory telemetry burned into the video.
    Directly compatible with HTML `<img src="..." />`, `<canvas>`, and video streaming clients.
    """
    files = _get_sorted_station_files(station)
    if not files:
        raise HTTPException(status_code=404, detail="No frames available for requested station")

    frame_delay = 1.0 / max(1, min(fps, 30))

    async def generate_mjpeg_stream():
        idx = start_index
        total = len(files)
        try:
            while True:
                t0 = time.time()
                filename = files[idx % total]
                file_path = os.path.join(SAMPLE_IMAGES_DIR, filename)

                img_orig = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img_orig is not None:
                    st_code = filename[-7:-5] if len(filename) >= 7 else "GONG"
                    st_info = GONG_STATIONS.get(st_code, {"name": "GONG Network", "station_id": st_code})
                    obs_time = _parse_filename_date(filename)

                    # Resize to standard 512x512 telecast transmission resolution
                    img_512 = cv2.resize(img_orig, (512, 512), interpolation=cv2.INTER_AREA)

                    # Run Stage 2 U-Net segmentation
                    _, tensor, orig_shape = service.preprocess_image(cv2.imencode(".jpg", img_512)[1].tobytes())
                    seg_res, bin_mask = service.segment_filaments(tensor, orig_shape, threshold=threshold)

                    # Build frame
                    if overlay:
                        filaments_dict_list = [f.model_dump() for f in seg_res.filaments]
                        frame_bgr = create_detection_overlay(img_512, bin_mask, filaments_dict_list)
                    else:
                        frame_bgr = cv2.cvtColor(img_512, cv2.COLOR_GRAY2BGR)

                    if hud:
                        # Top telemetry HUD bar
                        cv2.rectangle(frame_bgr, (0, 0), (512, 34), (10, 16, 26), -1)
                        cv2.line(frame_bgr, (0, 34), (512, 34), (45, 60, 78), 1)
                        # Blinking LIVE dot
                        dot_color = (134, 240, 206) if (int(time.time() * 2) % 2 == 0) else (40, 160, 130)
                        cv2.circle(frame_bgr, (14, 17), 5, dot_color, -1)
                        cv2.putText(
                            frame_bgr,
                            f"LIVE TELECAST | {st_info['station_id']} - {st_info['name'][:22]}",
                            (26, 21),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.38,
                            (220, 235, 245),
                            1,
                            cv2.LINE_AA
                        )

                        # Bottom telemetry HUD bar
                        cv2.rectangle(frame_bgr, (0, 478), (512, 512), (10, 16, 26), -1)
                        cv2.line(frame_bgr, (0, 478), (512, 478), (45, 60, 78), 1)

                        act_label = "HIGH ACTIVITY" if seg_res.has_high_solar_activity else "NOMINAL"
                        act_color = (98, 187, 244) if seg_res.has_high_solar_activity else (134, 240, 206)

                        cv2.putText(
                            frame_bgr,
                            f"{obs_time}  |  FILAMENTS: {seg_res.num_filaments_found}  |  AREA: {int(seg_res.total_filament_area):,} px",
                            (10, 498),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.34,
                            (180, 195, 210),
                            1,
                            cv2.LINE_AA
                        )
                        cv2.putText(
                            frame_bgr,
                            act_label,
                            (410, 498),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.34,
                            act_color,
                            1,
                            cv2.LINE_AA
                        )

                    # JPEG encode
                    ret, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" +
                            frame_bytes + b"\r\n"
                        )

                idx += 1
                elapsed = time.time() - t0
                sleep_time = max(0.005, frame_delay - elapsed)
                await asyncio.sleep(sleep_time)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/sequence-frame", summary="Get Sequential Chronological Telecast Frame")
async def get_sequence_frame(
    station: str = Query("Ch", description="Station code ('Bh', 'Ch', 'Lh', 'Mh', 'Th', 'Uh', 'all')"),
    index: int = Query(0, ge=0, description="Sequential frame index"),
    threshold: float = Query(0.4, ge=0.1, le=0.9),
    return_overlay: bool = Query(True),
    service: SolarPipelineService = Depends(get_pipeline_service)
):
    """
    Returns the next chronological observation frame for a telescope station with live U-Net segmentation.
    Allows smooth sequential playback with chronological progression instead of random image jumping.
    """
    files = _get_sorted_station_files(station)
    if not files:
        raise HTTPException(status_code=404, detail="No images found for this station")

    cur_idx = index % len(files)
    chosen_file = files[cur_idx]
    file_path = os.path.join(SAMPLE_IMAGES_DIR, chosen_file)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read observation frame")

    st_code = chosen_file[-7:-5] if len(chosen_file) >= 7 else "GONG"
    st_info = GONG_STATIONS.get(st_code, {"name": "GONG Network", "station_id": st_code, "location": "Global"})
    obs_time = _parse_filename_date(chosen_file)

    img_sim = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img_sim, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    raw_bytes = buffer.tobytes()
    frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

    # Perform segmentation
    t0 = time.time()
    _, tensor, orig_shape = service.preprocess_image(raw_bytes)
    seg_res, bin_mask = service.segment_filaments(tensor, orig_shape, threshold=threshold)

    overlay_b64 = None
    if return_overlay:
        filaments_dict_list = [f.model_dump() for f in seg_res.filaments]
        overlay_np = create_detection_overlay(img_sim, bin_mask, filaments_dict_list)
        ret, buf_ov = cv2.imencode(".jpg", overlay_np, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_ov).decode('utf-8')}"

    proc_time = round((time.time() - t0) * 1000, 1)

    return {
        "success": True,
        "index": cur_idx,
        "next_index": (cur_idx + 1) % len(files),
        "total_frames": len(files),
        "file_name": chosen_file,
        "station_code": st_code,
        "station_name": st_info["name"],
        "station_location": st_info.get("location", ""),
        "observation_time": obs_time,
        "width": 512,
        "height": 512,
        "frame_base64": frame_b64,
        "overlay_base64": overlay_b64,
        "filaments_detected": seg_res.num_filaments_found,
        "total_filament_area": seg_res.total_filament_area,
        "has_high_solar_activity": seg_res.has_high_solar_activity,
        "processing_time_ms": proc_time
    }


@router.post("/process-frame", response_model=TelecastFrameResponse, summary="Process Live Telecast Frame")
async def process_telecast_frame(
    req: TelecastFrameRequest,
    service: SolarPipelineService = Depends(get_pipeline_service)
):
    """
    Real-time frame ingestion endpoint for live telescope video streams, webcam, and video player feeds.
    Accepts base64-encoded solar images, segments active filaments, and returns annotated overlay.
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


@router.get("/sample-frame", summary="Get Sample Live Telecast Frame")
async def get_sample_telecast_frame(
    station: Optional[str] = Query(None, description="Optional station filter")
):
    """
    Fetches a real H-Alpha solar observation image from the dataset as a base64 frame.
    Supports station selection for consecutive monitoring.
    """
    files = _get_sorted_station_files(station if station else "all")
    if not files:
        raise HTTPException(status_code=404, detail="No images found in dataset directory")

    chosen_file = random.choice(files)
    file_path = os.path.join(SAMPLE_IMAGES_DIR, chosen_file)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read sample image")

    img_sim = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img_sim, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    st_code = chosen_file[-7:-5] if len(chosen_file) >= 7 else "GONG"

    return {
        "success": True,
        "file_name": chosen_file,
        "observation_time": _parse_filename_date(chosen_file),
        "width": 512,
        "height": 512,
        "frame_base64": frame_b64,
        "telescope_station": st_code
    }


@router.get("/status", summary="Telecast Stream Status")
async def get_telecast_status(service: SolarPipelineService = Depends(get_pipeline_service)):
    """Returns streaming health, accelerator device, and model readiness."""
    return {
        "status": "ready",
        "device": str(service.device),
        "stage1_cnn_ready": service.cnn_loaded,
        "stage2_unet_ready": service.unet_loaded,
        "recommended_fps": 15 if service.device.type == "cpu" else 30
    }
