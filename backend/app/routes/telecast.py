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
from backend.app.services.gong_fetcher import (
    fetch_latest_frame,
    list_available_stations_today,
    STATION_CODES,
    _list_frames,
    _directory_url,
)

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


# ─────────────────────────────────────────────────────────────────
# REAL-TIME LIVE GONG ENDPOINTS  (no API key, fully public server)
# Base URL: https://gong2.nso.edu/HA/has/YYYYMM/YYYYMMDD/
# ─────────────────────────────────────────────────────────────────

@router.get("/live/stations", summary="List GONG stations with live frame availability RIGHT NOW")
async def get_live_stations():
    """
    Polls the GONG server for today's available stations and frame counts.
    Returns real-time data — no API key required.
    """
    try:
        stations = list_available_stations_today()
        return {"stations": stations, "source": "GONG live server (gong2.nso.edu)"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"GONG server unreachable: {e}")


@router.get("/live/latest-frame", summary="Fetch the most recent GONG H-alpha frame from RIGHT NOW")
async def get_live_latest_frame(
    station: str = Query("all", description="Station code: Bh, Ch, Lh, Mh, Th, Uh, or 'all' for any"),
    threshold: float = Query(0.4, ge=0.1, le=0.9),
    overlay: bool = Query(True),
    service: SolarPipelineService = Depends(get_pipeline_service),
):
    try:
        filename, image_bytes, obs_time = await asyncio.to_thread(fetch_latest_frame, station)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    st_code = filename[14:16] if len(filename) >= 16 else "??"
    st_info = STATION_CODES.get(st_code, {"name": "GONG Network", "station_id": st_code, "location": "Global"})

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_orig = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise HTTPException(status_code=500, detail="Failed to decode live GONG frame")

    img_512 = cv2.resize(img_orig, (512, 512), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_512, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    raw_bytes = buf.tobytes()
    frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode()}"

    t0 = time.time()
    _, tensor, orig_shape = service.preprocess_image(raw_bytes)
    seg_res, bin_mask = service.segment_filaments(tensor, orig_shape, threshold=threshold, img_orig=img_512)

    overlay_b64 = None
    if overlay:
        fils_dict = [f.model_dump() for f in seg_res.filaments]
        overlay_np = create_detection_overlay(img_512, bin_mask, fils_dict)
        ret, buf_ov = cv2.imencode(".jpg", overlay_np, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ret:
            overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_ov).decode()}"

    proc_ms = round((time.time() - t0) * 1000, 1)

    return {
        "success": True,
        "is_live": True,
        "source": "GONG real-time server (gong2.nso.edu)",
        "file_name": filename,
        "observation_time": obs_time,
        "station_code": st_code,
        "station_name": st_info["name"],
        "station_location": st_info.get("location", ""),
        "width": 512,
        "height": 512,
        "frame_base64": frame_b64,
        "overlay_base64": overlay_b64,
        "filaments_detected": seg_res.num_filaments_found,
        "total_filament_area": seg_res.total_filament_area,
        "has_high_solar_activity": seg_res.has_high_solar_activity,
        "processing_time_ms": proc_ms,
    }


@router.get("/stream", summary="Real-Time Live Solar Stream from GONG Network")
@router.get("/live/stream", summary="Real-Time Live Solar Stream from GONG Network")
async def live_gong_stream(
    station: str = Query("all", description="Station code: Bh, Ch, Lh, Mh, Th, Uh, or 'all'"),
    fps: int = Query(2, ge=1, le=5),
    threshold: float = Query(0.4, ge=0.1, le=0.9),
    overlay: bool = Query(True),
    hud: bool = Query(True),
    service: SolarPipelineService = Depends(get_pipeline_service),
):
    frame_delay = 1.0 / max(1, min(fps, 5))

    async def generate():
        last_filename = None
        cached_fb = None

        while True:
            t0 = time.time()
            try:
                filename, image_bytes, obs_time = await asyncio.to_thread(fetch_latest_frame, station)
                is_new = (filename != last_filename)

                if not is_new and cached_fb is not None:
                    # Serve cached frame without recomputing neural inference
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(cached_fb)).encode() + b"\r\n\r\n" +
                        cached_fb + b"\r\n"
                    )
                    elapsed = time.time() - t0
                    await asyncio.sleep(max(0.1, frame_delay - elapsed))
                    continue

                last_filename = filename
                st_code = filename[14:16] if len(filename) >= 16 else "??"
                st_info = STATION_CODES.get(st_code, {"name": "GONG Network", "station_id": st_code})

                nparr = np.frombuffer(image_bytes, np.uint8)
                img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    await asyncio.sleep(0.5)
                    continue

                img_512 = cv2.resize(img_gray, (512, 512), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", img_512, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

                if overlay:
                    _, tensor, orig_shape = service.preprocess_image(buf.tobytes())
                    seg_res, bin_mask = service.segment_filaments(tensor, orig_shape, threshold=threshold, img_orig=img_512)
                    fils_dict = [f.model_dump() for f in seg_res.filaments]
                    frame_bgr = create_detection_overlay(img_512, bin_mask, fils_dict)
                else:
                    frame_bgr = cv2.cvtColor(img_512, cv2.COLOR_GRAY2BGR)
                    seg_res = type("R", (), {"num_filaments_found": 0, "total_filament_area": 0, "has_high_solar_activity": False})()

                if hud:
                    # Top bar
                    cv2.rectangle(frame_bgr, (0, 0), (512, 34), (10, 16, 26), -1)
                    cv2.line(frame_bgr, (0, 34), (512, 34), (45, 60, 78), 1)
                    cv2.circle(frame_bgr, (14, 17), 5, (0, 220, 80), -1)
                    cv2.putText(frame_bgr,
                        f"GONG LIVE | {st_info['station_id']} - {st_info['name'][:22]}",
                        (26, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 235, 245), 1, cv2.LINE_AA)

                    # Bottom bar
                    cv2.rectangle(frame_bgr, (0, 478), (512, 512), (10, 16, 26), -1)
                    cv2.line(frame_bgr, (0, 478), (512, 478), (45, 60, 78), 1)
                    act_label = "HIGH ACTIVITY" if seg_res.has_high_solar_activity else "NOMINAL"
                    act_color = (98, 187, 244) if seg_res.has_high_solar_activity else (134, 240, 206)
                    cv2.putText(frame_bgr,
                        f"{obs_time}  |  FILAMENTS: {seg_res.num_filaments_found}  |  AREA: {int(seg_res.total_filament_area):,} px",
                        (10, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (180, 195, 210), 1, cv2.LINE_AA)
                    cv2.putText(frame_bgr, act_label,
                        (415, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.31, act_color, 1, cv2.LINE_AA)

                ret, buf_out = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret:
                    cached_fb = buf_out.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(cached_fb)).encode() + b"\r\n\r\n" +
                        cached_fb + b"\r\n"
                    )

            except Exception as e:
                err_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                cv2.putText(err_frame, "GONG Live: Polling server...", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 120), 1, cv2.LINE_AA)
                cv2.putText(err_frame, str(e)[:65], (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
                _, buf_err = cv2.imencode(".jpg", err_frame)
                fb = buf_err.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(fb)).encode() + b"\r\n\r\n" +
                    fb + b"\r\n"
                )

            elapsed = time.time() - t0
            await asyncio.sleep(max(0.1, frame_delay - elapsed))

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )
