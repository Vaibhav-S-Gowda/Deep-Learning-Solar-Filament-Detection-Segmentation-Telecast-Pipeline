import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path to allow seamless imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.routes import inference, telecast, dataset_routes, train_routes
from backend.app.routes.inference import get_pipeline_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload models into memory
    print("[Startup] Initializing Solar Filament ML Pipeline Service...")
    service = get_pipeline_service()
    print(f"[Startup] Engine ready on device: {service.device} (CNN loaded: {service.cnn_loaded}, UNet loaded: {service.unet_loaded})")
    yield
    print("[Shutdown] Cleaning up pipeline resources...")


app = FastAPI(
    title="Solar Filament Detection & Segmentation API",
    description=(
        "Next.js-compatible REST backend for real-time solar filament detection and segmentation "
        "from GONG H-Alpha observations and live telescope telecasts."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(inference.router)
app.include_router(telecast.router)
app.include_router(dataset_routes.router)
app.include_router(train_routes.router)


@app.get("/", summary="Root API Information")
async def root():
    return {
        "name": "Solar Filament Detection & Telecast Segmentation API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "detect": "POST /api/v1/detect (Stage 1 Basic CNN presence)",
            "segment": "POST /api/v1/segment (Stage 2 U-Net segmentation)",
            "analyze": "POST /api/v1/analyze (Full pipeline + base64 overlay)",
            "telecast_frame": "POST /api/v1/telecast/process-frame (Live stream processing)",
            "telecast_sample": "GET /api/v1/telecast/sample-frame (Simulated live frame)",
            "dataset_stats": "GET /api/v1/dataset/stats (MAGFiLO dataset metrics)",
            "dataset_sample": "GET /api/v1/dataset/sample (Ground-truth overlay sample)",
            "training_start": "POST /api/v1/train/start (Background fine-tuning)",
            "training_status": "GET /api/v1/train/status (Real-time progress)"
        }
    }


@app.get("/health", summary="Health Check")
async def health_check():
    service = get_pipeline_service()
    return {
        "status": "healthy",
        "device": str(service.device),
        "stage1_cnn_ready": service.cnn_loaded,
        "stage2_unet_ready": service.unet_loaded
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
