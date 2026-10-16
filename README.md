# Deep Learning Solar Filament Detection, Segmentation & Telecast Pipeline

A production-grade, two-stage deep learning pipeline and REST API built with **PyTorch**, **FastAPI**, **MLflow**, and **Docker** for real-time solar filament screening, pixel-accurate U-Net segmentation, magnetic chirality classification, and live telescope telecast stream monitoring.

Tested and benchmarked against the standardized **MAGFiLO 1.0 dataset** ($1,154$ full-disk GONG H-$\alpha$ observations and $8,199$ ground-truth annotations).

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Docker Containerisation](#-docker-containerisation)
- [Repeatable Inference CLI](#-repeatable-inference-cli)
- [REST API Reference](#-rest-api-reference)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Automated Testing](#-automated-testing)
- [Performance Benchmarks](#-performance-benchmarks)
- [Next.js TypeScript Contract](#-nextjs-typescript-contract)

---

## 🔬 Overview & Architecture

Solar filaments are dense, cool plasma structures suspended along magnetic polarity inversion lines in the solar corona. When eruptive, they trigger Coronal Mass Ejections (CMEs) that pose severe threats to orbital satellites, GPS navigation, and terrestrial power grids.

This system provides a decoupled **two-tier architecture**:

```
[ Raw GONG Telescope Image: 2048x2048 ]
                   │
                   ▼
[ Preprocessing: Grayscale -> Resize (256x256) -> Float Normalize [0.0, 1.0] ]
                   │
                   ▼
  +───────────────────────────────────+
  │   STAGE 1: FilamentBasicCNN       │  ──> Latency: ~8.56 ms | Accuracy: 100%
  │   - Binary Presence Head          │  ──> Outputs: has_filament, probability
  │   - 4-Class Chirality Head        │  ──> Left, Right, Unidentifiable, Ambiguous
  +───────────────────────────────────+
                   │
         [ Does Filament Exist? ]
            /             \
        (No)              (Yes)
         │                  │
   [ Fast Return ]          ▼
                  +───────────────────────────────────+
                  │   STAGE 2: SolarFilamentUNet      │  ──> Latency: ~64.4 ms | Recall: 100%
                  │   - Encoder-Decoder + Skip Conns  │  ──> Optimized via Dice + BCE Loss
                  │   - Pixel Heatmap -> Threshold    │  ──> Binary Mask (256x256 / 2048x2048)
                  +───────────────────────────────────+
                            │
                            ▼
                  +───────────────────────────────────+
                  │   Geometric Extraction (OpenCV)   │  ──> COCO Polygons [[x0, y0, ...]]
                  │   - Douglas-Peucker Approximation │  ──> Bounding Boxes [x, y, w, h]
                  │   - Area & Coverage Percentage    │  ──> Total Filament Area (px)
                  │   - High Solar Activity Flag      │  ──> Triggered if Area > 1000 px
                  +───────────────────────────────────+
                            │
                            ▼
                  +───────────────────────────────────+
                  │   Visual Overlay Compositor       │  ──> Electric Cyan Mask
                  │   - Alpha-blended rendering       │  ──> Golden BBox + ID Badges
                  │   - Base64 JPEG Serialization     │  ──> Ready for <img src={...} />
                  +───────────────────────────────────+
```

---

## 🚀 Key Features

* **Hierarchical Two-Stage Inference:** High-speed CNN screening ($\sim 8.5\text{ ms}$) skips heavy pixel segmentation on empty solar disks, saving GPU/CPU compute.
* **4-Class Magnetic Chirality:** Classifies structural magnetic twist (*Left-bearing*, *Right-bearing*, *Unidentifiable*, *Ambiguous*).
* **Hybrid Dice + BCE Loss:** Overcomes severe foreground-background class imbalance where filaments occupy less than **$1\%$** of solar disk pixels.
* **Vector Geometry Extraction:** Emits lightweight, lossless COCO-compliant polygon coordinates, bounding boxes, and calibrated area metrics.
* **Live Telecast Streaming:** Sub-$40\text{ ms}$ telecast frame ingestion endpoint (`/api/v1/telecast/process-frame`) supporting camera feeds at $> 25\text{ FPS}$ with automated space weather alert triggers (`has_high_solar_activity`).
* **MLflow Tracking Integration:** Records hyperparameters, per-epoch metrics (Dice, IoU, Precision, Recall), model checkpoints, and PyTorch model registry flavors into SQLite.
* **Docker Ready:** Multi-stage production `Dockerfile` and `docker-compose.yml` for unified API and MLflow deployment.
* **Full Automated Test Coverage:** 17 comprehensive unit/integration tests verifying invalid image handling, API response schemas, and model math invariants.

---

## 📁 Project Directory Structure

```plaintext
Solar-Filament/
├── Dockerfile                          # Multi-stage production Docker container
├── docker-compose.yml                  # API (Port 8000) & MLflow UI (Port 5000)
├── requirements.txt                    # Frozen dependency specifications
├── METRICS_AND_BENCHMARKS.md           # In-depth benchmark results & analysis
├── mlflow.db                           # Local SQLite tracking database
│
├── scripts/
│   └── inference.py                    # Repeatable standalone CLI inference tool
│
├── backend/
│   ├── main.py                         # FastAPI application entry point
│   ├── train_pipeline.py               # CLI pipeline trainer with MLflow flag
│   ├── train_with_mlflow.py            # Dedicated MLflow experiment script
│   ├── benchmark.py                    # Latency, throughput, and memory benchmark
│   │
│   ├── app/
│   │   ├── schemas.py                  # Pydantic v2 typed data contracts
│   │   ├── routes/
│   │   │   ├── inference.py            # /detect, /segment, /analyze
│   │   │   ├── telecast.py             # /telecast/process-frame, /sample-frame
│   │   │   ├── dataset_routes.py       # /dataset/stats, /dataset/sample
│   │   │   └── train_routes.py         # /train/start, /train/status
│   │   └── services/
│   │       └── pipeline_service.py     # Singleton inference and visual compositor
│   │
│   ├── ml/
│   │   ├── cnn_model.py                # Stage 1: FilamentBasicCNN
│   │   ├── unet_model.py               # Stage 2: SolarFilamentUNet & polygon extractor
│   │   ├── losses.py                   # DiceLoss, DiceBCELoss & metric evaluators
│   │   ├── dataset.py                  # PyTorch Dataset for MAGFiLO GONG images
│   │   ├── trainer.py                  # Modular training loops & validation
│   │   └── mlflow_tracker.py           # MLflow experiment lifecycle manager
│   │
│   ├── models/
│   │   ├── basic_cnn.pth               # Pretrained Stage 1 CNN weights
│   │   ├── cnn_training_report.json    # CNN metrics history
│   │   ├── unet_segmenter.pth          # Pretrained Stage 2 U-Net weights
│   │   └── unet_training_report.json   # U-Net metrics history
│   │
│   └── tests/
│       ├── test_full_suite.py          # 17 comprehensive tests (invalid images, API, math)
│       ├── test_api.py                 # REST endpoint test suite
│       └── test_ml.py                  # PyTorch tensor & gradient unit tests
│
├── MAGFiLO_1.0_Kaggle_2026/             # Benchmark dataset (1,154 images, 8,199 anns)
├── Research Papers/                    # 5 peer-reviewed IEEE / Scopus reference papers
└── outputs/                            # Inferred masks, visual overlays & benchmark logs
```

---

## ⚙️ Installation & Quickstart

### 1. Prerequisites
* Python 3.10+ (tested on Python 3.11 and 3.13)
* Git

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Solar-Filament.git
cd Solar-Filament

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the FastAPI Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Swagger UI Documentation:** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/health`

---

## 🐳 Docker Containerisation

Run the API and MLflow tracking server simultaneously using Docker Compose:

```bash
# Build and start services
docker-compose up --build -d

# Check running containers
docker-compose ps
```

* **API Gateway:** `http://localhost:8000`
* **API Documentation:** `http://localhost:8000/docs`
* **MLflow Tracking Dashboard:** `http://localhost:5000`

---

## 🎯 Repeatable Inference CLI

The `scripts/inference.py` script enables reproducible evaluation on single images or entire observation folders:

```bash
# Inference on a single solar image
python scripts/inference.py \
  --image "MAGFiLO_1.0_Kaggle_2026/train/train_images/20110109104734Ch.jpeg" \
  --threshold 0.40 \
  --output-dir "outputs/inference"

# Batch inference on a directory of telescope observations
python scripts/inference.py \
  --image "path/to/telescope_folder/" \
  --device auto
```

### Outputs Generated in `outputs/inference/`:
* `*_mask.png`: Lossless binary segmentation mask.
* `*_overlay.jpg`: Alpha-blended visual overlay (electric cyan masks, golden bounding boxes, area tags).
* `*_metrics.json`: Quantitative metadata (filament count, area in px, coverage %, bounding box coordinates, and latency).

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | Returns service health, hardware accelerator (`cpu`/`cuda`), and model readiness flags. |
| `POST` | `/api/v1/detect` | Runs Stage 1 CNN screening for filament presence and 4-class magnetic chirality. |
| `POST` | `/api/v1/segment` | Runs Stage 2 U-Net segmentation, returning polygon vectors, bboxes, and metrics. |
| `POST` | `/api/v1/analyze` | End-to-end pipeline returning Stage 1 + Stage 2 data and an annotated Base64 image overlay. |
| `POST` | `/api/v1/telecast/process-frame` | Real-time frame buffer ingestion ($<40\text{ ms}$) for live telescope camera streams. |
| `GET` | `/api/v1/telecast/sample-frame` | Serves real dataset observation frames as Base64 for frontend streaming simulations. |
| `GET` | `/api/v1/dataset/stats` | Returns MAGFiLO 1.0 dataset breakdown, image counts, and chirality distributions. |
| `POST` | `/api/v1/train/start` | Initiates asynchronous background fine-tuning. |
| `GET` | `/api/v1/train/status` | Reports live training progression, current epoch, and validation loss. |

---

## 📊 MLflow Experiment Tracking

Track hyperparameters, loss trajectories, evaluation metrics, and PyTorch model artifacts:

```bash
# Train with full MLflow tracking
python backend/train_with_mlflow.py --epochs 3 --batch-size 8 --max-samples 60

# Or train via the main pipeline CLI
python backend/train_pipeline.py --stage both --epochs 3 --batch_size 8 --mlflow
```

To launch the MLflow dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Navigate to **`http://localhost:5000`** to view parameter logs, Dice/IoU charts, and registered model checkpoints.

---

## 🧪 Automated Testing

A full automated test suite is provided in [`backend/tests/test_full_suite.py`](file:///c:/Users/vaibh/Desktop/Solar-Filament/backend/tests/test_full_suite.py), covering:
1. **Invalid Images:** 0-byte empty files, corrupted byte payloads, plain text masquerading as images, negative/out-of-bounds thresholds, and malformed Base64 frames.
2. **API Contracts:** Schema validation for `/health`, `/api/v1/detect`, `/api/v1/segment`, `/api/v1/analyze`, and `/api/v1/telecast/process-frame`.
3. **Model Invariants:** Tensor shape checks, $[0.0, 1.0]$ probability bounds, softmax normalization, contour geometric validity, and backprop gradient stability.

Run all tests:
```bash
python -m unittest backend/tests/test_full_suite.py
```
```plaintext
Ran 17 tests in 0.658s
OK
```

---

## 📈 Performance Benchmarks

Inference latency, throughput, and memory consumption measured across batch sizes on the local test environment:

| Model | Batch Size | Batch Latency (ms) | Per-Image Latency (ms) | Throughput (FPS) | RAM Usage (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FilamentBasicCNN** | **1** | **8.56 ± 1.0** | **8.56 ms** | **116.9 FPS** | 253.1 MB |
| **FilamentBasicCNN** | 4 | 35.90 ± 1.2 | 8.98 ms | 111.4 FPS | 336.7 MB |
| **FilamentBasicCNN** | 16 | 171.36 ± 6.1 | 10.71 ms | 93.4 FPS | 479.5 MB |
| **SolarFilamentUNet** | **1** | **64.42 ± 4.8** | **64.42 ms** | **15.5 FPS** | 330.2 MB |
| **SolarFilamentUNet** | 4 | 299.72 ± 10.7 | 74.93 ms | 13.3 FPS | 538.8 MB |
| **SolarFilamentUNet** | 16 | 1267.77 ± 38.1 | 79.24 ms | 12.6 FPS | 1376.8 MB |

👉 *For full percentiles (P50, P95, P99), hardware details, and accuracy metrics, see [METRICS_AND_BENCHMARKS.md](file:///c:/Users/vaibh/Desktop/Solar-Filament/METRICS_AND_BENCHMARKS.md).*

---

## 💻 Next.js TypeScript Contract

Pre-configured TypeScript definitions for seamless frontend integration (`types/solar.ts`):

```typescript
export interface FilamentPolygon {
  id: string;
  segmentation: number[][]; // [[x0, y0, x1, y1, ...]]
  bbox: [number, number, number, number]; // [x, y, width, height]
  area: number;
  confidence: number;
  category_id: number;
  category_name: string;
}

export interface SegmentationResponse {
  success: boolean;
  num_filaments_found: number;
  total_filament_area: number;
  coverage_percentage: number;
  has_high_solar_activity: boolean;
  filaments: FilamentPolygon[];
  rle_mask?: { size: [number, number]; counts: string };
  binary_mask_base64?: string;
  inference_time_ms: number;
}
```
Directly render the annotated visual overlay in React:
```tsx
<img src={data.overlay_image_base64} alt="Solar Filament Detection Overlay" />
```
