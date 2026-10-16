# Solar Filament ML Pipeline & FastAPI Backend

A production-ready deep learning pipeline and REST API built with **PyTorch**, **FastAPI**, **MLflow**, and **Docker** for detecting and segmenting solar filaments from GONG H-Alpha solar telescope observations and live telecasts.

---

## 🚀 Key Features

- **Stage 1 (Basic CNN Classifier)**: Real-time screening for solar filament presence (~8.56 ms) and 4-class orientation/chirality categorization (`Left`, `Right`, `Unidentifiable`, `Ambiguous`).
- **Stage 2 (Solar Filament U-Net Segmenter)**: Fine-grained pixel segmentation predicting lossless binary masks, COCO-style polygons (`[x0, y0, x1, y1, ...]`), and bounding boxes.
- **Combined Dice + BCE Loss**: Specially engineered to overcome severe class imbalance (filaments cover < 1% of the solar disk).
- **MLflow Experiment Tracking**: Logs hyperparameters, per-epoch loss, Dice, IoU, recall, and model artifacts to SQLite (`sqlite:///mlflow.db`).
- **Repeatable Inference CLI**: Standalone inference tool (`scripts/inference.py`) supporting single images and directories.
- **Docker Containerisation**: Production `Dockerfile` and `docker-compose.yml` packaging the API (Port 8000) and MLflow UI (Port 5000).
- **Next.js Compatibility**:
  - Full CORS middleware preconfigured for `http://localhost:3000` and `http://localhost:3001`.
  - JSON schemas matching TypeScript interfaces with `coverage_percentage`, `has_high_solar_activity`, and `binary_mask_base64`.
  - Direct Base64 visual overlay responses for immediate rendering in `<img src={result.overlay_image_base64} />`.
  - Live Telecast endpoints for continuous frame streaming (< 40 ms latency) and simulated telescope feeds.

---

## 🛠️ Quick Start

### 1. Run the FastAPI Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation will be available at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### 2. Run the Automated Test Suite

```bash
python -m unittest backend/tests/test_full_suite.py
```

### 3. Run Performance Benchmarks

```bash
python backend/benchmark.py --batch-sizes 1 2 4 8 16 --iterations 15
```

### 4. Train or Fine-Tune with MLflow

```bash
python backend/train_with_mlflow.py --epochs 3 --batch-size 8
```

---

## 📡 REST API Reference

### 1. Full Pipeline Analysis (`POST /api/v1/analyze`)
Runs Stage 1 CNN and Stage 2 U-Net, returning detection stats, polygon segmentations, and an annotated image overlay.

- **Request**: `multipart/form-data` with `file: <binary image>` and optional `threshold: 0.4`.

### 2. U-Net Segmentation (`POST /api/v1/segment`)
Extracts fine-grained pixel masks, COCO polygons, area, and bounding boxes.

### 3. Live Telecast Frame Processing (`POST /api/v1/telecast/process-frame`)
Designed for live telescope camera feeds. Accepts base64 frames and returns annotated overlays with latency tracking (<40 ms).

### 4. Simulated Telecast Sample (`GET /api/v1/telecast/sample-frame`)
Fetches a sample observation frame from the dataset as base64 to simulate live telescope streams.

### 5. Dataset Statistics (`GET /api/v1/dataset/stats`)
Returns dataset breakdown, category distribution, and telescope details.

---

👉 *For detailed empirical metrics, latency percentiles, and hardware memory profiles, see [METRICS_AND_BENCHMARKS.md](../METRICS_AND_BENCHMARKS.md).*
