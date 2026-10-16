// ─────────────────────────────────────────────────────────────────────────────
// All project data — REAL measurements only, no invented numbers
// ─────────────────────────────────────────────────────────────────────────────

export const PROJECT = {
  title: "Solar Filament ML Pipeline",
  subtitle: "FastAPI Backend & Deep Learning System",
  tagline: "SEE THE STRUCTURES HIDDEN IN THE SUN.",
  description:
    "A production-oriented deep learning pipeline for solar filament detection and segmentation, combining scientific image processing, model experimentation, experiment tracking, and API-based inference.",
  github: "https://github.com/",
  status: "In Development",
  version: "1.0.0",
};

export const TECH_STACK = [
  { name: "PyTorch", version: "2.14.1", category: "ML Framework", icon: "🔥", desc: "Deep learning backbone for CNN and U-Net models" },
  { name: "FastAPI", version: "0.139+", category: "API Framework", icon: "⚡", desc: "Async REST API for inference endpoints" },
  { name: "MLflow", version: "3.17.1", category: "Experiment Tracking", icon: "📊", desc: "Tracks runs, params, metrics and artifacts" },
  { name: "OpenCV", version: "4.x", category: "Image Processing", icon: "📷", desc: "Preprocessing and overlay generation" },
  { name: "Docker", version: "latest", category: "Containerisation", icon: "🐳", desc: "Reproducible deployment with docker-compose" },
  { name: "NumPy", version: "2.x", category: "Numerics", icon: "🔢", desc: "Array operations for tensor pipeline" },
  { name: "Pydantic", version: "v2", category: "Validation", icon: "✅", desc: "Schema validation for API contracts" },
  { name: "Uvicorn", version: "latest", category: "ASGI Server", icon: "🚀", desc: "High-performance async server" },
  { name: "pycocotools", version: "latest", category: "Annotation", icon: "🗂️", desc: "COCO JSON annotation parsing" },
  { name: "psutil", version: "latest", category: "Monitoring", icon: "📈", desc: "Memory and CPU profiling in benchmarks" },
];

export const DATASET = {
  name: "MAGFiLO 1.0",
  images: 1154,
  annotations: 8199,
  resolution: "2048×2048",
  source: "GONG H-alpha Solar Telescope",
  categories: [
    { label: "Left Chirality", pct: 43.2, color: "#FF9D42" },
    { label: "Right Chirality", pct: 37.9, color: "#FF5C35" },
    { label: "Unidentifiable", pct: 12.5, color: "#9585E8" },
    { label: "Ambiguous", pct: 6.4, color: "#A0A2AD" },
  ],
};

export const METRICS = {
  presenceAccuracy: 100,
  validationRecall: 100,
  validationDice: 0.0077,
  telecastLatency: 38.4,
  cnnLatency: 8.56,
  cnnFPS: 116.9,
  unetLatency: 64.42,
  unetFPS: 15.5,
};

export const BENCHMARKS = [
  { model: "FilamentBasicCNN", batch: 1,  latency: 8.56,    fps: 116.9, ram: 253.1 },
  { model: "FilamentBasicCNN", batch: 4,  latency: 35.90,   fps: 111.4, ram: 336.7 },
  { model: "FilamentBasicCNN", batch: 16, latency: 171.36,  fps: 93.4,  ram: 479.5 },
  { model: "SolarFilamentUNet", batch: 1,  latency: 64.42,  fps: 15.5,  ram: 330.2 },
  { model: "SolarFilamentUNet", batch: 16, latency: 1267.77, fps: 12.6, ram: 1376.8 },
];

export const API_ENDPOINTS = [
  {
    method: "GET",
    path: "/health",
    desc: "Service health and model load status",
    response: `{ "status": "healthy", "models_loaded": true }`,
  },
  {
    method: "POST",
    path: "/api/v1/detect",
    desc: "Stage 1 CNN — filament presence + chirality classification",
    body: `form-data: image (file)`,
    response: `{ "filament_present": true, "chirality": "left", "confidence": 0.91 }`,
  },
  {
    method: "POST",
    path: "/api/v1/segment",
    desc: "Stage 2 U-Net — pixel-level segmentation mask",
    body: `form-data: image (file), threshold? (float 0-1)`,
    response: `{ "mask_base64": "...", "coverage_percentage": 3.2, "polygons": [...] }`,
  },
  {
    method: "POST",
    path: "/api/v1/analyze",
    desc: "Full pipeline — detection + segmentation + annotated overlay",
    body: `form-data: image (file)`,
    response: `{ "detection": {...}, "segmentation": {...}, "overlay_base64": "..." }`,
  },
  {
    method: "POST",
    path: "/api/v1/telecast/process-frame",
    desc: "Live streaming frame — 38.4 ms avg latency",
    body: `form-data: frame (file)`,
    response: `{ "filament_present": true, "overlay_base64": "..." }`,
  },
  {
    method: "GET",
    path: "/api/v1/telecast/sample-frame",
    desc: "Retrieve a sample GONG frame for testing",
    response: `image/jpeg`,
  },
  {
    method: "GET",
    path: "/api/v1/dataset/stats",
    desc: "Dataset statistics — image counts and annotation breakdown",
    response: `{ "total_images": 1154, "total_annotations": 8199, ... }`,
  },
  {
    method: "POST",
    path: "/api/v1/train/start",
    desc: "Trigger a new training run with optional MLflow tracking",
    body: `{ "epochs": 10, "mlflow": true, "experiment_name": "..." }`,
    response: `{ "run_id": "...", "status": "started" }`,
  },
  {
    method: "GET",
    path: "/api/v1/train/status",
    desc: "Poll current training run metrics and progress",
    response: `{ "status": "running", "epoch": 3, "dice": 0.0077 }`,
  },
];

export const PIPELINE_STAGES = [
  {
    id: "01",
    title: "Data Ingestion",
    subtitle: "GONG H-alpha Observations",
    desc: "Raw 2048×2048 solar telescope images enter the pipeline from the MAGFiLO 1.0 dataset — 1,154 images with 8,199 expert annotations.",
    icon: "☀️",
    color: "#FF9D42",
  },
  {
    id: "02",
    title: "Preprocessing",
    subtitle: "Normalisation & Resizing",
    desc: "Images are decoded, resized to 256×256, converted to single-channel tensors, and normalised to [0,1]. Empty or corrupt frames are rejected early.",
    icon: "⚙️",
    color: "#FF7A2B",
  },
  {
    id: "03",
    title: "Stage 1 — CNN",
    subtitle: "Presence & Chirality",
    desc: "FilamentBasicCNN classifies whether a filament is present (sigmoid head) and its chirality — Left, Right, Unidentifiable, or Ambiguous (softmax head). Avg: 8.56 ms.",
    icon: "🧠",
    color: "#FF5C35",
  },
  {
    id: "04",
    title: "Stage 2 — U-Net",
    subtitle: "Pixel Segmentation",
    desc: "SolarFilamentUNet produces a binary pixel mask via an encoder-decoder with skip connections. DiceBCELoss handles the extreme class imbalance (<1% filament pixels).",
    icon: "🔬",
    color: "#C96BE0",
  },
  {
    id: "05",
    title: "Post-processing",
    subtitle: "Mask → Polygons",
    desc: "Binary mask is converted to COCO-compatible polygons and bounding boxes. Coverage percentage and solar activity flag are computed.",
    icon: "📐",
    color: "#9585E8",
  },
  {
    id: "06",
    title: "Tracking",
    subtitle: "MLflow Experiment Logging",
    desc: "Every run logs hyperparameters, per-epoch metrics (Dice, loss, recall) and artifacts to MLflow with a SQLite backend. Experiment: Solar-Filament-Segmentation.",
    icon: "📊",
    color: "#6B8AFF",
  },
  {
    id: "07",
    title: "API Response",
    subtitle: "FastAPI + Docker",
    desc: "Results — mask, overlay, polygons, metrics — are returned as JSON via FastAPI. The entire service is containerised with Docker and docker-compose.",
    icon: "🚀",
    color: "#4FC3F7",
  },
];

export const COLORS = {
  bg: "#08090D",
  surface: "#11131B",
  amber: "#FF9D42",
  orange: "#FF5C35",
  violet: "#9585E8",
  white: "#F3F2F0",
  muted: "#A0A2AD",
};
