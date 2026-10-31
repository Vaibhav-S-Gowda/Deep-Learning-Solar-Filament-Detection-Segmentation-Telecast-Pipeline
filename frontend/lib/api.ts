/**
 * api.ts — Centralized, type-safe API client for the SolarScope FastAPI backend.
 *
 * All fetch calls go through Next.js rewrites (/api/* → FastAPI /api/*) so
 * no CORS headers are needed and the backend URL never appears in the browser
 * bundle. The NEXT_PUBLIC_API_BASE_URL env var is the fallback for when the
 * user overrides the endpoint in the connection panel.
 */

// ─── Base helpers ─────────────────────────────────────────────────────────────

/** Strip trailing slashes. */
export function normalizeBase(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

/**
 * Core fetch wrapper. Returns typed JSON or throws a descriptive Error.
 * Uses the Next.js rewrite proxy when `base` is empty / the default.
 */
export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
  base = ""
): Promise<T> {
  const root = base ? normalizeBase(base) : "";
  const url = root ? `${root}${path}` : path; // relative when using proxy

  const response = await fetch(url, { cache: "no-store", ...init });
  const raw = await response.text();

  let body: unknown = null;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    throw new Error(
      `Expected JSON from the backend (HTTP ${response.status}).`
    );
  }

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : `Backend request failed (HTTP ${response.status}).`;
    throw new Error(detail);
  }

  if (!body || typeof body !== "object") {
    throw new Error("The backend returned an empty or malformed response.");
  }

  return body as T;
}

// ─── Response types (mirrors backend/app/schemas.py) ─────────────────────────

export interface HealthStatus {
  status: string;
  device: string;
  stage1_cnn_ready: boolean;
  stage2_unet_ready: boolean;
}

export interface TelecastStatus {
  status: string;
  device: string;
  stage1_cnn_ready: boolean;
  stage2_unet_ready: boolean;
  recommended_fps: number;
}

export interface StationInfo {
  code: string;
  name: string;
  location: string;
  station_id: string;
  total_frames: number;
  start_date: string | null;
  end_date: string | null;
}

export interface StationsResponse {
  stations: StationInfo[];
  total_dataset_frames: number;
}

export interface SequenceFrameResponse {
  success: boolean;
  index: number;
  next_index: number;
  total_frames: number;
  file_name: string;
  station_code: string;
  station_name: string;
  station_location: string;
  observation_time: string;
  width: number;
  height: number;
  frame_base64: string;
  overlay_base64: string | null;
  filaments_detected: number;
  total_filament_area: number;
  has_high_solar_activity: boolean;
  processing_time_ms: number;
}

export interface SampleFrame {
  success: boolean;
  file_name: string;
  observation_time?: string;
  width: number;
  height: number;
  frame_base64: string;
  telescope_station: string;
}

export interface ProcessedFrame {
  success: boolean;
  timestamp: string;
  filaments_detected: number;
  total_filament_area: number;
  has_high_solar_activity: boolean;
  overlay_frame_base64: string | null;
  processing_time_ms: number;
}

export interface CategoryScore {
  category_id: number;
  name: string;
  probability: number;
}

export interface CNNDetectionResponse {
  success: boolean;
  is_solar_observation?: boolean;
  validation_message?: string | null;
  has_filament: boolean;
  filament_probability: number;
  dominant_category: string;
  category_scores: CategoryScore[];
  inference_time_ms: number;
}

export interface FilamentPolygon {
  id: string;
  segmentation: number[][];
  bbox: number[];
  area: number;
  confidence: number;
  category_id: number;
  category_name: string;
}

export interface SegmentationResponse {
  success: boolean;
  is_solar_observation?: boolean;
  num_filaments_found: number;
  total_filament_area: number;
  coverage_percentage: number;
  has_high_solar_activity: boolean;
  filaments: FilamentPolygon[];
  rle_mask: Record<string, unknown> | null;
  binary_mask_base64: string | null;
  inference_time_ms: number;
}

export interface FullAnalysisResponse {
  success: boolean;
  is_solar_observation?: boolean;
  validation_message?: string | null;
  image_metadata: Record<string, unknown>;
  stage1_cnn: CNNDetectionResponse;
  stage2_segmentation: SegmentationResponse;
  overlay_image_base64: string | null;
  total_processing_time_ms: number;
}

export interface DatasetStats {
  dataset_name: string;
  total_train_images: number;
  total_test_images: number;
  total_annotations: number;
  category_counts: Record<string, number>;
  image_resolution: string;
  instrument: string;
}

export interface DatasetSample {
  success: boolean;
  image_id: number;
  file_name: string;
  date_captured: string | null;
  num_filaments_annotated: number;
  overlay_base64: string;
  annotations: {
    id: number;
    category_id: number;
    bbox: number[];
    area: number;
  }[];
}

export interface TrainRequest {
  stage: "cnn" | "unet" | "both";
  epochs: number;
  batch_size: number;
  max_samples: number;
}

export interface TrainStatus {
  is_training: boolean;
  current_stage: string | null;
  progress_percentage: number;
  status_message: string;
  latest_metrics: Record<string, unknown> | null;
}

export interface TrainingHistory {
  stage1_cnn: Record<string, unknown> | null;
  stage2_unet: Record<string, unknown> | null;
}

// ─── API functions ────────────────────────────────────────────────────────────

/** Health check — `/health` */
export const getHealth = (base = "") =>
  apiFetch<HealthStatus>("/health", {}, base);

/** Telecast status — `/api/v1/telecast/status` */
export const getTelecastStatus = (base = "") =>
  apiFetch<TelecastStatus>("/api/v1/telecast/status", {}, base);

/** List available GONG stations — `/api/v1/telecast/stations` */
export const getTelecastStations = (base = "") =>
  apiFetch<StationsResponse>("/api/v1/telecast/stations", {}, base);

/** Get a sequence frame chronologically — `/api/v1/telecast/sequence-frame` */
export const getSequenceFrame = (
  station = "Ch",
  index = 0,
  threshold = 0.4,
  return_overlay = true,
  base = ""
) =>
  apiFetch<SequenceFrameResponse>(
    `/api/v1/telecast/sequence-frame?station=${encodeURIComponent(
      station
    )}&index=${index}&threshold=${threshold}&return_overlay=${return_overlay}`,
    {},
    base
  );

/**
 * Get direct URL for live MJPEG video stream — `/api/v1/telecast/stream`
 * Suitable for native `<img src="..." />` continuous streaming.
 */
export function getLiveStreamUrl(
  station = "Ch",
  fps = 10,
  threshold = 0.4,
  overlay = true,
  hud = true,
  base = ""
): string {
  const root = base ? normalizeBase(base) : "";
  const params = new URLSearchParams({
    station,
    fps: String(fps),
    threshold: String(threshold),
    overlay: String(overlay),
    hud: String(hud),
    t: String(Date.now()), // Cache bust on new connection
  });
  return `${root}/api/v1/telecast/stream?${params.toString()}`;
}

/** Get a random sample frame from the dataset — `/api/v1/telecast/sample-frame` */
export const getSampleFrame = (station = "", base = "") => {
  const query = station ? `?station=${encodeURIComponent(station)}` : "";
  return apiFetch<SampleFrame>(`/api/v1/telecast/sample-frame${query}`, {}, base);
};

/** Process a telecast frame through the ML pipeline — `POST /api/v1/telecast/process-frame` */
export const processFrame = (
  frame_base64: string,
  detection_threshold = 0.4,
  return_overlay = true,
  base = ""
) =>
  apiFetch<ProcessedFrame>(
    "/api/v1/telecast/process-frame",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_base64,
        timestamp: new Date().toISOString(),
        return_overlay,
        detection_threshold,
      }),
    },
    base
  );

/** Stage 1 CNN detection — `POST /api/v1/detect` (multipart/form-data file upload) */
export const detectFilament = (file: File, base = "") => {
  const form = new FormData();
  form.append("file", file);
  return apiFetch<CNNDetectionResponse>(
    "/api/v1/detect",
    { method: "POST", body: form },
    base
  );
};

/** Stage 2 U-Net segmentation — `POST /api/v1/segment` */
export const segmentFilament = (
  file: File,
  threshold = 0.4,
  base = ""
) => {
  const form = new FormData();
  form.append("file", file);
  form.append("threshold", String(threshold));
  return apiFetch<SegmentationResponse>(
    "/api/v1/segment",
    { method: "POST", body: form },
    base
  );
};

/** Full pipeline analysis — `POST /api/v1/analyze` */
export const analyzeImage = (
  file: File,
  threshold = 0.4,
  base = ""
) => {
  const form = new FormData();
  form.append("file", file);
  form.append("threshold", String(threshold));
  return apiFetch<FullAnalysisResponse>(
    "/api/v1/analyze",
    { method: "POST", body: form },
    base
  );
};

/** Dataset statistics — `GET /api/v1/dataset/stats` */
export const getDatasetStats = (base = "") =>
  apiFetch<DatasetStats>("/api/v1/dataset/stats", {}, base);

/** Ground-truth annotated sample — `GET /api/v1/dataset/sample` */
export const getDatasetSample = (base = "") =>
  apiFetch<DatasetSample>("/api/v1/dataset/sample", {}, base);

/** Start a training run — `POST /api/v1/train/start` */
export const startTraining = (req: TrainRequest, base = "") =>
  apiFetch<TrainStatus>(
    "/api/v1/train/start",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    },
    base
  );

/** Polling training status — `GET /api/v1/train/status` */
export const getTrainingStatus = (base = "") =>
  apiFetch<TrainStatus>("/api/v1/train/status", {}, base);

/** Historical training curves — `GET /api/v1/train/history` */
export const getTrainingHistory = (base = "") =>
  apiFetch<TrainingHistory>("/api/v1/train/history", {}, base);
