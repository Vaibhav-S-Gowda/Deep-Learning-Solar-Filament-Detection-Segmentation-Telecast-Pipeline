# SolarScope Live — frontend only

A small Next.js dashboard for the FastAPI project in `Solar-Filament.zip`. This folder contains no FastAPI code, model weights, dataset, or backend services.

## Run it

Requirements: Node.js 20.9+ and pnpm 11. From this folder:

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`. The initial FastAPI URL is `http://localhost:8000`; change it in the **FastAPI base URL** field and select **Apply URL**. The selected value is saved in this browser. Alternatively, set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` before starting Next.js. To create `.env.local` from the example, use `cp .env.example .env.local` on macOS/Linux or `Copy-Item .env.example .env.local` in PowerShell. The in-app setting takes precedence after it is saved.

The FastAPI server must allow browser requests from the Next.js origin. The supplied `backend/main.py` configures CORS. The frontend calls the backend directly; it does not proxy or modify the backend.

## API contract used

- `GET /api/v1/telecast/status` — check the telecast service and model readiness.
- `GET /api/v1/telecast/sample-frame` — request a GONG H-alpha sample frame.
- `POST /api/v1/telecast/process-frame` — submit that frame as `frame_base64`, with `timestamp`, `return_overlay: true`, and the selected `detection_threshold`; display `overlay_frame_base64` and the returned segmentation metrics.

The viewer refreshes and processes frames sequentially while the stream is active. It avoids overlapping requests, supports pause/resume/disconnect, and automatically retries after request failures with increasing delays.

## Important note about “live” GONG data

The supplied backend does **not** expose SSE/WebSocket streaming or a direct live GONG ingest route. Its `sample-frame` endpoint chooses a random JPEG from the backend's local dataset folder (and is documented there as a way to simulate a live telecast); `process-frame` segments one frame per HTTP request. Therefore this frontend's continuously refreshed display is the demo/polling stream provided by those existing endpoints, not a direct real-time telescope feed. The bundled FastAPI code also uses a machine-specific Windows dataset path for sample images, so that directory must exist on the backend machine or the sample-frame call will return an error.

To show actual live GONG observations, the existing backend must expose its live image source through a frame endpoint or streaming transport. The frontend intentionally does not invent or replace that backend behavior.
