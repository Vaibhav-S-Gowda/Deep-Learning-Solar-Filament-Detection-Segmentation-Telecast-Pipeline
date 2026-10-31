"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  Radio,
  Clock,
  Video,
  Play,
  Pause,
  Square,
  Circle,
  Download,
  Upload,
  Telescope,
  Activity,
  Sliders,
  Layers,
  RefreshCw,
} from "lucide-react";
import {
  normalizeBase,
  getTelecastStatus,
  getTelecastStations,
  getSequenceFrame,
  getLiveStreamUrl,
  type TelecastStatus,
  type StationInfo,
  type SequenceFrameResponse,
} from "@/lib/api";

type ConnectionState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "streaming"
  | "paused"
  | "reconnecting"
  | "error";

type StreamEngineMode = "mjpeg" | "timelapse" | "video";
type ViewLayerMode = "segmented" | "source";

interface RecordedVideo {
  url: string;
  blob: Blob;
  name: string;
  durationSec: number;
  recordedAt: string;
}

const DEFAULT_API = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://localhost:8000";
const STORAGE_KEY = "solarscope-api-endpoint";

function stateLabel(state: ConnectionState) {
  const labels: Record<ConnectionState, string> = {
    disconnected: "Disconnected",
    connecting: "Connecting",
    connected: "System ready",
    streaming: "Stream active",
    paused: "Paused",
    reconnecting: "Reconnecting",
    error: "Connection failure",
  };
  return labels[state];
}

export default function LiveMonitor() {
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [endpoint, setEndpoint] = useState(DEFAULT_API);
  const [engineMode, setEngineMode] = useState<StreamEngineMode>("mjpeg");
  const [selectedStation, setSelectedStation] = useState<string>("Ch");
  const [stations, setStations] = useState<StationInfo[]>([]);
  const [threshold, setThreshold] = useState<number>(0.4);
  const [streamFps, setStreamFps] = useState<number>(10);
  const [viewLayer, setViewLayer] = useState<ViewLayerMode>("segmented");

  // Stream state
  const [streamUrl, setStreamUrl] = useState<string>("");
  const [backendStatus, setBackendStatus] = useState<TelecastStatus | null>(null);
  const [lastError, setLastError] = useState<string>("");
  const [activeFrame, setActiveFrame] = useState<SequenceFrameResponse | null>(null);

  // Recording state
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordSeconds, setRecordSeconds] = useState<number>(0);
  const [recordedVideo, setRecordedVideo] = useState<RecordedVideo | null>(null);

  // Video file mode
  const [videoFileSrc, setVideoFileSrc] = useState<string>("/videos/cerro_tololo_sequence.mp4");

  // Telemetry metrics
  const [metrics, setMetrics] = useState({
    filaments: 0,
    area: 0,
    latency: 0,
    highActivity: false,
    timestamp: "Awaiting stream initialization",
  });

  // Refs
  const endpointRef = useRef(endpoint);
  const engineModeRef = useRef(engineMode);
  const stationRef = useRef(selectedStation);
  const thresholdRef = useRef(threshold);
  const isPlayingRef = useRef(false);
  const sequenceIndexRef = useRef(0);
  const seqTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Recorder refs
  const recordCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<NodeJS.Timeout | null>(null);
  const videoElemRef = useRef<HTMLVideoElement | null>(null);
  const imgStreamRef = useRef<HTMLImageElement | null>(null);

  endpointRef.current = endpoint;
  engineModeRef.current = engineMode;
  stationRef.current = selectedStation;
  thresholdRef.current = threshold;

  const getBase = useCallback(() => {
    const ep = normalizeBase(endpointRef.current);
    const def = normalizeBase(DEFAULT_API);
    return ep === def ? "" : ep;
  }, []);

  // Initialize: load stations list
  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const normalized = normalizeBase(saved);
      endpointRef.current = normalized;
      setEndpoint(normalized);
    }

    const base = getBase();
    getTelecastStations(base)
      .then((res) => {
        if (res.stations?.length) {
          setStations(res.stations);
        }
      })
      .catch(() => null);

    getTelecastStatus(base)
      .then((st) => {
        setBackendStatus(st);
        setConnectionState("connected");
      })
      .catch(() => null);

    return () => {
      isPlayingRef.current = false;
      if (seqTimerRef.current) clearTimeout(seqTimerRef.current);
      if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    };
  }, [getBase]);

  // Connect to backend
  const handleConnect = async () => {
    setConnectionState("connecting");
    setLastError("");
    try {
      const base = getBase();
      const status = await getTelecastStatus(base);
      if (status.status !== "ready") throw new Error("Telecast service is not ready");
      setBackendStatus(status);
      const st = await getTelecastStations(base);
      if (st.stations?.length) setStations(st.stations);
      setConnectionState("connected");
    } catch (err) {
      setLastError(err instanceof Error ? err.message : "Failed to establish connection with server");
      setConnectionState("error");
    }
  };

  // Launch stream
  const startStream = useCallback(() => {
    setLastError("");
    isPlayingRef.current = true;
    setConnectionState("streaming");

    if (engineMode === "mjpeg") {
      const base = getBase();
      const url = getLiveStreamUrl(
        stationRef.current,
        streamFps,
        thresholdRef.current,
        viewLayer === "segmented",
        true,
        base
      );
      setStreamUrl(url);
    } else if (engineMode === "timelapse") {
      const runSequence = async () => {
        if (!isPlayingRef.current) return;
        try {
          const base = getBase();
          const frame = await getSequenceFrame(
            stationRef.current,
            sequenceIndexRef.current,
            thresholdRef.current,
            viewLayer === "segmented",
            base
          );
          if (isPlayingRef.current) {
            setActiveFrame(frame);
            sequenceIndexRef.current = frame.next_index;
            setMetrics({
              filaments: frame.filaments_detected,
              area: frame.total_filament_area,
              latency: frame.processing_time_ms,
              highActivity: frame.has_high_solar_activity,
              timestamp: frame.observation_time,
            });
            seqTimerRef.current = setTimeout(runSequence, Math.max(100, 1000 / streamFps));
          }
        } catch {
          if (isPlayingRef.current) {
            seqTimerRef.current = setTimeout(runSequence, 1000);
          }
        }
      };
      void runSequence();
    } else if (engineMode === "video") {
      if (videoElemRef.current) {
        videoElemRef.current.play().catch(() => null);
      }
    }
  }, [engineMode, getBase, streamFps, viewLayer]);

  const pauseStream = useCallback(() => {
    isPlayingRef.current = false;
    setConnectionState("paused");
    if (seqTimerRef.current) clearTimeout(seqTimerRef.current);
    if (videoElemRef.current) videoElemRef.current.pause();
  }, []);

  const stopStream = useCallback(() => {
    isPlayingRef.current = false;
    setConnectionState("connected");
    setStreamUrl("");
    setActiveFrame(null);
    if (seqTimerRef.current) clearTimeout(seqTimerRef.current);
    if (videoElemRef.current) {
      videoElemRef.current.pause();
      videoElemRef.current.currentTime = 0;
    }
  }, []);

  // Handle engine mode or station change while running
  useEffect(() => {
    if (connectionState === "streaming") {
      if (seqTimerRef.current) clearTimeout(seqTimerRef.current);
      startStream();
    }
  }, [engineMode, selectedStation, streamFps, viewLayer, startStream, connectionState]);

  // Real-Time Video Recording Engine
  const startRecording = () => {
    const canvas = recordCanvasRef.current;
    if (!canvas) {
      setLastError("Observation render canvas is not initialized.");
      return;
    }

    try {
      const stream = canvas.captureStream(30);
      const mimeTypes = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
      let selectedMime = "";
      for (const mime of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mime)) {
          selectedMime = mime;
          break;
        }
      }

      const recorder = new MediaRecorder(stream, selectedMime ? { mimeType: selectedMime } : undefined);
      recordedChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const mime = recorder.mimeType || "video/webm";
        const blob = new Blob(recordedChunksRef.current, { type: mime });
        const url = URL.createObjectURL(blob);
        const now = new Date();
        const stamp = now.toISOString().replace(/[-:T.]/g, "").slice(0, 14);
        const filename = `solar_filament_acquisition_${stamp}.webm`;

        const recObj: RecordedVideo = {
          url,
          blob,
          name: filename,
          durationSec: recordSeconds,
          recordedAt: now.toLocaleTimeString(),
        };
        setRecordedVideo(recObj);

        // Download output file
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      };

      recorder.start(500);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setRecordSeconds(0);

      recordTimerRef.current = setInterval(() => {
        setRecordSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setLastError(err instanceof Error ? err.message : "Video recording initialization failed.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (recordTimerRef.current) {
      clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
  };

  // Canvas drawing loop for live recording HUD burn-in
  useEffect(() => {
    let animId: number;
    const canvas = recordCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const renderLoop = () => {
      ctx.fillStyle = "#10151b";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (engineMode === "mjpeg" && imgStreamRef.current && imgStreamRef.current.complete) {
        try {
          ctx.drawImage(imgStreamRef.current, 0, 0, canvas.width, canvas.height);
        } catch {
          // Handled gracefully during frame updates
        }
      } else if (engineMode === "timelapse" && activeFrame) {
        const imgObj = new Image();
        const src = viewLayer === "segmented" && activeFrame.overlay_base64
          ? activeFrame.overlay_base64
          : activeFrame.frame_base64;
        imgObj.src = src;
        if (imgObj.complete) {
          ctx.drawImage(imgObj, 0, 0, canvas.width, canvas.height);
        }
      } else if (engineMode === "video" && videoElemRef.current) {
        const vid = videoElemRef.current;
        if (!vid.paused && !vid.ended) {
          ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
        }
      }

      if (isRecording) {
        ctx.fillStyle = "rgba(16, 21, 27, 0.85)";
        ctx.fillRect(12, 12, 175, 30);
        ctx.strokeStyle = "#c9302c";
        ctx.lineWidth = 1;
        ctx.strokeRect(12, 12, 175, 30);

        const blink = Math.floor(Date.now() / 500) % 2 === 0;
        ctx.fillStyle = blink ? "#c9302c" : "#5a1e1b";
        ctx.beginPath();
        ctx.arc(26, 27, 5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.font = "600 11px monospace";
        const mins = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
        const secs = String(recordSeconds % 60).padStart(2, "0");
        ctx.fillText(`REC ${mins}:${secs}`, 39, 31);
      }

      animId = requestAnimationFrame(renderLoop);
    };

    animId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animId);
  }, [engineMode, activeFrame, viewLayer, isRecording, recordSeconds]);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoFileSrc(url);
      setEngineMode("video");
      if (connectionState === "streaming") {
        startStream();
      }
    }
  };

  const isStreaming = connectionState === "streaming";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <Link href="/" style={{ display: "flex", alignItems: "center", gap: 12, textDecoration: "none" }}>
            <span className="brand-mark">
              <Telescope size={20} />
            </span>
            <div>
              <div className="brand-name">SOLAR OBSERVATORY<span> / TELECAST MONITOR</span></div>
              <div className="brand-caption">GONG H-ALPHA CHROMOSPHERE OBSERVATION NETWORK</div>
            </div>
          </Link>
        </div>

        <nav className="topbar-right" aria-label="Observatory navigation">
          <Link href="/" className="text-button active">Live Monitor</Link>
          <Link href="/dataset" className="text-button">Dataset</Link>
          <Link href="/analyze" className="text-button">Analysis</Link>
          <Link href="/training" className="text-button">Training</Link>
        </nav>
      </header>

      <section className="page-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-rule" /> CONTINUOUS CHROMOSPHERIC TELECAST</div>
          <h1>Observation <span>Telecast & Tracking</span></h1>
          <p className="heading-copy">
            Continuous real-time H-alpha observation feed with automated U-Net solar filament segmentation and telemetry capture.
          </p>
        </div>
        <div className="observation-note">
          <Activity size={15} style={{ color: "var(--accent)" }} />
          <span>Observatory station: <strong>{selectedStation.toUpperCase()}</strong><br />Calibration active</span>
        </div>
      </section>

      <div className="workspace">
        {/* Main Observation Viewer */}
        <section className="viewer-column" aria-label="Solar observation display">
          <div className="viewer-card">
            <div className="viewer-toolbar">
              <div className="viewer-title-group">
                <span className="live-square" style={{ background: isStreaming ? "var(--success)" : "var(--text-muted)" }} />
                <div>
                  <div className="viewer-title">
                    {engineMode === "mjpeg" && "CONTINUOUS TELECAST STREAM (MJPEG)"}
                    {engineMode === "timelapse" && "CHRONOLOGICAL OBSERVATION SEQUENCE"}
                    {engineMode === "video" && "SOLAR OBSERVATION VIDEO FEED"}
                  </div>
                  <div className="viewer-subtitle">
                    {stations.find((s) => s.code === selectedStation)?.name || "GONG NETWORK"}
                    <span className="separator">/</span> 512 x 512 resolution
                  </div>
                </div>
              </div>
              <div className="frame-meta">
                <span className="meta-label">STATUS</span>
                <span>{isStreaming ? (isRecording ? "RECORDING ACTIVE" : "STREAM ACTIVE") : "STANDBY"}</span>
              </div>
            </div>

            {/* Video Stage */}
            <div className="image-stage">
              <div className="stage-corner stage-tl" />
              <div className="stage-corner stage-tr" />
              <div className="stage-corner stage-bl" />
              <div className="stage-corner stage-br" />
              <div className="stage-grid" aria-hidden="true" />

              {/* Mode 1: Live MJPEG Stream */}
              {engineMode === "mjpeg" && isStreaming && (
                <img
                  ref={imgStreamRef}
                  className="solar-image"
                  src={streamUrl}
                  alt="Solar observation telecast with filament segmentation overlay"
                  crossOrigin="anonymous"
                />
              )}

              {/* Mode 2: Chronological Sequence */}
              {engineMode === "timelapse" && isStreaming && activeFrame && (
                <img
                  className="solar-image"
                  src={
                    viewLayer === "segmented" && activeFrame.overlay_base64
                      ? activeFrame.overlay_base64
                      : activeFrame.frame_base64
                  }
                  alt="Solar observation frame"
                />
              )}

              {/* Mode 3: Continuous Solar Video File */}
              {engineMode === "video" && (
                <video
                  ref={videoElemRef}
                  src={videoFileSrc}
                  className="solar-image"
                  loop
                  muted
                  playsInline
                  autoPlay={isStreaming}
                />
              )}

              {/* Standby Empty State */}
              {!isStreaming && (
                <div className="empty-state">
                  <div className="empty-sun"><span /></div>
                  <div className="empty-title">OBSERVATION FEED INACTIVE</div>
                  <p>Initiate stream acquisition to connect to the observation pipeline.</p>
                </div>
              )}

              {/* Recording Canvas */}
              <canvas
                ref={recordCanvasRef}
                width={512}
                height={512}
                style={{ display: "none" }}
              />

              <div className="stage-coordinate coord-top">N [H-ALPHA 656.28 nm]</div>
              <div className="stage-coordinate coord-bottom">
                {activeFrame?.file_name || `${selectedStation} OBSERVATORY CALIBRATED`}
              </div>
              {isStreaming && (
                <div className="filament-count-badge">
                  <span className="glow-dot" />
                  <span>{metrics.filaments} {metrics.filaments === 1 ? "FILAMENT DETECTED" : "FILAMENTS DETECTED"}</span>
                </div>
              )}
              {metrics.highActivity && <div className="activity-flag" style={{ top: 52 }}>ELEVATED FILAMENT ACTIVITY</div>}
            </div>

            {/* Viewer Footer */}
            <div className="viewer-footer">
              <div className="view-switch" role="group" aria-label="Layer selection">
                <button
                  className={viewLayer === "segmented" ? "view-option selected" : "view-option"}
                  onClick={() => setViewLayer("segmented")}
                  type="button"
                >
                  Filament Overlay
                </button>
                <button
                  className={viewLayer === "source" ? "view-option selected" : "view-option"}
                  onClick={() => setViewLayer("source")}
                  type="button"
                >
                  Raw Solar Disk
                </button>
              </div>

              {/* In-Browser Recording Button */}
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                {isRecording ? (
                  <button
                    className="button button-record button-record-active"
                    type="button"
                    onClick={stopRecording}
                  >
                    <Square size={13} fill="currentColor" /> Stop Recording ({recordSeconds}s)
                  </button>
                ) : (
                  <button
                    className="button button-record"
                    type="button"
                    onClick={startRecording}
                    disabled={!isStreaming}
                  >
                    <Circle size={11} fill="currentColor" /> Record Stream
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Telemetry Metrics Row */}
          <div className="metrics-row" aria-live="polite">
            <div className="metric-card">
              <span className="metric-label">FILAMENTS IDENTIFIED</span>
              <strong>{isStreaming ? (metrics.filaments || "0") : "—"}</strong>
              <span className="metric-foot">discrete structures</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">SEGMENTED PIXEL AREA</span>
              <strong>{isStreaming ? `${metrics.area.toLocaleString()}` : "—"}<small> px</small></strong>
              <span className="metric-foot">chromospheric footprint</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">TRANSMISSION RATE</span>
              <strong>{isStreaming ? `${streamFps}` : "—"}<small> FPS</small></strong>
              <span className="metric-foot">pipeline target</span>
            </div>
            <div className="metric-card metric-state">
              <span className="metric-label">CHROMOSPHERIC STATE</span>
              <strong className={metrics.highActivity ? "activity-value" : "quiet-value"}>
                {isStreaming ? (metrics.highActivity ? "Elevated" : "Nominal") : "Standby"}
              </strong>
              <span className="metric-foot">{isStreaming ? "continuous monitor" : "awaiting stream"}</span>
            </div>
          </div>

          {/* Recorded Video Playback Section */}
          {recordedVideo && (
            <div className="recorded-video-card">
              <div className="viewer-title" style={{ marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
                <span>RECORDED OBSERVATION PLAYBACK</span>
                <span style={{ color: "var(--accent)", fontFamily: "var(--font-plex-mono), monospace" }}>
                  Duration: {recordedVideo.durationSec}s | Captured: {recordedVideo.recordedAt}
                </span>
              </div>
              <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
                <video
                  src={recordedVideo.url}
                  controls
                  style={{ width: 280, height: 280, borderRadius: 6, background: "#10151b" }}
                />
                <div style={{ flex: 1, minWidth: 220 }}>
                  <div style={{ color: "var(--text)", fontWeight: 600, marginBottom: 6, fontSize: 13 }}>
                    {recordedVideo.name}
                  </div>
                  <p style={{ color: "var(--text-secondary)", fontSize: 12, margin: "0 0 14px 0", lineHeight: 1.6 }}>
                    Recorded sequence with integrated U-Net segmentation overlays, observatory timestamp, and active filament region bounding coordinates.
                  </p>
                  <a
                    href={recordedVideo.url}
                    download={recordedVideo.name}
                    className="button button-primary"
                    style={{ display: "inline-flex", alignItems: "center", gap: 7, textDecoration: "none", padding: "0 16px" }}
                  >
                    <Download size={14} /> Download Capture File ({Math.round(recordedVideo.blob.size / 1024)} KB)
                  </a>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Right Sidebar Controls */}
        <aside className="control-column" aria-label="Stream parameters">
          {/* Stream Engine Selector */}
          <section className="control-card">
            <div className="card-heading">
              <div><span className="section-index">01</span><h2>Transmission Mode</h2></div>
            </div>

            <div style={{ padding: "16px 18px" }}>
              <div className="stream-mode-tabs" role="tablist">
                <button
                  className={`stream-tab ${engineMode === "mjpeg" ? "active" : ""}`}
                  onClick={() => setEngineMode("mjpeg")}
                  type="button"
                >
                  <Radio size={13} /> Live MJPEG
                </button>
                <button
                  className={`stream-tab ${engineMode === "timelapse" ? "active" : ""}`}
                  onClick={() => setEngineMode("timelapse")}
                  type="button"
                >
                  <Clock size={13} /> Sequence
                </button>
                <button
                  className={`stream-tab ${engineMode === "video" ? "active" : ""}`}
                  onClick={() => setEngineMode("video")}
                  type="button"
                >
                  <Video size={13} /> Video File
                </button>
              </div>

              {/* Station Selector */}
              {engineMode !== "video" && (
                <div>
                  <label className="meta-label" style={{ display: "block", marginBottom: 6 }}>
                    OBSERVATORY STATION
                  </label>
                  <select
                    className="station-select"
                    value={selectedStation}
                    onChange={(e) => setSelectedStation(e.target.value)}
                  >
                    <option value="Ch">Cerro Tololo Inter-American Observatory (CTIO, Chile)</option>
                    <option value="Bh">Big Bear Solar Observatory (BBSO, California)</option>
                    <option value="Mh">Mauna Loa Solar Observatory (MLSO, Hawaii)</option>
                    <option value="Lh">Learmonth Solar Observatory (LEAR, Australia)</option>
                    <option value="Th">Teide Observatory (OT, Spain)</option>
                    <option value="Uh">Udaipur Solar Observatory (USO, India)</option>
                    <option value="all">Full GONG Global Network Sequence</option>
                  </select>
                </div>
              )}

              {/* Video File Selector */}
              {engineMode === "video" && (
                <div>
                  <label className="meta-label" style={{ display: "block", marginBottom: 6 }}>
                    TIMELAPSE SEQUENCE
                  </label>
                  <select
                    className="station-select"
                    value={videoFileSrc}
                    onChange={(e) => setVideoFileSrc(e.target.value)}
                  >
                    <option value="/videos/cerro_tololo_sequence.mp4">Cerro Tololo Observation Sequence (125 frames)</option>
                    <option value="/videos/big_bear_sequence.mp4">Big Bear Observation Sequence (118 frames)</option>
                  </select>
                  <div style={{ marginTop: 10 }}>
                    <label className="text-button" style={{ display: "inline-flex", alignItems: "center", gap: 7, cursor: "pointer", padding: "6px 0" }}>
                      <Upload size={13} /> Load Local Video File (MP4, WebM)
                      <input
                        type="file"
                        accept="video/mp4,video/webm,video/avi"
                        style={{ display: "none" }}
                        onChange={handleVideoUpload}
                      />
                    </label>
                  </div>
                </div>
              )}

              {/* Stream Action Buttons */}
              <div className="connection-actions" style={{ marginTop: 16 }}>
                {lastError && (
                  <div className="error-message" role="alert" style={{ marginBottom: 4 }}>
                    {lastError}
                  </div>
                )}
                {isStreaming ? (
                  <button className="button button-primary" type="button" onClick={pauseStream}>
                    <Pause size={14} style={{ marginRight: 6 }} /> Pause Telecast
                  </button>
                ) : (
                  <button className="button button-primary" type="button" onClick={startStream}>
                    <Play size={14} style={{ marginRight: 6 }} fill="currentColor" /> Start Live Stream
                  </button>
                )}
                <button
                  className="button button-stop"
                  type="button"
                  onClick={stopStream}
                  disabled={!isStreaming && connectionState !== "paused"}
                >
                  <Square size={13} style={{ marginRight: 6 }} /> Stop Telecast
                </button>
              </div>
            </div>
          </section>

          {/* Framerate & Threshold Tuning */}
          <section className="control-card tuning-card">
            <div className="card-heading">
              <div><span className="section-index">02</span><h2>Acquisition Tuning</h2></div>
              <span className="threshold-value">{streamFps} FPS</span>
            </div>

            <div style={{ padding: "16px 18px" }}>
              <label className="meta-label" style={{ display: "block", marginBottom: 6 }}>
                FRAME RATE
              </label>
              <input
                className="threshold-range"
                type="range"
                min="5"
                max="25"
                step="1"
                value={streamFps}
                onChange={(e) => setStreamFps(Number(e.target.value))}
                style={{ "--range-progress": `${((streamFps - 5) / 20) * 100}%` } as React.CSSProperties}
              />
              <div className="range-scale"><span>5 FPS (low compute)</span><span>25 FPS (standard)</span></div>

              <div style={{ marginTop: 18 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <span className="meta-label">DETECTION THRESHOLD</span>
                  <span className="threshold-value">{threshold.toFixed(2)}</span>
                </div>
                <input
                  className="threshold-range"
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  style={{ "--range-progress": `${((threshold - 0.1) / 0.8) * 100}%` } as React.CSSProperties}
                />
                <div className="range-scale"><span>High sensitivity (0.10)</span><span>High confidence (0.90)</span></div>
              </div>
            </div>
          </section>
        </aside>
      </div>

      <footer className="page-footer">
        <span>Solar Observatory Telecast & Analysis System | National Solar Observatory GONG Program</span>
        <span>H-alpha 656.28 nm observations processed with deep convolutional neural networks.</span>
      </footer>
    </main>
  );
}
