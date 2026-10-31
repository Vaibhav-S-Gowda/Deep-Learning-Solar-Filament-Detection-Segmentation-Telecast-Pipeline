"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  Telescope,
  Cpu,
  Layers,
  Play,
  CheckCircle2,
  AlertCircle,
  BarChart3,
  RefreshCw,
  Sliders,
  Activity,
} from "lucide-react";
import {
  getTrainingStatus,
  getTrainingHistory,
  startTraining,
  type TrainStatus,
  type TrainingHistory,
  type TrainRequest,
} from "@/lib/api";

type Stage = "cnn" | "unet" | "both";

export default function TrainingPage() {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [stage, setStage] = useState<Stage>("both");
  const [epochs, setEpochs] = useState<number>(3);
  const [batchSize, setBatchSize] = useState<number>(8);
  const [maxSamples, setMaxSamples] = useState<number>(150);
  const [starting, setStarting] = useState<boolean>(false);
  const [startError, setStartError] = useState<string>("");
  const pollerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    void getTrainingStatus().then(setStatus).catch(() => null);
    void getTrainingHistory().then(setHistory).catch(() => null);
  }, []);

  useEffect(() => {
    if (status?.is_training) {
      pollerRef.current = setInterval(() => {
        void getTrainingStatus()
          .then((s) => {
            setStatus(s);
            if (!s.is_training) {
              if (pollerRef.current) clearInterval(pollerRef.current);
              void getTrainingHistory().then(setHistory).catch(() => null);
            }
          })
          .catch(() => null);
      }, 2000);
    }
    return () => {
      if (pollerRef.current) clearInterval(pollerRef.current);
    };
  }, [status?.is_training]);

  const handleStart = async () => {
    setStarting(true);
    setStartError("");
    try {
      const req: TrainRequest = {
        stage,
        epochs,
        batch_size: batchSize,
        max_samples: maxSamples,
      };
      const s = await startTraining(req);
      setStatus(s);
    } catch (e: unknown) {
      setStartError(e instanceof Error ? e.message : "Failed to initialize model training");
    } finally {
      setStarting(false);
    }
  };

  const isRunning = status?.is_training ?? false;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <Link href="/" style={{ display: "flex", alignItems: "center", gap: 12, textDecoration: "none" }}>
            <span className="brand-mark">
              <Telescope size={20} />
            </span>
            <div>
              <div className="brand-name">SOLAR OBSERVATORY<span> / MODEL TRAINING CONSOLE</span></div>
              <div className="brand-caption">NEURAL NETWORK OPTIMIZATION & FINE-TUNING</div>
            </div>
          </Link>
        </div>

        <nav className="topbar-right" aria-label="Observatory navigation">
          <Link href="/" className="text-button">Live Monitor</Link>
          <Link href="/dataset" className="text-button">Dataset</Link>
          <Link href="/analyze" className="text-button">Analysis</Link>
          <Link href="/training" className="text-button active">Training</Link>
        </nav>
      </header>

      <section className="page-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-rule" /> PIPELINE FINE-TUNING</div>
          <h1>Model <span>Training & Calibration</span></h1>
          <p className="heading-copy">
            Execute asynchronous background fine-tuning routines for Stage 1 CNN detection
            and Stage 2 U-Net segmentation models on calibrated GONG observations.
          </p>
        </div>
        <div className="observation-note">
          <Cpu size={15} style={{ color: "var(--accent)" }} />
          <span>Compute State: <strong>{isRunning ? "OPTIMIZING" : "STANDBY"}</strong><br />PyTorch Engine</span>
        </div>
      </section>

      <div className="workspace">
        {/* Main Status & History Area */}
        <section className="viewer-column" aria-label="Training status and metrics">
          {/* Active Training Status Card */}
          <div className="viewer-card" style={{ padding: "20px 22px" }}>
            <div className="viewer-toolbar" style={{ padding: "0 0 14px 0", borderBottom: "1px solid var(--border)" }}>
              <div className="viewer-title-group">
                <span
                  className="live-square"
                  style={{ background: isRunning ? "var(--accent)" : "var(--success)" }}
                />
                <div>
                  <div className="viewer-title">OPTIMIZATION STATUS</div>
                  <div className="viewer-subtitle">
                    {status?.current_stage ? `Target: ${status.current_stage}` : "Compute cluster idle"}
                  </div>
                </div>
              </div>
              <div className="frame-meta">
                <span className="meta-label">PROGRESS</span>
                <span style={{ fontFamily: "var(--font-plex-mono), monospace", fontWeight: 600 }}>
                  {status ? `${status.progress_percentage.toFixed(0)}%` : "0%"}
                </span>
              </div>
            </div>

            {/* Progress bar */}
            <div
              style={{
                height: 6,
                background: "var(--surface-muted)",
                borderRadius: 4,
                overflow: "hidden",
                margin: "18px 0 14px",
                border: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${status ? Math.max(status.progress_percentage, isRunning ? 5 : 0) : 0}%`,
                  background: isRunning ? "var(--accent)" : "var(--success)",
                  transition: "width 400ms ease",
                }}
              />
            </div>

            <div className="connection-readout" style={{ padding: "8px 0 0" }}>
              <div className="readout-line">
                <span>SYSTEM MESSAGE</span>
                <strong style={{ color: isRunning ? "var(--accent)" : "var(--text)" }}>
                  {status?.status_message || "Awaiting task dispatch"}
                </strong>
              </div>
              <div className="readout-line">
                <span>ACTIVE STAGE</span>
                <strong>{status?.current_stage || "None"}</strong>
              </div>
            </div>
          </div>

          {/* Historical Training Metrics */}
          {history && (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "16px 0 12px" }}>
                <BarChart3 size={16} style={{ color: "var(--text-secondary)" }} />
                <span className="meta-label" style={{ fontSize: 11 }}>CONVERGENCE HISTORY</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                {/* Stage 1 CNN History */}
                <div className="viewer-card" style={{ padding: "16px 18px" }}>
                  <div className="viewer-title" style={{ marginBottom: 10, display: "flex", justifyContent: "space-between" }}>
                    <span>STAGE 1 CNN (PRESENCE)</span>
                    <span className="meta-label">CHECKPOINT</span>
                  </div>

                  {history.stage1_cnn ? (
                    <div className="connection-readout" style={{ padding: 0 }}>
                      {Object.entries(history.stage1_cnn).map(([metric, values]) => (
                        <div className="readout-line" key={metric}>
                          <span>{metric.toUpperCase().replace(/_/g, " ")}</span>
                          <strong style={{ fontFamily: "var(--font-plex-mono), monospace" }}>
                            {Array.isArray(values) && values.length > 0
                              ? Number(values[values.length - 1]).toFixed(4)
                              : typeof values === "number"
                              ? values.toFixed(4)
                              : String(values)}
                          </strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p style={{ color: "var(--text-muted)", fontSize: 12, margin: "8px 0" }}>
                      No prior checkpoint records available for Stage 1 CNN.
                    </p>
                  )}
                </div>

                {/* Stage 2 U-Net History */}
                <div className="viewer-card" style={{ padding: "16px 18px" }}>
                  <div className="viewer-title" style={{ marginBottom: 10, display: "flex", justifyContent: "space-between" }}>
                    <span>STAGE 2 U-NET (SEGMENTER)</span>
                    <span className="meta-label">CHECKPOINT</span>
                  </div>

                  {history.stage2_unet ? (
                    <div className="connection-readout" style={{ padding: 0 }}>
                      {Object.entries(history.stage2_unet).map(([metric, values]) => (
                        <div className="readout-line" key={metric}>
                          <span>{metric.toUpperCase().replace(/_/g, " ")}</span>
                          <strong style={{ fontFamily: "var(--font-plex-mono), monospace" }}>
                            {Array.isArray(values) && values.length > 0
                              ? Number(values[values.length - 1]).toFixed(4)
                              : typeof values === "number"
                              ? values.toFixed(4)
                              : String(values)}
                          </strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p style={{ color: "var(--text-muted)", fontSize: 12, margin: "8px 0" }}>
                      No prior checkpoint records available for Stage 2 U-Net.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Right Sidebar Controls */}
        <aside className="control-column" aria-label="Training parameters">
          {/* Architecture Target */}
          <section className="control-card">
            <div className="card-heading">
              <div><span className="section-index">01</span><h2>Target Pipeline</h2></div>
            </div>

            <div style={{ padding: "16px 18px" }}>
              <div className="view-switch" role="group" style={{ display: "flex", flexDirection: "column", width: "100%", gap: 6 }}>
                {[
                  { id: "both" as Stage, label: "Dual Pipeline (CNN + U-Net)" },
                  { id: "cnn" as Stage, label: "Stage 1 CNN (Detection Only)" },
                  { id: "unet" as Stage, label: "Stage 2 U-Net (Segmentation Only)" },
                ].map(({ id, label }) => (
                  <button
                    key={id}
                    className={`view-option ${stage === id ? "selected" : ""}`}
                    style={{ textAlign: "left", width: "100%", padding: "9px 12px" }}
                    onClick={() => setStage(id)}
                    type="button"
                    disabled={isRunning}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </section>

          {/* Hyperparameters */}
          <section className="control-card">
            <div className="card-heading">
              <div><span className="section-index">02</span><h2>Hyperparameters</h2></div>
            </div>

            <div style={{ padding: "16px 18px", display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label className="meta-label" style={{ display: "block", marginBottom: 6 }}>
                  TRAINING EPOCHS
                </label>
                <input
                  className="station-select"
                  type="number"
                  min={1}
                  max={50}
                  value={epochs}
                  disabled={isRunning}
                  onChange={(e) => setEpochs(Math.max(1, Math.min(50, Number(e.target.value))))}
                />
              </div>

              <div>
                <label className="meta-label" style={{ display: "block", marginBottom: 6 }}>
                  MINIBATCH SIZE
                </label>
                <input
                  className="station-select"
                  type="number"
                  min={1}
                  max={64}
                  value={batchSize}
                  disabled={isRunning}
                  onChange={(e) => setBatchSize(Math.max(1, Math.min(64, Number(e.target.value))))}
                />
              </div>

              <div>
                <label className="meta-label" style={{ display: "block", marginBottom: 6 }}>
                  OBSERVATION SAMPLE CAP
                </label>
                <input
                  className="station-select"
                  type="number"
                  min={10}
                  max={1000}
                  value={maxSamples}
                  disabled={isRunning}
                  onChange={(e) => setMaxSamples(Math.max(10, Math.min(1000, Number(e.target.value))))}
                />
              </div>
            </div>
          </section>

          {/* Launch Execution */}
          <section className="control-card">
            <div className="card-heading">
              <div><span className="section-index">03</span><h2>Task Dispatch</h2></div>
            </div>

            <div style={{ padding: "16px 18px" }}>
              {startError && (
                <div className="error-message" role="alert" style={{ marginBottom: 12 }}>
                  {startError}
                </div>
              )}

              <button
                className="button button-primary"
                style={{ width: "100%", gap: 8 }}
                onClick={() => void handleStart()}
                disabled={isRunning || starting}
                type="button"
              >
                {isRunning ? (
                  <>
                    <RefreshCw size={14} className="ds-spinning" /> Optimization In Progress
                  </>
                ) : starting ? (
                  "Dispatching Task..."
                ) : (
                  <>
                    <Play size={14} fill="currentColor" /> Initiate Model Training
                  </>
                )}
              </button>

              <p className="backend-note" style={{ marginTop: 12, marginInline: 0 }}>
                <span className="note-line" />
                Training runs asynchronously via FastAPI BackgroundTasks. Checkpoint weights are saved to backend/models.
              </p>
            </div>
          </section>
        </aside>
      </div>

      <footer className="page-footer">
        <span>Solar Observatory Machine Learning Console | National Solar Observatory GONG Program</span>
        <span>Deep convolutional neural networks calibrated against MAGFiLO H-alpha ground truth.</span>
      </footer>
    </main>
  );
}
