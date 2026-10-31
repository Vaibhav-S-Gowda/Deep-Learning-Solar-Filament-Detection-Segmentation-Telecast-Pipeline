"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Activity,
  Database,
  FileImage,
  Layers,
  RefreshCw,
  Telescope,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import {
  getDatasetStats,
  getDatasetSample,
  type DatasetStats,
  type DatasetSample,
} from "@/lib/api";

export default function DatasetPage() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [sample, setSample] = useState<DatasetSample | null>(null);
  const [statsError, setStatsError] = useState("");
  const [sampleError, setSampleError] = useState("");
  const [loadingStats, setLoadingStats] = useState(true);
  const [loadingSample, setLoadingSample] = useState(false);

  useEffect(() => {
    getDatasetStats()
      .then(setStats)
      .catch((e: unknown) =>
        setStatsError(
          e instanceof Error ? e.message : "Failed to load dataset statistics"
        )
      )
      .finally(() => setLoadingStats(false));
  }, []);

  async function fetchSample() {
    setLoadingSample(true);
    setSampleError("");

    try {
      const result = await getDatasetSample();
      setSample(result);
    } catch (e: unknown) {
      setSampleError(
        e instanceof Error ? e.message : "Failed to load sample image"
      );
    } finally {
      setLoadingSample(false);
    }
  }

  const totalImages = stats
    ? stats.total_train_images + stats.total_test_images
    : null;

  return (
    <main className="app-shell">
      {/* Universal Topbar */}
      <header className="topbar">
        <div className="brand-lockup">
          <Link href="/" style={{ display: "flex", alignItems: "center", gap: 12, textDecoration: "none" }}>
            <span className="brand-mark">
              <Telescope size={20} />
            </span>
            <div>
              <div className="brand-name">SOLAR OBSERVATORY<span> / DATASET EXPLORER</span></div>
              <div className="brand-caption">MAGFILO 1.0 GROUND-TRUTH REPOSITORY</div>
            </div>
          </Link>
        </div>

        <nav className="topbar-right" aria-label="Observatory navigation">
          <Link href="/" className="text-button">Live Monitor</Link>
          <Link href="/dataset" className="text-button active">Dataset</Link>
          <Link href="/analyze" className="text-button">Analysis</Link>
          <Link href="/training" className="text-button">Training</Link>
        </nav>
      </header>

      {/* Page Heading */}
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-rule" /> MAGFILO 1.0 COLLECTION
          </div>
          <h1>Dataset <span>Explorer & Ground Truth</span></h1>
          <p className="heading-copy">
            Inspect calibrated full-disk H-alpha solar observations and verified ground-truth filament annotations
            from the standardized MAGFiLO 1.0 dataset.
          </p>
        </div>
        <div className="observation-note">
          <Database size={15} style={{ color: "var(--accent)" }} />
          <span>Repository: <strong>MAGFiLO 1.0</strong><br />GONG H-Alpha Network</span>
        </div>
      </section>

      {/* Metrics Row */}
      <div className="metrics-row" style={{ maxWidth: 1380, margin: "0 auto 20px" }} aria-label="Dataset overview">
        <div className="metric-card">
          <span className="metric-label">TOTAL OBSERVATIONS</span>
          <strong>{loadingStats ? "—" : totalImages?.toLocaleString() ?? "—"}</strong>
          <span className="metric-foot">Calibrated full-disk frames</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">TRAINING PARTITION</span>
          <strong>{loadingStats ? "—" : stats?.total_train_images.toLocaleString() ?? "—"}</strong>
          <span className="metric-foot">Model optimization split</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">TEST PARTITION</span>
          <strong>{loadingStats ? "—" : stats?.total_test_images.toLocaleString() ?? "—"}</strong>
          <span className="metric-foot">Benchmark validation split</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">GROUND-TRUTH REGIONS</span>
          <strong>{loadingStats ? "—" : stats?.total_annotations.toLocaleString() ?? "—"}</strong>
          <span className="metric-foot">Segmented structures</span>
        </div>
      </div>

      {statsError && (
        <div className="error-message" style={{ maxWidth: 1380, margin: "0 auto 16px" }} role="alert">
          <AlertCircle size={15} style={{ display: "inline", verticalAlign: "middle", marginRight: 6 }} />
          {statsError}
        </div>
      )}

      {/* Workspace Grid */}
      <div className="workspace">
        {/* Left Column: Ground-Truth Viewer */}
        <section className="viewer-column" aria-label="Ground-truth observation display">
          <div className="viewer-card">
            <div className="viewer-toolbar">
              <div className="viewer-title-group">
                <span className="live-square" style={{ background: sample ? "var(--success)" : "var(--accent)" }} />
                <div>
                  <div className="viewer-title">GROUND-TRUTH OBSERVATION VIEWER</div>
                  <div className="viewer-subtitle">
                    {sample ? sample.file_name : "Expert-annotated reference observation"}
                    <span style={{ margin: "0 6px", opacity: 0.5 }}>/</span>
                    {sample ? `${sample.num_filaments_annotated} annotated filaments` : "Archive sample"}
                  </div>
                </div>
              </div>

              <button
                type="button"
                className="button"
                style={{ padding: "6px 14px", height: 34, gap: 7, fontSize: 12 }}
                onClick={() => void fetchSample()}
                disabled={loadingSample}
              >
                <RefreshCw size={13} className={loadingSample ? "ds-spinning" : ""} />
                {loadingSample ? "Retrieving Frame..." : "Load Sample Observation"}
              </button>
            </div>

            {/* Solar Stage */}
            <div className="image-stage" style={{ minHeight: 460 }}>
              <div className="stage-corner stage-tl" />
              <div className="stage-corner stage-tr" />
              <div className="stage-corner stage-bl" />
              <div className="stage-corner stage-br" />
              <div className="stage-grid" aria-hidden="true" />

              {sample ? (
                <img
                  className="solar-image"
                  src={
                    sample.overlay_base64.startsWith("data:")
                      ? sample.overlay_base64
                      : `data:image/jpeg;base64,${sample.overlay_base64}`
                  }
                  alt="Solar observation with ground-truth filament annotations"
                />
              ) : (
                <div className="empty-state">
                  <div className="empty-sun"><span /></div>
                  <div className="empty-title">NO OBSERVATION LOADED</div>
                  <p>Load a ground-truth observation from the MAGFiLO archive to inspect verified filament polygons.</p>
                  <button
                    type="button"
                    className="button button-primary"
                    style={{ marginTop: 14, gap: 7 }}
                    onClick={() => void fetchSample()}
                    disabled={loadingSample}
                  >
                    <RefreshCw size={14} className={loadingSample ? "ds-spinning" : ""} />
                    {loadingSample ? "Loading Archive..." : "Load Observation Sample"}
                  </button>
                </div>
              )}

              <div className="stage-coordinate coord-top">N [H-ALPHA 656.28 nm]</div>
              <div className="stage-coordinate coord-bottom">
                {sample ? `${sample.file_name} • MAGFiLO 1.0` : "STANDBY • MAGFiLO ARCHIVE"}
              </div>
            </div>

            {/* Viewer Footer */}
            <div className="viewer-footer">
              <div className="overlay-key">
                <span className="key-swatch" style={{ background: "var(--accent)" }} />
                <span>Ground-truth contour mask & polygon boundary</span>
              </div>
              {sample && (
                <span style={{ fontFamily: "var(--font-plex-mono), monospace", fontSize: 11, color: "var(--text-secondary)" }}>
                  {sample.annotations.length} verified structures
                </span>
              )}
            </div>

            {sampleError && (
              <div className="error-message" role="alert" style={{ margin: "14px 18px" }}>
                {sampleError}
              </div>
            )}

            {/* Metadata Readout */}
            {sample && (
              <div className="connection-readout" style={{ borderTop: "1px solid var(--border)", paddingTop: 4 }}>
                <div className="readout-line">
                  <span>FILE IDENTIFIER</span>
                  <strong style={{ fontFamily: "var(--font-plex-mono), monospace" }}>{sample.file_name}</strong>
                </div>
                <div className="readout-line">
                  <span>CAPTURE TIMESTAMP</span>
                  <strong>{sample.date_captured || "Reference Observation"}</strong>
                </div>
                <div className="readout-line">
                  <span>ANNOTATED STRUCTURES</span>
                  <strong style={{ color: "var(--accent)" }}>{sample.num_filaments_annotated} verified filaments</strong>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Right Column: Controls & Metadata */}
        <aside className="control-column" aria-label="Dataset parameters">
          {/* Card 01: Dataset Specification */}
          <section className="control-card">
            <div className="card-heading">
              <div>
                <span className="section-index">01</span>
                <h2>Dataset Metadata</h2>
              </div>
            </div>

            {loadingStats ? (
              <div style={{ padding: "20px 18px", color: "var(--text-muted)", fontSize: 12 }}>
                Loading metadata parameters...
              </div>
            ) : stats ? (
              <div className="connection-readout">
                <div className="readout-line">
                  <span>DATASET NAME</span>
                  <strong>{stats.dataset_name}</strong>
                </div>
                <div className="readout-line">
                  <span>OBSERVATION RESOLUTION</span>
                  <strong style={{ fontFamily: "var(--font-plex-mono), monospace" }}>{stats.image_resolution}</strong>
                </div>
                <div className="readout-line">
                  <span>PRIMARY INSTRUMENT</span>
                  <strong>{stats.instrument}</strong>
                </div>
                <div className="readout-line">
                  <span>TRAINING SET</span>
                  <strong>{stats.total_train_images.toLocaleString()} frames</strong>
                </div>
                <div className="readout-line">
                  <span>TEST SET</span>
                  <strong>{stats.total_test_images.toLocaleString()} frames</strong>
                </div>
              </div>
            ) : (
              <div style={{ padding: "16px 18px", color: "var(--text-muted)", fontSize: 12 }}>
                Dataset metadata currently unavailable.
              </div>
            )}
          </section>

          {/* Card 02: Category Distribution */}
          <section className="control-card">
            <div className="card-heading">
              <div>
                <span className="section-index">02</span>
                <h2>Chirality & Categories</h2>
              </div>
            </div>

            <div style={{ padding: "14px 18px" }}>
              {loadingStats ? (
                <div style={{ color: "var(--text-muted)", fontSize: 12 }}>Loading category counts...</div>
              ) : stats && Object.keys(stats.category_counts).length > 0 ? (
                <div>
                  {Object.entries(stats.category_counts).map(([name, count]) => (
                    <div className="category-row" key={name}>
                      <span className="category-badge">
                        <span className="category-dot" />
                        <span>{name}</span>
                      </span>
                      <strong style={{ fontFamily: "var(--font-plex-mono), monospace", color: "var(--text)" }}>
                        {count.toLocaleString()}
                      </strong>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ margin: 0, color: "var(--text-muted)", fontSize: 12 }}>
                  No category records available.
                </p>
              )}
            </div>
          </section>
        </aside>
      </div>

      {/* Universal Footer */}
      <footer className="page-footer">
        <span>Solar Observatory | National Solar Observatory GONG Program</span>
        <span>MAGFiLO 1.0 Dataset Explorer • Calibrated Ground-Truth Reference</span>
      </footer>
    </main>
  );
}
