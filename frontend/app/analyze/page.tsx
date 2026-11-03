"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  FileImage,
  FlaskConical,
  ScanSearch,
  Upload,
  X,
  Telescope,
  Cpu,
  Layers,
  Activity,
  RefreshCw,
  Download,
  Ruler,
  Maximize2,
  Compass,
  GitBranch,
  Navigation,
  HelpCircle,
} from "lucide-react";
import {
  analyzeImage,
  type FullAnalysisResponse,
} from "@/lib/api";

export default function AnalyzePage() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [threshold, setThreshold] = useState<number>(0.4);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<FullAnalysisResponse | null>(null);
  const [selectedFilamentId, setSelectedFilamentId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  function handleFile(file?: File) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file (JPEG or PNG).");
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setError("The observation image must be smaller than 25 MB.");
      return;
    }

    setError("");
    setImageUrl(URL.createObjectURL(file));
    setRawFile(file);
    setFileName(file.name);
    setResult(null);
  }

  function clearImage() {
    setImageUrl(null);
    setRawFile(null);
    setFileName("");
    setResult(null);
    setError("");

    if (fileInput.current) fileInput.current.value = "";
  }

  async function runAnalysis() {
    if (!rawFile) {
      setError("Upload a solar observation image before initiating inference.");
      return;
    }

    setError("");
    setIsAnalyzing(true);
    setResult(null);

    try {
      const resp = await analyzeImage(rawFile, threshold);
      setResult(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference pipeline failure.");
    } finally {
      setIsAnalyzing(false);
    }
  }

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
              <div className="brand-name">SOLAR OBSERVATORY<span> / FILAMENT ANALYSIS</span></div>
              <div className="brand-caption">TWO-STAGE NEURAL DETECTION & SEGMENTATION PIPELINE</div>
            </div>
          </Link>
        </div>

        <nav className="topbar-right" aria-label="Observatory navigation">
          <Link href="/" className="text-button">Live Monitor</Link>
          <Link href="/dataset" className="text-button">Dataset</Link>
          <Link href="/analyze" className="text-button active">Analysis</Link>
          <Link href="/training" className="text-button">Training</Link>
        </nav>
      </header>

      {/* Page Heading */}
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-rule" /> CHROMOSPHERIC SEGMENTATION PIPELINE
          </div>
          <h1>Observation <span>Analysis & Segmentation</span></h1>
          <p className="heading-copy">
            Process high-resolution H-alpha observations through Stage 1 CNN screening and Stage 2 U-Net
            semantic segmentation for automated filament detection and polygon quantification.
          </p>
        </div>
        <div className="observation-note">
          <FlaskConical size={15} style={{ color: "var(--accent)" }} />
          <span>Pipeline: <strong>CNN + U-Net</strong><br />PyTorch Engine</span>
        </div>
      </section>

      {/* Main Workspace Grid */}
      <div className="workspace">
        {/* Left Column: Observation Input */}
        <section className="viewer-column" aria-label="Input solar observation">
          <div className="viewer-card">
            <div className="viewer-toolbar">
              <div className="viewer-title-group">
                <span className="section-index" style={{ width: 24, height: 24, fontSize: 10 }}>01</span>
                <div>
                  <div className="viewer-title">INPUT OBSERVATION</div>
                  <div className="viewer-subtitle">
                    {fileName ? fileName : "Upload H-alpha full-disk observation (JPEG / PNG)"}
                  </div>
                </div>
              </div>
              {fileName && (
                <button
                  type="button"
                  className="button"
                  style={{ padding: "4px 10px", height: 28, fontSize: 11 }}
                  onClick={clearImage}
                >
                  <X size={12} style={{ marginRight: 4 }} /> Clear
                </button>
              )}
            </div>

            {/* Dropzone / Preview Stage */}
            <div style={{ padding: 18 }}>
              <div
                className={`upload-dropzone ${imageUrl ? "has-image" : ""}`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  handleFile(e.dataTransfer.files[0]);
                }}
              >
                {imageUrl ? (
                  <div style={{ position: "relative", width: "100%", textAlign: "center" }}>
                    <button
                      type="button"
                      className="remove-img-btn"
                      onClick={clearImage}
                      aria-label="Remove uploaded image"
                    >
                      <X size={15} />
                    </button>
                    <img
                      className="uploaded-preview-img"
                      src={imageUrl}
                      alt="Uploaded solar observation"
                    />
                    <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-secondary)", fontFamily: "var(--font-plex-mono), monospace" }}>
                      {fileName}
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="upload-icon-box">
                      <Upload size={22} />
                    </div>
                    <div className="upload-title">Select Observation Image</div>
                    <div className="upload-desc">
                      Drag and drop an H-alpha observation here, or browse your local file system.
                    </div>
                    <button
                      type="button"
                      className="button"
                      onClick={() => fileInput.current?.click()}
                    >
                      Browse Local Files
                    </button>
                    <span style={{ display: "block", marginTop: 12, fontSize: 11, color: "var(--text-muted)" }}>
                      JPEG or PNG • Maximum 25 MB
                    </span>
                  </>
                )}
              </div>

              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => handleFile(e.target.files?.[0])}
              />

              {error && (
                <div className="error-message" role="alert" style={{ marginTop: 12, marginInline: 0 }}>
                  <AlertCircle size={14} style={{ display: "inline", verticalAlign: "middle", marginRight: 6 }} />
                  {error}
                </div>
              )}

              {/* Threshold Slider */}
              <div style={{ marginTop: 18, padding: "14px 16px", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--surface-muted)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span className="meta-label">DETECTION THRESHOLD</span>
                  <span className="threshold-value">{threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="threshold-range"
                  style={{ "--range-progress": `${((threshold - 0.1) / 0.8) * 100}%` } as React.CSSProperties}
                />
                <div className="range-scale">
                  <span>0.10 (High Sensitivity)</span>
                  <span>0.90 (High Precision)</span>
                </div>
              </div>

              {/* Action Button */}
              <button
                type="button"
                className="button button-primary"
                onClick={runAnalysis}
                disabled={!imageUrl || isAnalyzing}
                style={{ width: "100%", marginTop: 16, height: 42, gap: 8 }}
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw size={15} className="ds-spinning" />
                    Executing Neural Pipeline...
                  </>
                ) : (
                  <>
                    <ScanSearch size={15} />
                    Execute Filament Analysis
                    <ArrowRight size={14} style={{ marginLeft: 4 }} />
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* Right Column: Pipeline Output */}
        <section className="control-column" aria-label="Pipeline results">
          <div className="viewer-card">
            <div className="viewer-toolbar">
              <div className="viewer-title-group">
                <span className="section-index" style={{ width: 24, height: 24, fontSize: 10 }}>02</span>
                <div>
                  <div className="viewer-title">PIPELINE OUTPUT</div>
                  <div className="viewer-subtitle">
                    {result ? "Quantified segmentation metrics & overlay" : "Inference results pending"}
                  </div>
                </div>
              </div>
              <div className="frame-meta">
                <span className="meta-label">STATE</span>
                <span style={{ fontWeight: 600, color: result ? "var(--success)" : "var(--text-muted)" }}>
                  {result ? "COMPLETE" : "STANDBY"}
                </span>
              </div>
            </div>

            <div style={{ padding: 18 }}>
              {!result ? (
                <div className="empty-state" style={{ minHeight: 380, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
                  <div className="empty-sun"><span /></div>
                  <div className="empty-title">AWAITING OBSERVATION</div>
                  <p style={{ maxWidth: 320 }}>
                    Upload an observation and execute the pipeline to generate presence classification,
                    U-Net segmentation contours, and total filament chromospheric area.
                  </p>

                  <div style={{ width: "100%", marginTop: 20, textAlign: "left" }}>
                    <div className="readout-line" style={{ padding: "10px 0" }}>
                      <span>Stage 1: Basic CNN</span>
                      <strong style={{ color: "var(--text-muted)", fontSize: 11 }}>Standby (Presence & Chirality)</strong>
                    </div>
                    <div className="readout-line" style={{ padding: "10px 0" }}>
                      <span>Stage 2: U-Net Segmenter</span>
                      <strong style={{ color: "var(--text-muted)", fontSize: 11 }}>Standby (Semantic Contours)</strong>
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  {/* Status notification banner: Success or Non-Solar Rejection */}
                  {result.is_solar_observation === false ? (
                    <div style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 10,
                      padding: "12px 14px",
                      border: "1px solid #fbd38d",
                      borderRadius: "var(--radius-sm)",
                      background: "#fffaf0",
                      color: "#c05621",
                      marginBottom: 16,
                      fontSize: 12,
                    }}>
                      <AlertCircle size={18} style={{ flexShrink: 0, color: "#dd6b20", marginTop: 1 }} />
                      <div>
                        <strong style={{ fontSize: 13, color: "#9c4221", display: "block" }}>
                          Non-Solar Observation Detected
                        </strong>
                        <div style={{ fontSize: 11, color: "#7b341e", marginTop: 2, lineHeight: 1.4 }}>
                          {result.validation_message || "Uploaded image does not match H-alpha solar disk geometry. Filament analysis is restricted strictly to solar observations."}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      padding: "10px 14px",
                      border: "1px solid #d6e5da",
                      borderRadius: "var(--radius-sm)",
                      background: "#f0f7f2",
                      color: "var(--success)",
                      marginBottom: 16,
                      fontSize: 12,
                    }}>
                      <CheckCircle2 size={16} style={{ flexShrink: 0 }} />
                      <div>
                        <strong>Inference Completed Successfully</strong>
                        <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>
                          Latency: {result.total_processing_time_ms.toFixed(0)} ms • Device: {String(result.image_metadata.device || "CPU")}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Segmentation Overlay Image */}
                  {result.overlay_image_base64 && (
                    <div style={{
                      borderRadius: "var(--radius-md)",
                      overflow: "hidden",
                      border: "1px solid var(--border)",
                      background: "#10151b",
                      marginBottom: 16,
                    }}>
                      <div style={{
                        padding: "8px 12px",
                        borderBottom: "1px solid rgba(255,255,255,0.08)",
                        fontSize: 11,
                        color: "#8994a4",
                        fontFamily: "var(--font-plex-mono), monospace",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                      }}>
                        <span style={{ fontWeight: 600, letterSpacing: "0.04em" }}>
                          {result.is_solar_observation === false ? "REJECTION OVERLAY" : "SEGMENTATION OVERLAY (U-NET)"}
                        </span>
                        {result.is_solar_observation === false ? (
                          <span style={{ color: "#dd6b20", fontSize: 10 }}>Out of Domain</span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => {
                              if (!result?.overlay_image_base64) return;
                              const a = document.createElement("a");
                              a.href = result.overlay_image_base64;
                              const baseName = fileName ? fileName.replace(/\.[^/.]+$/, "") : "solar_filament_overlay";
                              a.download = `${baseName}_segmented_overlay.jpg`;
                              document.body.appendChild(a);
                              a.click();
                              document.body.removeChild(a);
                            }}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 5,
                              background: "rgba(235, 94, 40, 0.15)",
                              border: "1px solid rgba(235, 94, 40, 0.4)",
                              borderRadius: 4,
                              color: "#ff9e6c",
                              padding: "4px 10px",
                              fontSize: 11,
                              fontWeight: 600,
                              cursor: "pointer",
                              fontFamily: "var(--font-plex-mono), monospace",
                              transition: "all 0.15s ease",
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.background = "rgba(235, 94, 40, 0.35)";
                              e.currentTarget.style.color = "#ffffff";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.background = "rgba(235, 94, 40, 0.15)";
                              e.currentTarget.style.color = "#ff9e6c";
                            }}
                            title="Download segmented overlay image"
                          >
                            <Download size={13} /> Download Overlay Image
                          </button>
                        )}
                      </div>
                      <img
                        src={result.overlay_image_base64}
                        alt="Segmented solar filament overlay"
                        style={{ width: "100%", maxHeight: 280, objectFit: "contain", display: "block" }}
                      />
                    </div>
                  )}

                  {/* 2x2 Metrics Cards */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
                    <div className="metric-card" style={{ padding: 12 }}>
                      <span className="metric-label">FILAMENTS DETECTED</span>
                      <strong style={{ fontSize: 18 }}>
                        {result.stage2_segmentation.num_filaments_found}
                      </strong>
                      <span className="metric-foot">
                        {result.is_solar_observation === false ? "Non-solar rejection" : "discrete structures"}
                      </span>
                    </div>

                    <div className="metric-card" style={{ padding: 12 }}>
                      <span className="metric-label">TOTAL PIXEL AREA</span>
                      <strong style={{ fontSize: 18 }}>
                        {result.stage2_segmentation.total_filament_area.toLocaleString()}<small> px</small>
                      </strong>
                      <span className="metric-foot">
                        {result.is_solar_observation === false ? "0.00% (out of domain)" : `${result.stage2_segmentation.coverage_percentage.toFixed(2)}% disk coverage`}
                      </span>
                    </div>

                    <div className="metric-card" style={{ padding: 12 }}>
                      <span className="metric-label">STAGE 1 PRESENCE</span>
                      <strong style={{
                        fontSize: 14,
                        color: result.is_solar_observation === false
                          ? "#dd6b20"
                          : (result.stage1_cnn.has_filament ? "var(--accent)" : "var(--success)")
                      }}>
                        {result.is_solar_observation === false
                          ? "NON-SOLAR TARGET"
                          : (result.stage1_cnn.has_filament ? "FILAMENT PRESENT" : "QUIET SUN")}
                      </strong>
                      <span className="metric-foot">
                        {result.is_solar_observation === false
                          ? "Filament classifier inactive"
                          : `Confidence: ${(result.stage1_cnn.filament_probability * 100).toFixed(1)}%`}
                      </span>
                    </div>

                    <div className="metric-card" style={{ padding: 12 }}>
                      <span className="metric-label">DOMINANT CHIRALITY</span>
                      <strong style={{ fontSize: 14 }}>
                        {result.is_solar_observation === false ? "N/A" : result.stage1_cnn.dominant_category}
                      </strong>
                      <span className="metric-foot">
                        {result.is_solar_observation === false
                          ? "State: Disqualified"
                          : `State: ${result.stage2_segmentation.has_high_solar_activity ? "Elevated" : "Nominal"}`}
                      </span>
                    </div>
                  </div>

                  {/* Readout Summary */}
                  <div className="connection-readout" style={{ padding: 0 }}>
                    <div className="readout-line">
                      <span>PIPELINE LATENCY</span>
                      <strong style={{ fontFamily: "var(--font-plex-mono), monospace" }}>
                        {result.total_processing_time_ms.toFixed(1)} ms
                      </strong>
                    </div>
                    <div className="readout-line">
                      <span>SOLAR ACTIVITY STATUS</span>
                      <strong style={{
                        color: result.is_solar_observation === false
                          ? "#dd6b20"
                          : (result.stage2_segmentation.has_high_solar_activity ? "var(--accent)" : "var(--success)")
                      }}>
                        {result.is_solar_observation === false
                          ? "INVALID OBSERVATION (NON-SOLAR)"
                          : (result.stage2_segmentation.has_high_solar_activity ? "ELEVATED (Area > 1000 px)" : "NOMINAL BACKGROUND")}
                      </strong>
                    </div>
                  </div>

                  {/* Morphological & Heliographic Q&A Inspection Panel */}
                  {result.is_solar_observation !== false && result.stage2_segmentation.filaments.length > 0 && (
                    <div style={{
                      marginTop: 20,
                      paddingTop: 18,
                      borderTop: "1px solid var(--border)",
                    }}>
                      <div style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        marginBottom: 12,
                      }}>
                        <div>
                          <h4 style={{ fontSize: 13, fontWeight: 700, margin: 0, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                            <Compass size={15} style={{ color: "var(--accent)" }} />
                            Filament Morphological & Heliographic Q&A
                          </h4>
                          <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                            Select any detected structure to inspect physical geometry and solar position
                          </span>
                        </div>
                        <span style={{
                          fontSize: 10,
                          fontFamily: "var(--font-plex-mono), monospace",
                          background: "rgba(255, 140, 20, 0.12)",
                          color: "#c05621",
                          padding: "3px 8px",
                          borderRadius: 4,
                          border: "1px solid rgba(255, 140, 20, 0.25)"
                        }}>
                          {result.stage2_segmentation.filaments.length} STRUCTURES
                        </span>
                      </div>

                      {/* Filament Selector Pill List */}
                      <div style={{
                        display: "flex",
                        gap: 6,
                        overflowX: "auto",
                        paddingBottom: 8,
                        marginBottom: 14,
                      }}>
                        {result.stage2_segmentation.filaments.map((f, idx) => {
                          const isSelected = (selectedFilamentId || result.stage2_segmentation.filaments[0].id) === f.id;
                          return (
                            <button
                              key={f.id}
                              type="button"
                              onClick={() => setSelectedFilamentId(f.id)}
                              style={{
                                flexShrink: 0,
                                padding: "5px 10px",
                                borderRadius: "var(--radius-sm)",
                                fontSize: 11,
                                fontFamily: "var(--font-plex-mono), monospace",
                                fontWeight: isSelected ? 700 : 500,
                                border: isSelected ? "1px solid var(--accent)" : "1px solid var(--border)",
                                background: isSelected ? "rgba(235, 94, 40, 0.12)" : "var(--bg-elevated)",
                                color: isSelected ? "var(--accent)" : "var(--text-secondary)",
                                cursor: "pointer",
                                transition: "all 0.15s ease",
                              }}
                            >
                              #{idx + 1} ({Math.round(f.area)} px)
                            </button>
                          );
                        })}
                      </div>

                      {/* Active Filament Morphological Q&A Grid */}
                      {(() => {
                        const activeFilament = result.stage2_segmentation.filaments.find(
                          (f) => f.id === (selectedFilamentId || result.stage2_segmentation.filaments[0].id)
                        ) || result.stage2_segmentation.filaments[0];

                        if (!activeFilament) return null;
                        const m = activeFilament.morphology;
                        const qa = activeFilament.diagnostic_qa;

                        return (
                          <div style={{
                            background: "var(--bg-elevated)",
                            border: "1px solid var(--border)",
                            borderRadius: "var(--radius-md)",
                            padding: 14,
                          }}>
                            <div style={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              marginBottom: 12,
                              paddingBottom: 8,
                              borderBottom: "1px solid var(--border)",
                            }}>
                              <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)" }}>
                                Analysis for <strong style={{ color: "var(--accent)" }}>{activeFilament.id.toUpperCase().replace("_", " ")}</strong>
                              </span>
                              <span style={{ fontSize: 11, color: "var(--text-secondary)", fontFamily: "var(--font-plex-mono), monospace" }}>
                                Confidence: {(activeFilament.confidence * 100).toFixed(1)}% • Area: {Math.round(activeFilament.area)} px
                              </span>
                            </div>

                            <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 10 }}>
                              {/* Q1: How long is this filament? */}
                              <div style={{
                                background: "rgba(0, 0, 0, 0.02)",
                                border: "1px solid var(--border)",
                                borderRadius: "var(--radius-sm)",
                                padding: "10px 12px",
                              }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                                    <Ruler size={13} style={{ color: "var(--accent)" }} />
                                    How long is this filament?
                                  </span>
                                  {m && (
                                    <span style={{
                                      fontSize: 10,
                                      fontWeight: 700,
                                      fontFamily: "var(--font-plex-mono), monospace",
                                      color: "var(--accent)",
                                      background: "rgba(235, 94, 40, 0.10)",
                                      padding: "2px 6px",
                                      borderRadius: 3,
                                    }}>
                                      {m.length_megameters} Mm
                                    </span>
                                  )}
                                </div>
                                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                                  {qa?.how_long || (m ? `${m.length_px} px (~${m.length_km.toLocaleString()} km / ${m.length_megameters} Mm)` : "N/A")}
                                </div>
                              </div>

                              {/* Q2: How wide is it? */}
                              <div style={{
                                background: "rgba(0, 0, 0, 0.02)",
                                border: "1px solid var(--border)",
                                borderRadius: "var(--radius-sm)",
                                padding: "10px 12px",
                              }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                                    <Maximize2 size={13} style={{ color: "var(--accent)" }} />
                                    How wide is it?
                                  </span>
                                  {m && (
                                    <span style={{
                                      fontSize: 10,
                                      fontFamily: "var(--font-plex-mono), monospace",
                                      color: "var(--text-secondary)",
                                      background: "var(--bg-surface)",
                                      padding: "2px 6px",
                                      borderRadius: 3,
                                      border: "1px solid var(--border)",
                                    }}>
                                      Aspect: {m.aspect_ratio}:1
                                    </span>
                                  )}
                                </div>
                                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                                  {qa?.how_wide || (m ? `${m.width_px} px (~${m.width_km.toLocaleString()} km), aspect ratio ${m.aspect_ratio}:1` : "N/A")}
                                </div>
                              </div>

                              {/* Q3: Is it straight, curved or fragmented? */}
                              <div style={{
                                background: "rgba(0, 0, 0, 0.02)",
                                border: "1px solid var(--border)",
                                borderRadius: "var(--radius-sm)",
                                padding: "10px 12px",
                              }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                                    <Compass size={13} style={{ color: "var(--accent)" }} />
                                    Is it straight, curved or fragmented?
                                  </span>
                                  {m && (
                                    <span style={{
                                      fontSize: 10,
                                      fontWeight: 700,
                                      fontFamily: "var(--font-plex-mono), monospace",
                                      color: m.curvature_class === "Straight" ? "#2b6cb0" : (m.curvature_class === "Curved" ? "#c05621" : "#c53030"),
                                      background: m.curvature_class === "Straight" ? "rgba(49, 130, 206, 0.12)" : (m.curvature_class === "Curved" ? "rgba(221, 107, 32, 0.12)" : "rgba(229, 62, 62, 0.12)"),
                                      padding: "2px 6px",
                                      borderRadius: 3,
                                    }}>
                                      {m.curvature_class.toUpperCase()}
                                    </span>
                                  )}
                                </div>
                                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                                  {qa?.curvature || (m ? `${m.curvature_class} (sinuosity index: ${m.sinuosity})` : "N/A")}
                                </div>
                              </div>

                              {/* Q4: Does it have multiple branches? */}
                              <div style={{
                                background: "rgba(0, 0, 0, 0.02)",
                                border: "1px solid var(--border)",
                                borderRadius: "var(--radius-sm)",
                                padding: "10px 12px",
                              }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                                    <GitBranch size={13} style={{ color: "var(--accent)" }} />
                                    Does it have multiple branches?
                                  </span>
                                  {m && (
                                    <span style={{
                                      fontSize: 10,
                                      fontFamily: "var(--font-plex-mono), monospace",
                                      color: m.has_multiple_branches ? "#c05621" : "#2f855a",
                                      background: m.has_multiple_branches ? "rgba(221, 107, 32, 0.12)" : "rgba(56, 161, 105, 0.12)",
                                      padding: "2px 6px",
                                      borderRadius: 3,
                                    }}>
                                      {m.has_multiple_branches ? `${m.branch_count} Branches` : "Single Spine"}
                                    </span>
                                  )}
                                </div>
                                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                                  {qa?.branches || (m ? (m.has_multiple_branches ? `Yes (${m.branch_count} branches)` : "No (single spine)") : "N/A")}
                                </div>
                              </div>

                              {/* Q5: Where is it located relative to the solar disk? */}
                              <div style={{
                                background: "rgba(0, 0, 0, 0.02)",
                                border: "1px solid var(--border)",
                                borderRadius: "var(--radius-sm)",
                                padding: "10px 12px",
                              }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                                    <Navigation size={13} style={{ color: "var(--accent)" }} />
                                    Where is it located relative to the solar disk?
                                  </span>
                                  {m && (
                                    <span style={{
                                      fontSize: 10,
                                      fontFamily: "var(--font-plex-mono), monospace",
                                      color: "var(--text-secondary)",
                                      background: "var(--bg-surface)",
                                      padding: "2px 6px",
                                      borderRadius: 3,
                                      border: "1px solid var(--border)",
                                    }}>
                                      r/R_sun = {m.radial_distance_ratio}
                                    </span>
                                  )}
                                </div>
                                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                                  {qa?.location || (m ? m.disk_location_description : "N/A")}
                                </div>
                              </div>

                              {/* Q6: Is it close to the solar limb? */}
                              <div style={{
                                background: "rgba(0, 0, 0, 0.02)",
                                border: "1px solid var(--border)",
                                borderRadius: "var(--radius-sm)",
                                padding: "10px 12px",
                              }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                                    <HelpCircle size={13} style={{ color: "var(--accent)" }} />
                                    Is it close to the solar limb?
                                  </span>
                                  {m && (
                                    <span style={{
                                      fontSize: 10,
                                      fontWeight: 700,
                                      fontFamily: "var(--font-plex-mono), monospace",
                                      color: m.is_close_to_limb ? "#c53030" : "#2f855a",
                                      background: m.is_close_to_limb ? "rgba(229, 62, 62, 0.12)" : "rgba(56, 161, 105, 0.12)",
                                      padding: "2px 6px",
                                      borderRadius: 3,
                                    }}>
                                      {m.proximity_to_limb.toUpperCase()}
                                    </span>
                                  )}
                                </div>
                                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                                  {qa?.close_to_limb || (m ? (m.is_close_to_limb ? "Yes, close to limb" : "No, intermediate disk") : "N/A")}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>

      {/* Observation Processing Protocol Workflow */}
      <section className="workflow-section">
        <div style={{ marginBottom: 12 }}>
          <div className="eyebrow">
            <span className="eyebrow-rule" /> AUTOMATED PIPELINE PROTOCOL
          </div>
          <h2 style={{ fontSize: 18, margin: "6px 0 4px" }}>Observation Processing Sequence</h2>
          <p style={{ fontSize: 13, margin: 0, color: "var(--text-secondary)" }}>
            Standardized two-tier deep learning architecture for high-resolution chromospheric filament detection.
          </p>
        </div>

        <div className="workflow-steps-grid">
          <div className="workflow-step-card">
            <span className="workflow-step-number">STEP 01</span>
            <h3>Observation Ingestion</h3>
            <p>Full-disk H-alpha observation is normalized [0, 1] and calibrated for model input tensors.</p>
          </div>

          <div className="workflow-step-card">
            <span className="workflow-step-number">STEP 02</span>
            <h3>Stage 1 CNN Screening</h3>
            <p>Fast convolutional classifier screens filament presence and determines magnetic twist chirality in ~8.5 ms.</p>
          </div>

          <div className="workflow-step-card">
            <span className="workflow-step-number">STEP 03</span>
            <h3>Stage 2 U-Net Segmentation</h3>
            <p>Deep encoder-decoder generates pixel-accurate probability heatmaps and binary segmentation masks.</p>
          </div>

          <div className="workflow-step-card">
            <span className="workflow-step-number">STEP 04</span>
            <h3>Morphological Telemetry</h3>
            <p>OpenCV extracts polygon coordinates, calculates calibrated area, and evaluates activity alert triggers.</p>
          </div>
        </div>
      </section>

      {/* Universal Footer */}
      <footer className="page-footer">
        <span>Solar Observatory | National Solar Observatory GONG Program</span>
        <span>Automated H-Alpha Solar Filament Analysis & Semantic Segmentation Protocol</span>
      </footer>
    </main>
  );
}
