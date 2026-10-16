"use client";
import { motion } from "framer-motion";

const ARCH_LAYERS = [
  {
    label: "Client Layer",
    color: "#A0A2AD",
    items: ["Browser / curl", "Streaming Client"],
  },
  {
    label: "API Gateway",
    color: "#FF9D42",
    items: ["FastAPI (Uvicorn)", "CORS Middleware", "Pydantic v2 Validation"],
  },
  {
    label: "Inference Engine",
    color: "#FF5C35",
    items: ["SolarPipelineService", "FilamentBasicCNN (Stage 1)", "SolarFilamentUNet (Stage 2)"],
  },
  {
    label: "ML Tracking",
    color: "#9585E8",
    items: ["MLflow 3.17.1", "SQLite Backend (mlflow.db)", "Experiment: Solar-Filament-Segmentation"],
  },
  {
    label: "Storage",
    color: "#4FC3F7",
    items: ["models/ .pth checkpoints", "outputs/ masks + overlays", "MAGFiLO 1.0 Dataset"],
  },
  {
    label: "Infrastructure",
    color: "#6B8AFF",
    items: ["Docker + docker-compose", "python:3.11-slim image", "Port 8000 (API) + 5000 (MLflow UI)"],
  },
];

const ROUTES = [
  { method: "GET",  path: "/health",                       color: "#4FC3F7" },
  { method: "POST", path: "/api/v1/detect",                color: "#FF9D42" },
  { method: "POST", path: "/api/v1/segment",               color: "#FF9D42" },
  { method: "POST", path: "/api/v1/analyze",               color: "#FF9D42" },
  { method: "POST", path: "/api/v1/telecast/process-frame",color: "#FF5C35" },
  { method: "GET",  path: "/api/v1/dataset/stats",         color: "#9585E8" },
  { method: "POST", path: "/api/v1/train/start",           color: "#6B8AFF" },
];

export default function Architecture() {
  return (
    <section
      id="architecture"
      style={{
        backgroundColor: "#08090D",
        padding: "100px 24px",
        borderTop: "1px solid rgba(255,157,66,0.08)",
      }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          style={{ marginBottom: 56 }}
        >
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.25em", textTransform: "uppercase", color: "#FF9D42" }}>
            04 — System Architecture
          </span>
          <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: "clamp(1.8rem, 4vw, 3rem)", color: "#F3F2F0", margin: "12px 0 0", letterSpacing: "-0.02em" }}>
            How the System is{" "}
            <span style={{ color: "#FF9D42" }}>Structured</span>
          </h2>
        </motion.div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
          {/* Architecture layers */}
          <div>
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.2em", textTransform: "uppercase", color: "#A0A2AD", marginBottom: 20 }}>
              System Layers
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {ARCH_LAYERS.map((layer, i) => (
                <motion.div
                  key={layer.label}
                  initial={{ x: -20, opacity: 0 }}
                  whileInView={{ x: 0, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.07 }}
                  style={{
                    padding: "16px 20px",
                    backgroundColor: "#11131B",
                    border: "1px solid rgba(255,157,66,0.08)",
                    borderLeft: `3px solid ${layer.color}`,
                    borderRadius: 2,
                  }}
                >
                  <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", color: layer.color, letterSpacing: "0.1em", marginBottom: 8 }}>
                    {layer.label.toUpperCase()}
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {layer.items.map((item) => (
                      <span
                        key={item}
                        style={{
                          fontFamily: "JetBrains Mono, monospace",
                          fontSize: "0.68rem",
                          color: "#A0A2AD",
                          backgroundColor: "rgba(255,255,255,0.03)",
                          padding: "2px 8px",
                          borderRadius: 2,
                          border: "1px solid rgba(255,255,255,0.05)",
                        }}
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Routes + model architecture */}
          <div>
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.2em", textTransform: "uppercase", color: "#A0A2AD", marginBottom: 20 }}>
              API Routes
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2, marginBottom: 32 }}>
              {ROUTES.map((r, i) => (
                <motion.div
                  key={r.path}
                  initial={{ x: 20, opacity: 0 }}
                  whileInView={{ x: 0, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.06 }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "10px 16px",
                    backgroundColor: "#11131B",
                    border: "1px solid rgba(255,157,66,0.08)",
                    borderRadius: 2,
                  }}
                >
                  <span
                    style={{
                      fontFamily: "JetBrains Mono, monospace",
                      fontSize: "0.6rem",
                      fontWeight: 700,
                      color: r.color,
                      minWidth: 36,
                      letterSpacing: "0.05em",
                    }}
                  >
                    {r.method}
                  </span>
                  <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.68rem", color: "#A0A2AD" }}>
                    {r.path}
                  </span>
                </motion.div>
              ))}
            </div>

            {/* Model architecture boxes */}
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.2em", textTransform: "uppercase", color: "#A0A2AD", marginBottom: 16 }}>
              Model Architecture
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                {
                  name: "FilamentBasicCNN",
                  desc: "4× conv blocks → GAP → [presence sigmoid] + [5-class softmax]",
                  color: "#FF9D42",
                  latency: "8.56 ms",
                },
                {
                  name: "SolarFilamentUNet",
                  desc: "4× encoder down + 4× decoder up + skip connections → binary mask",
                  color: "#9585E8",
                  latency: "64.42 ms",
                },
              ].map((m) => (
                <motion.div
                  key={m.name}
                  initial={{ y: 10, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  viewport={{ once: true }}
                  style={{
                    padding: "16px",
                    backgroundColor: "#11131B",
                    border: `1px solid ${m.color}30`,
                    borderRadius: 4,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.72rem", color: m.color, fontWeight: 700 }}>
                      {m.name}
                    </span>
                    <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", color: "#A0A2AD" }}>
                      {m.latency} @ batch=1
                    </span>
                  </div>
                  <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.78rem", color: "#A0A2AD" }}>
                    {m.desc}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          #architecture > div > div:last-child {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </section>
  );
}
