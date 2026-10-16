"use client";
import { motion } from "framer-motion";
import { BENCHMARKS, METRICS, DATASET } from "@/lib/constants";

function MetricCard({ value, label, unit, color }: { value: string | number; label: string; unit?: string; color: string }) {
  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      style={{
        padding: "24px",
        backgroundColor: "#11131B",
        border: `1px solid ${color}25`,
        borderTop: `2px solid ${color}`,
        borderRadius: 4,
      }}
    >
      <div style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: "2rem", color, lineHeight: 1 }}>
        {value}
        {unit && <span style={{ fontSize: "0.9rem", fontWeight: 400, color: "#A0A2AD", marginLeft: 4 }}>{unit}</span>}
      </div>
      <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "#A0A2AD", marginTop: 8 }}>
        {label}
      </div>
    </motion.div>
  );
}

function Bar({ pct, color, label }: { pct: number; color: string; label: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", color: "#A0A2AD" }}>{label}</span>
        <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", color }}>
          {pct}%
        </span>
      </div>
      <div style={{ height: 4, backgroundColor: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${pct}%` }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ height: "100%", backgroundColor: color, borderRadius: 2 }}
        />
      </div>
    </div>
  );
}

export default function Results() {
  const cnnBenchmarks = BENCHMARKS.filter((b) => b.model === "FilamentBasicCNN");
  const unetBenchmarks = BENCHMARKS.filter((b) => b.model === "SolarFilamentUNet");

  return (
    <section
      id="results"
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
            05 — Results &amp; Benchmarks
          </span>
          <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: "clamp(1.8rem, 4vw, 3rem)", color: "#F3F2F0", margin: "12px 0 0", letterSpacing: "-0.02em" }}>
            Real Measured{" "}
            <span style={{ color: "#FF9D42" }}>Performance</span>
          </h2>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.9rem", color: "#A0A2AD", marginTop: 10, lineHeight: 1.7 }}>
            All metrics are from actual benchmark runs on CPU (Windows 11, no GPU). No invented numbers.
          </p>
        </motion.div>

        {/* Key metrics */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 48 }}>
          <MetricCard value="100%" label="Presence Accuracy" color="#FF9D42" />
          <MetricCard value="100%" label="Validation Recall" color="#FF9D42" />
          <MetricCard value="0.0077" label="Validation Dice Score" color="#9585E8" />
          <MetricCard value="8.56" unit="ms" label="CNN Latency (Batch 1)" color="#FF5C35" />
          <MetricCard value="116.9" unit="FPS" label="CNN Throughput (Batch 1)" color="#FF5C35" />
          <MetricCard value="38.4" unit="ms" label="Telecast Frame Latency" color="#6B8AFF" />
        </div>

        {/* Benchmark table */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{ once: true }}
          style={{
            backgroundColor: "#11131B",
            border: "1px solid rgba(255,157,66,0.1)",
            borderRadius: 4,
            overflow: "hidden",
            marginBottom: 48,
          }}
        >
          <div style={{ padding: "16px 20px", borderBottom: "1px solid rgba(255,157,66,0.1)", display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", letterSpacing: "0.15em", color: "#A0A2AD", textTransform: "uppercase" }}>
              Benchmark Results — CPU Inference
            </span>
            <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.55rem", color: "rgba(160,162,173,0.4)" }}>
              10 iterations per configuration
            </span>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "JetBrains Mono, monospace", fontSize: "0.72rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(255,157,66,0.1)" }}>
                  {["Model", "Batch", "Latency (ms)", "FPS", "RAM (MB)"].map((h) => (
                    <th key={h} style={{ padding: "10px 16px", textAlign: "left", color: "#A0A2AD", fontWeight: 500, letterSpacing: "0.08em" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {BENCHMARKS.map((row, i) => (
                  <tr
                    key={i}
                    style={{
                      borderBottom: "1px solid rgba(255,255,255,0.03)",
                      backgroundColor: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent",
                    }}
                  >
                    <td style={{ padding: "10px 16px", color: row.model === "FilamentBasicCNN" ? "#FF9D42" : "#9585E8" }}>
                      {row.model}
                    </td>
                    <td style={{ padding: "10px 16px", color: "#F3F2F0" }}>{row.batch}</td>
                    <td style={{ padding: "10px 16px", color: "#F3F2F0" }}>{row.latency.toFixed(2)}</td>
                    <td style={{ padding: "10px 16px", color: "#F3F2F0" }}>{row.fps.toFixed(1)}</td>
                    <td style={{ padding: "10px 16px", color: "#F3F2F0" }}>{row.ram.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Dataset distribution + Dice note */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          {/* Dataset distribution */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            viewport={{ once: true }}
            style={{ padding: "24px", backgroundColor: "#11131B", border: "1px solid rgba(255,157,66,0.1)", borderRadius: 4 }}
          >
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", color: "#A0A2AD", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 20 }}>
              MAGFiLO 1.0 — Category Distribution
            </div>
            {DATASET.categories.map((cat) => (
              <Bar key={cat.label} pct={cat.pct} color={cat.color} label={cat.label} />
            ))}
            <div style={{ marginTop: 16, fontFamily: "JetBrains Mono, monospace", fontSize: "0.62rem", color: "rgba(160,162,173,0.5)" }}>
              {DATASET.images.toLocaleString()} images · {DATASET.annotations.toLocaleString()} annotations · {DATASET.resolution}
            </div>
          </motion.div>

          {/* Training note */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            viewport={{ once: true }}
            style={{ padding: "24px", backgroundColor: "#11131B", border: "1px solid rgba(149,133,232,0.2)", borderRadius: 4 }}
          >
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", color: "#9585E8", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 16 }}>
              Training Status
            </div>
            <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", color: "#A0A2AD", lineHeight: 1.7, marginBottom: 16 }}>
              The Dice score of{" "}
              <span style={{ color: "#9585E8", fontWeight: 700 }}>0.0077</span> represents
              early-stage training (baseline checkpoint after 2 epochs). The model's
              DiceBCELoss handles extreme class imbalance where filament pixels represent
              less than 1% of each image.
            </div>
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", color: "rgba(160,162,173,0.5)" }}>
              MLflow run: unet_mlflow_baseline<br />
              Experiment: Solar-Filament-Segmentation<br />
              Backend: SQLite (mlflow.db)
            </div>
          </motion.div>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          #results > div > div:last-child {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </section>
  );
}
