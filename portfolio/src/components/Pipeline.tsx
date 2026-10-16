"use client";
import { motion } from "framer-motion";
import { PIPELINE_STAGES } from "@/lib/constants";

export default function Pipeline() {
  return (
    <section
      id="pipeline"
      style={{
        backgroundColor: "#08090D",
        padding: "100px 24px",
        borderTop: "1px solid rgba(255,157,66,0.08)",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Background grid */}
      <div style={{ position: "absolute", inset: 0, backgroundImage: "linear-gradient(rgba(255,157,66,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,157,66,0.02) 1px, transparent 1px)", backgroundSize: "60px 60px", pointerEvents: "none" }} />

      <div style={{ maxWidth: 1200, margin: "0 auto", position: "relative" }}>
        {/* Header */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          style={{ marginBottom: 64 }}
        >
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.25em", textTransform: "uppercase", color: "#FF9D42" }}>
            03 — ML Pipeline
          </span>
          <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: "clamp(1.8rem, 4vw, 3rem)", color: "#F3F2F0", margin: "12px 0 0", letterSpacing: "-0.02em" }}>
            From Observation to{" "}
            <span style={{ color: "#FF9D42" }}>Inference</span>
          </h2>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.95rem", color: "#A0A2AD", marginTop: 12, maxWidth: 600, lineHeight: 1.7 }}>
            A two-stage deep learning pipeline processes raw solar images through detection,
            segmentation, post-processing and API delivery — all tracked with MLflow.
          </p>
        </motion.div>

        {/* Horizontal scroll pipeline on mobile, grid on desktop */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 1,
            backgroundColor: "rgba(255,157,66,0.06)",
            border: "1px solid rgba(255,157,66,0.1)",
            borderRadius: 4,
          }}
        >
          {PIPELINE_STAGES.map((stage, i) => (
            <motion.div
              key={stage.id}
              initial={{ y: 30, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              whileHover={{ backgroundColor: "rgba(255,157,66,0.04)" }}
              style={{
                padding: "28px 24px",
                backgroundColor: "#08090D",
                position: "relative",
                cursor: "default",
                transition: "background-color 0.2s",
              }}
            >
              {/* Stage number */}
              <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.2em", color: "rgba(160,162,173,0.4)", marginBottom: 16 }}>
                STAGE {stage.id}
              </div>

              {/* Icon */}
              <div style={{ fontSize: "1.8rem", marginBottom: 12 }}>{stage.icon}</div>

              {/* Title */}
              <div style={{ fontFamily: "Inter, sans-serif", fontWeight: 700, fontSize: "1rem", color: "#F3F2F0", marginBottom: 4 }}>
                {stage.title}
              </div>

              {/* Subtitle */}
              <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem", letterSpacing: "0.08em", color: stage.color, marginBottom: 12 }}>
                {stage.subtitle}
              </div>

              {/* Description */}
              <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.82rem", color: "#A0A2AD", lineHeight: 1.65 }}>
                {stage.desc}
              </div>

              {/* Bottom accent bar */}
              <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 2, background: `linear-gradient(to right, ${stage.color}60, transparent)` }} />
            </motion.div>
          ))}
        </div>

        {/* Flow summary bar */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          style={{
            marginTop: 32,
            padding: "16px 24px",
            backgroundColor: "#11131B",
            border: "1px solid rgba(255,157,66,0.1)",
            borderRadius: 4,
            display: "flex",
            alignItems: "center",
            gap: 16,
            flexWrap: "wrap",
          }}
        >
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.2em", textTransform: "uppercase", color: "#A0A2AD" }}>
            End-to-end latency:
          </span>
          {[
            { label: "CNN Stage", value: "8.56 ms", color: "#FF9D42" },
            { label: "UNet Stage", value: "64.42 ms", color: "#9585E8" },
            { label: "Telecast Frame", value: "38.4 ms", color: "#FF5C35" },
          ].map((m) => (
            <div key={m.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: m.color }} />
              <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.68rem", color: "#A0A2AD" }}>
                {m.label}:{" "}
                <span style={{ color: m.color, fontWeight: 700 }}>{m.value}</span>
              </span>
            </div>
          ))}
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.58rem", color: "rgba(160,162,173,0.4)", marginLeft: "auto" }}>
            CPU only · Windows 11 · No GPU
          </span>
        </motion.div>
      </div>
    </section>
  );
}
