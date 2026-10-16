"use client";
import { useState, useRef } from "react";
import { motion } from "framer-motion";

type Mode = "original" | "mask" | "overlay";

export default function ImageShowcase() {
  const [mode, setMode] = useState<Mode>("original");
  const [sliderX, setSliderX] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const IMAGE_MAP: Record<Mode, { src: string; label: string; desc: string }> = {
    original: {
      src: "/images/gong_solar.jpeg",
      label: "Original H-α Observation",
      desc: "Raw 2048×2048 GONG H-alpha solar disc image — 20110109 104734 UT",
    },
    mask: {
      src: "/images/mask.png",
      label: "Segmentation Mask",
      desc: "Binary pixel mask output from SolarFilamentUNet — white pixels indicate detected filament regions",
    },
    overlay: {
      src: "/images/overlay.jpg",
      label: "Detection Overlay",
      desc: "Filament polygons and bounding boxes drawn over the original observation",
    },
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100));
    setSliderX(pct);
  };

  const current = IMAGE_MAP[mode];

  return (
    <section
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
          style={{ marginBottom: 48 }}
        >
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.25em", textTransform: "uppercase", color: "#FF9D42" }}>
            02 — Image Analysis
          </span>
          <h2
            style={{
              fontFamily: "Inter, sans-serif",
              fontWeight: 800,
              fontSize: "clamp(1.8rem, 4vw, 3rem)",
              color: "#F3F2F0",
              margin: "12px 0 0",
              letterSpacing: "-0.02em",
            }}
          >
            From Observation to{" "}
            <span style={{ color: "#FF9D42" }}>Segmentation</span>
          </h2>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.95rem", color: "#A0A2AD", marginTop: 12, maxWidth: 580, lineHeight: 1.7 }}>
            Solar filaments are elongated structures of cool, dense plasma suspended in the Sun's corona.
            The model identifies their spatial extent from GONG H-α images.
          </p>
        </motion.div>

        {/* Toggle buttons */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
          {(["original", "mask", "overlay"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                padding: "8px 20px",
                fontFamily: "JetBrains Mono, monospace",
                fontSize: "0.68rem",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                cursor: "pointer",
                borderRadius: 3,
                border: mode === m ? "1px solid #FF9D42" : "1px solid rgba(255,157,66,0.2)",
                backgroundColor: mode === m ? "rgba(255,157,66,0.12)" : "transparent",
                color: mode === m ? "#FF9D42" : "#A0A2AD",
                transition: "all 0.2s",
              }}
            >
              {m === "original" ? "H-α Original" : m === "mask" ? "Mask Output" : "Overlay"}
            </button>
          ))}
        </div>

        {/* Image display */}
        <motion.div
          key={mode}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          ref={containerRef}
          onPointerMove={onPointerMove}
          onPointerUp={() => setDragging(false)}
          style={{
            position: "relative",
            width: "100%",
            borderRadius: 4,
            overflow: "hidden",
            border: "1px solid rgba(255,157,66,0.15)",
            backgroundColor: "#11131B",
            userSelect: "none",
          }}
        >
          {/* Overlay comparison for original vs overlay mode */}
          {mode === "overlay" ? (
            <div style={{ position: "relative" }}>
              {/* Left: original — clipped by slider */}
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: `${sliderX}%`,
                  height: "100%",
                  overflow: "hidden",
                  zIndex: 2,
                }}
              >
                <img
                  src="/images/gong_solar.jpeg"
                  alt="Original"
                  style={{ width: `${100 / (sliderX / 100)}%`, maxWidth: "none", height: "100%", objectFit: "cover", display: "block" }}
                />
              </div>

              {/* Right: overlay */}
              <img
                src="/images/overlay.jpg"
                alt="Overlay"
                style={{ width: "100%", display: "block", maxHeight: 600, objectFit: "contain" }}
              />

              {/* Slider handle */}
              <div
                onPointerDown={(e) => { setDragging(true); e.currentTarget.setPointerCapture(e.pointerId); }}
                onPointerUp={() => setDragging(false)}
                style={{
                  position: "absolute",
                  top: 0,
                  left: `${sliderX}%`,
                  transform: "translateX(-50%)",
                  height: "100%",
                  width: 2,
                  backgroundColor: "#FF9D42",
                  zIndex: 10,
                  cursor: "ew-resize",
                }}
              >
                <div style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%,-50%)",
                  width: 32,
                  height: 32,
                  borderRadius: "50%",
                  backgroundColor: "#FF9D42",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "0.7rem",
                  color: "#08090D",
                  fontWeight: 700,
                  boxShadow: "0 0 20px rgba(255,157,66,0.4)",
                }}>
                  ⟷
                </div>
              </div>

              {/* Labels */}
              <div style={{ position: "absolute", top: 12, left: 12, zIndex: 5, fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.1em", backgroundColor: "rgba(8,9,13,0.7)", padding: "4px 8px", color: "#A0A2AD" }}>
                ORIGINAL
              </div>
              <div style={{ position: "absolute", top: 12, right: 12, zIndex: 5, fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.1em", backgroundColor: "rgba(8,9,13,0.7)", padding: "4px 8px", color: "#FF9D42" }}>
                OVERLAY
              </div>
            </div>
          ) : (
            <img
              src={current.src}
              alt={current.label}
              style={{ width: "100%", maxHeight: 600, objectFit: "contain", display: "block" }}
            />
          )}

          {/* Scanline effect */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              backgroundImage: "repeating-linear-gradient(to bottom, transparent 0px, transparent 3px, rgba(0,0,0,0.06) 3px, rgba(0,0,0,0.06) 4px)",
              pointerEvents: "none",
              zIndex: 20,
            }}
          />

          {/* Bottom metadata strip */}
          <div style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            padding: "8px 16px",
            backgroundColor: "rgba(8,9,13,0.85)",
            borderTop: "1px solid rgba(255,157,66,0.1)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            zIndex: 15,
          }}>
            <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.62rem", color: "#A0A2AD" }}>
              {current.label}
            </span>
            <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.58rem", color: "rgba(160,162,173,0.5)", letterSpacing: "0.1em" }}>
              ILLUSTRATIVE — ACTUAL MODEL OUTPUT
            </span>
          </div>
        </motion.div>

        {/* Description */}
        <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", color: "#A0A2AD", marginTop: 16, lineHeight: 1.7 }}>
          {current.desc}
        </p>

        {/* Info cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16, marginTop: 40 }}>
          {[
            { title: "What is a Solar Filament?", body: "An elongated structure of relatively cool, dense plasma (≈8,000 K) suspended in the Sun's hotter corona (>1 MK) by magnetic field lines." },
            { title: "Why Segment Them?", body: "Filament morphology and evolution are key indicators in solar physics and space-weather research. Automated segmentation enables large-scale monitoring." },
            { title: "GONG H-α Imaging", body: "Global Oscillation Network Group provides near-continuous full-disc H-alpha images at 2048×2048 resolution — the input source for this pipeline." },
          ].map((card) => (
            <div
              key={card.title}
              style={{
                padding: "20px",
                backgroundColor: "#11131B",
                border: "1px solid rgba(255,157,66,0.1)",
                borderRadius: 4,
              }}
            >
              <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.7rem", color: "#FF9D42", marginBottom: 8 }}>
                {card.title}
              </div>
              <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.82rem", color: "#A0A2AD", lineHeight: 1.65 }}>
                {card.body}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
