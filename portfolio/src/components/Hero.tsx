"use client";
import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ArrowDown, ExternalLink } from "lucide-react";

const STAGGER = {
  container: { hidden: {}, visible: { transition: { staggerChildren: 0.12, delayChildren: 0.3 } } },
  item: { hidden: { y: 30, opacity: 0 }, visible: { y: 0, opacity: 1, transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] } } },
};

export default function Hero() {
  const imgRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onScroll = () => {
      if (imgRef.current) {
        const y = window.scrollY * 0.3;
        imgRef.current.style.transform = `translateY(${y}px)`;
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <section
      id="overview"
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        overflow: "hidden",
        backgroundColor: "#08090D",
      }}
    >
      {/* Grid overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(255,157,66,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,157,66,0.025) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
          pointerEvents: "none",
        }}
      />

      {/* Solar image — right-side, parallax */}
      <div
        ref={imgRef}
        style={{
          position: "absolute",
          right: "-5%",
          top: "5%",
          width: "55%",
          height: "90%",
          maxWidth: 720,
          borderRadius: 2,
          overflow: "hidden",
          opacity: 0.55,
          willChange: "transform",
        }}
      >
        {/* Gradient mask over left edge */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 2,
            background:
              "linear-gradient(to right, #08090D 0%, rgba(8,9,13,0.7) 30%, transparent 70%)",
          }}
        />
        {/* Vignette bottom */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 2,
            background:
              "linear-gradient(to top, #08090D 0%, transparent 40%)",
          }}
        />
        {/* Scanlines */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 3,
            backgroundImage:
              "repeating-linear-gradient(to bottom, transparent 0px, transparent 2px, rgba(0,0,0,0.1) 2px, rgba(0,0,0,0.1) 4px)",
            pointerEvents: "none",
          }}
        />
        <img
          src="/images/gong_solar.jpeg"
          alt="GONG H-alpha solar observation"
          style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
        />
      </div>

      {/* Corner coordinate markers */}
      <div style={{ position: "absolute", top: 80, left: 24, fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", color: "rgba(255,157,66,0.3)", letterSpacing: "0.1em" }}>
        RA 00h 00m · DEC +00° 00′
      </div>
      <div style={{ position: "absolute", bottom: 24, right: 24, fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", color: "rgba(255,157,66,0.3)", letterSpacing: "0.1em" }}>
        GONG H-α · 2048×2048px
      </div>

      {/* Content */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px", width: "100%", position: "relative", zIndex: 10 }}>
        <motion.div
          variants={STAGGER.container}
          initial="hidden"
          animate="visible"
          style={{ maxWidth: 640 }}
        >
          {/* Eyebrow */}
          <motion.div variants={STAGGER.item}>
            <span
              style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: "0.65rem",
                letterSpacing: "0.25em",
                textTransform: "uppercase",
                color: "#FF9D42",
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  backgroundColor: "#FF9D42",
                  display: "inline-block",
                  animation: "pulse-dot 2s ease-in-out infinite",
                }}
              />
              Computer Vision · Solar Physics · MLOps
            </span>
          </motion.div>

          {/* Main headline */}
          <motion.h1
            variants={STAGGER.item}
            style={{
              fontFamily: "Inter, sans-serif",
              fontWeight: 900,
              fontSize: "clamp(2.8rem, 6vw, 5rem)",
              lineHeight: 1.0,
              color: "#F3F2F0",
              margin: "20px 0 0",
              letterSpacing: "-0.02em",
            }}
          >
            SEE THE STRUCTURES{" "}
            <span style={{ color: "#FF9D42" }}>HIDDEN</span>
            <br />
            IN THE SUN.
          </motion.h1>

          {/* Project title */}
          <motion.p
            variants={STAGGER.item}
            style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: "0.8rem",
              letterSpacing: "0.1em",
              color: "#A0A2AD",
              margin: "20px 0 0",
              textTransform: "uppercase",
            }}
          >
            Solar Filament ML Pipeline &amp; FastAPI Backend
          </motion.p>

          {/* Description */}
          <motion.p
            variants={STAGGER.item}
            style={{
              fontFamily: "Inter, sans-serif",
              fontSize: "1rem",
              lineHeight: 1.7,
              color: "#A0A2AD",
              margin: "16px 0 0",
              maxWidth: 520,
            }}
          >
            A production-oriented deep learning pipeline for solar filament detection and
            segmentation — combining scientific image processing, model experimentation,
            experiment tracking, and API-based inference.
          </motion.p>

          {/* CTAs */}
          <motion.div
            variants={STAGGER.item}
            style={{ display: "flex", gap: 16, marginTop: 36, flexWrap: "wrap" }}
          >
            <button
              onClick={() => document.querySelector("#pipeline")?.scrollIntoView({ behavior: "smooth" })}
              style={{
                padding: "12px 28px",
                backgroundColor: "#FF9D42",
                color: "#08090D",
                border: "none",
                borderRadius: 4,
                fontFamily: "JetBrains Mono, monospace",
                fontSize: "0.75rem",
                fontWeight: 700,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <ArrowDown size={14} />
              Explore the Pipeline
            </button>
            <a
              href="https://github.com/"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                padding: "12px 28px",
                backgroundColor: "transparent",
                color: "#F3F2F0",
                border: "1px solid rgba(243,242,240,0.2)",
                borderRadius: 4,
                fontFamily: "JetBrains Mono, monospace",
                fontSize: "0.75rem",
                fontWeight: 500,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                cursor: "pointer",
                textDecoration: "none",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <ExternalLink size={14} />
              View on GitHub
            </a>
          </motion.div>

          {/* Tech strip */}
          <motion.div
            variants={STAGGER.item}
            style={{
              marginTop: 48,
              paddingTop: 24,
              borderTop: "1px solid rgba(255,157,66,0.12)",
              display: "flex",
              gap: 28,
              flexWrap: "wrap",
            }}
          >
            {[
              { label: "Framework", value: "PyTorch 2.14 + FastAPI" },
              { label: "Data Source", value: "GONG H-α" },
              { label: "Status", value: "In Development" },
              { label: "Dataset", value: "MAGFiLO 1.0" },
            ].map((item) => (
              <div key={item.label}>
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.58rem", letterSpacing: "0.2em", textTransform: "uppercase", color: "rgba(160,162,173,0.6)", marginBottom: 4 }}>
                  {item.label}
                </div>
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.72rem", color: "#F3F2F0" }}>
                  {item.value}
                </div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
