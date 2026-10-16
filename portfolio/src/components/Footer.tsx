"use client";
import { motion } from "framer-motion";
import { Github, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer
      style={{
        backgroundColor: "#08090D",
        borderTop: "1px solid rgba(255,157,66,0.1)",
        padding: "60px 24px 40px",
      }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 32, marginBottom: 40 }}>
          {/* Identity */}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
              <div style={{ width: 28, height: 28, borderRadius: "50%", background: "radial-gradient(circle at 40% 40%, #FF9D42, #FF5C35 60%, #08090D)", border: "1px solid rgba(255,157,66,0.4)" }} />
              <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.7rem", letterSpacing: "0.2em", color: "#A0A2AD" }}>
                SOLAR / ML SYSTEMS
              </span>
            </div>
            <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.82rem", color: "#A0A2AD", maxWidth: 340, lineHeight: 1.7, margin: 0 }}>
              Solar Filament ML Pipeline — a research and engineering project combining
              deep learning, solar physics observations, and production ML infrastructure.
            </p>
          </div>

          {/* Links */}
          <div style={{ display: "flex", gap: 48, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.2em", textTransform: "uppercase", color: "rgba(160,162,173,0.5)", marginBottom: 16 }}>
                Navigation
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {["Overview", "Pipeline", "Architecture", "Results", "API", "Docker"].map((item) => (
                  <button
                    key={item}
                    onClick={() => document.querySelector(`#${item.toLowerCase()}`)?.scrollIntoView({ behavior: "smooth" })}
                    style={{
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      fontFamily: "JetBrains Mono, monospace",
                      fontSize: "0.72rem",
                      color: "#A0A2AD",
                      textAlign: "left",
                      padding: 0,
                      transition: "color 0.2s",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "#FF9D42")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "#A0A2AD")}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.2em", textTransform: "uppercase", color: "rgba(160,162,173,0.5)", marginBottom: 16 }}>
                Resources
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {[
                  { label: "GitHub", href: "https://github.com/" },
                  { label: "FastAPI Docs", href: "http://localhost:8000/docs" },
                  { label: "MLflow UI", href: "http://localhost:5000" },
                  { label: "GONG / NSO", href: "https://gong.nso.edu" },
                  { label: "MAGFiLO Dataset", href: "https://www.kaggle.com/" },
                ].map((link) => (
                  <a
                    key={link.label}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      fontFamily: "JetBrains Mono, monospace",
                      fontSize: "0.72rem",
                      color: "#A0A2AD",
                      textDecoration: "none",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      transition: "color 0.2s",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "#FF9D42")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "#A0A2AD")}
                  >
                    <ExternalLink size={10} />
                    {link.label}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div
          style={{
            borderTop: "1px solid rgba(255,157,66,0.08)",
            paddingTop: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.62rem", color: "rgba(160,162,173,0.4)" }}>
            PyTorch 2.14.1 · FastAPI · MLflow 3.17.1 · Docker · MAGFiLO 1.0 — 1,154 images · 8,199 annotations
          </span>
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.62rem", color: "rgba(160,162,173,0.3)" }}>
            SOLAR / ML SYSTEMS — In Development
          </span>
        </div>
      </div>
    </footer>
  );
}
