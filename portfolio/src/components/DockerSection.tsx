"use client";
import { motion } from "framer-motion";

const COMMANDS = [
  { comment: "# Clone the repository", cmd: "" },
  { comment: "", cmd: "git clone https://github.com/your-repo/solar-filament.git" },
  { comment: "", cmd: "cd solar-filament" },
  { comment: "", cmd: "" },
  { comment: "# Build and run with Docker Compose", cmd: "" },
  { comment: "", cmd: "docker-compose up --build" },
  { comment: "", cmd: "" },
  { comment: "# API is available at", cmd: "" },
  { comment: "", cmd: "# http://localhost:8000  ← FastAPI" },
  { comment: "", cmd: "# http://localhost:5000  ← MLflow UI" },
];

const DOCKER_SERVICES = [
  {
    name: "solar-api",
    image: "python:3.11-slim",
    port: "8000:8000",
    desc: "FastAPI inference server with both ML models pre-loaded",
    color: "#FF9D42",
  },
  {
    name: "mlflow-ui",
    image: "python:3.11-slim",
    port: "5000:5000",
    desc: "MLflow tracking server with SQLite backend for experiment viewing",
    color: "#9585E8",
  },
];

export default function DockerSection() {
  return (
    <section
      id="docker"
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
            08 — Deployment
          </span>
          <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: "clamp(1.8rem, 4vw, 3rem)", color: "#F3F2F0", margin: "12px 0 0", letterSpacing: "-0.02em" }}>
            Containerised with{" "}
            <span style={{ color: "#FF9D42" }}>Docker</span>
          </h2>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.9rem", color: "#A0A2AD", marginTop: 10, lineHeight: 1.7 }}>
            The entire pipeline — API, models, and MLflow — runs with a single{" "}
            <code style={{ fontFamily: "JetBrains Mono, monospace", color: "#FF9D42", backgroundColor: "rgba(255,157,66,0.1)", padding: "2px 5px", borderRadius: 3 }}>
              docker-compose up
            </code>{" "}
            command for reproducible deployment.
          </p>
        </motion.div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          {/* Terminal */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            viewport={{ once: true }}
            style={{
              backgroundColor: "#0A0C12",
              border: "1px solid rgba(255,157,66,0.12)",
              borderRadius: 6,
              overflow: "hidden",
              fontFamily: "JetBrains Mono, monospace",
            }}
          >
            {/* Terminal title bar */}
            <div style={{ padding: "10px 16px", backgroundColor: "#11131B", borderBottom: "1px solid rgba(255,157,66,0.1)", display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#FF5C35" }} />
              <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#FF9D42" }} />
              <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#4FC3F7" }} />
              <span style={{ fontSize: "0.6rem", color: "#A0A2AD", marginLeft: 8, letterSpacing: "0.1em" }}>bash</span>
            </div>

            {/* Terminal body */}
            <div style={{ padding: "20px 20px", lineHeight: 2 }}>
              {COMMANDS.map((line, i) => (
                <div key={i}>
                  {line.comment && (
                    <div style={{ fontSize: "0.68rem", color: "rgba(160,162,173,0.5)" }}>
                      {line.comment}
                    </div>
                  )}
                  {line.cmd && (
                    <div style={{ fontSize: "0.72rem", color: "#F3F2F0" }}>
                      <span style={{ color: "#FF9D42" }}>$ </span>
                      {line.cmd}
                    </div>
                  )}
                  {!line.comment && !line.cmd && <div style={{ height: 8 }} />}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Services */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {DOCKER_SERVICES.map((svc, i) => (
              <motion.div
                key={svc.name}
                initial={{ x: 20, opacity: 0 }}
                whileInView={{ x: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                style={{
                  padding: "24px",
                  backgroundColor: "#11131B",
                  border: `1px solid ${svc.color}25`,
                  borderLeft: `3px solid ${svc.color}`,
                  borderRadius: 4,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                  <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.75rem", color: svc.color, fontWeight: 700 }}>
                    {svc.name}
                  </span>
                  <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", color: "#A0A2AD", backgroundColor: "rgba(255,255,255,0.04)", padding: "2px 8px", borderRadius: 2, border: "1px solid rgba(255,255,255,0.06)" }}>
                    :{svc.port.split(":")[0]}
                  </span>
                </div>
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.62rem", color: "rgba(160,162,173,0.5)", marginBottom: 8 }}>
                  {svc.image}
                </div>
                <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.82rem", color: "#A0A2AD", lineHeight: 1.6 }}>
                  {svc.desc}
                </div>
              </motion.div>
            ))}

            {/* Docker info cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {[
                { label: "Base Image", value: "python:3.11-slim" },
                { label: "Health Check", value: "GET /health" },
                { label: "Workers", value: "2 (Uvicorn)" },
                { label: "Libs", value: "libgl1, libglib2" },
              ].map((item) => (
                <div
                  key={item.label}
                  style={{
                    padding: "12px 14px",
                    backgroundColor: "#0D0F16",
                    border: "1px solid rgba(255,255,255,0.05)",
                    borderRadius: 3,
                  }}
                >
                  <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.55rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(160,162,173,0.5)", marginBottom: 4 }}>
                    {item.label}
                  </div>
                  <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.7rem", color: "#F3F2F0" }}>
                    {item.value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          #docker > div > div:last-child {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </section>
  );
}
