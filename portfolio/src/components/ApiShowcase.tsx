"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { Copy, Check } from "lucide-react";
import { API_ENDPOINTS } from "@/lib/constants";

const METHOD_COLORS: Record<string, string> = {
  GET: "#4FC3F7",
  POST: "#FF9D42",
};

export default function ApiShowcase() {
  const [active, setActive] = useState(0);
  const [copied, setCopied] = useState(false);

  const endpoint = API_ENDPOINTS[active];

  const curlSnippet = endpoint.method === "GET"
    ? `curl -X GET http://localhost:8000${endpoint.path}`
    : `curl -X POST http://localhost:8000${endpoint.path} \\
  -F "image=@/path/to/solar_image.jpeg"`;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(curlSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section
      id="api"
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
            06 — API Documentation
          </span>
          <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: "clamp(1.8rem, 4vw, 3rem)", color: "#F3F2F0", margin: "12px 0 0", letterSpacing: "-0.02em" }}>
            REST API{" "}
            <span style={{ color: "#FF9D42" }}>Reference</span>
          </h2>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.9rem", color: "#A0A2AD", marginTop: 10, lineHeight: 1.7 }}>
            9 endpoints served by FastAPI at <code style={{ color: "#FF9D42", backgroundColor: "rgba(255,157,66,0.1)", padding: "2px 6px", borderRadius: 3, fontFamily: "JetBrains Mono, monospace", fontSize: "0.8rem" }}>http://localhost:8000</code>
          </p>
        </motion.div>

        <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 24 }}>
          {/* Endpoint list */}
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {API_ENDPOINTS.map((ep, i) => (
              <button
                key={i}
                onClick={() => setActive(i)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 14px",
                  backgroundColor: active === i ? "rgba(255,157,66,0.1)" : "#11131B",
                  border: active === i ? "1px solid rgba(255,157,66,0.3)" : "1px solid rgba(255,157,66,0.06)",
                  borderRadius: 3,
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "all 0.2s",
                }}
              >
                <span
                  style={{
                    fontFamily: "JetBrains Mono, monospace",
                    fontSize: "0.58rem",
                    fontWeight: 700,
                    color: METHOD_COLORS[ep.method],
                    minWidth: 30,
                    letterSpacing: "0.05em",
                  }}
                >
                  {ep.method}
                </span>
                <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.62rem", color: active === i ? "#F3F2F0" : "#A0A2AD", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {ep.path}
                </span>
              </button>
            ))}
          </div>

          {/* Endpoint detail */}
          <motion.div
            key={active}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.25 }}
          >
            {/* Method + path */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "14px 20px",
                backgroundColor: "#11131B",
                border: "1px solid rgba(255,157,66,0.12)",
                borderRadius: "4px 4px 0 0",
              }}
            >
              <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.72rem", fontWeight: 700, color: METHOD_COLORS[endpoint.method] }}>
                {endpoint.method}
              </span>
              <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.78rem", color: "#F3F2F0" }}>
                {endpoint.path}
              </span>
            </div>

            {/* Description */}
            <div
              style={{
                padding: "16px 20px",
                backgroundColor: "#11131B",
                borderLeft: "1px solid rgba(255,157,66,0.12)",
                borderRight: "1px solid rgba(255,157,66,0.12)",
              }}
            >
              <p style={{ fontFamily: "Inter, sans-serif", fontSize: "0.85rem", color: "#A0A2AD", lineHeight: 1.65, margin: 0 }}>
                {endpoint.desc}
              </p>
            </div>

            {/* Request body */}
            {endpoint.body && (
              <div style={{ backgroundColor: "#0D0F16", borderLeft: "1px solid rgba(255,157,66,0.12)", borderRight: "1px solid rgba(255,157,66,0.12)", borderTop: "1px solid rgba(255,157,66,0.06)" }}>
                <div style={{ padding: "8px 20px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                  <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.58rem", letterSpacing: "0.15em", color: "#A0A2AD", textTransform: "uppercase" }}>Request Body</span>
                </div>
                <pre style={{ padding: "14px 20px", fontFamily: "JetBrains Mono, monospace", fontSize: "0.72rem", color: "#9585E8", margin: 0, overflowX: "auto" }}>
                  {endpoint.body}
                </pre>
              </div>
            )}

            {/* Response */}
            <div style={{ backgroundColor: "#0A0C12", borderLeft: "1px solid rgba(255,157,66,0.12)", borderRight: "1px solid rgba(255,157,66,0.12)", borderTop: "1px solid rgba(255,255,255,0.04)" }}>
              <div style={{ padding: "8px 20px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.58rem", letterSpacing: "0.15em", color: "#A0A2AD", textTransform: "uppercase" }}>Response</span>
              </div>
              <pre style={{ padding: "14px 20px", fontFamily: "JetBrains Mono, monospace", fontSize: "0.72rem", color: "#4FC3F7", margin: 0, overflowX: "auto" }}>
                {endpoint.response}
              </pre>
            </div>

            {/* cURL snippet */}
            <div
              style={{
                backgroundColor: "#08090D",
                border: "1px solid rgba(255,157,66,0.12)",
                borderTop: "1px solid rgba(255,255,255,0.04)",
                borderRadius: "0 0 4px 4px",
              }}
            >
              <div style={{ padding: "8px 20px 8px 20px", borderBottom: "1px solid rgba(255,255,255,0.04)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.58rem", letterSpacing: "0.15em", color: "#A0A2AD", textTransform: "uppercase" }}>cURL Example</span>
                <button
                  onClick={handleCopy}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    color: copied ? "#FF9D42" : "#A0A2AD",
                    fontFamily: "JetBrains Mono, monospace",
                    fontSize: "0.58rem",
                    letterSpacing: "0.1em",
                    transition: "color 0.2s",
                  }}
                >
                  {copied ? <Check size={12} /> : <Copy size={12} />}
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
              <pre style={{ padding: "14px 20px", fontFamily: "JetBrains Mono, monospace", fontSize: "0.72rem", color: "#FF9D42", margin: 0, overflowX: "auto" }}>
                {curlSnippet}
              </pre>
            </div>
          </motion.div>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          #api > div > div:last-child {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </section>
  );
}
