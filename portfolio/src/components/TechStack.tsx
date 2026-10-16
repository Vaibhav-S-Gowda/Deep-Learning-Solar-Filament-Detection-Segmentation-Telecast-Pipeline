"use client";
import { motion } from "framer-motion";
import { TECH_STACK } from "@/lib/constants";

export default function TechStack() {
  return (
    <section
      style={{
        backgroundColor: "#08090D",
        padding: "100px 24px",
        borderTop: "1px solid rgba(255,157,66,0.08)",
      }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          style={{ marginBottom: 48 }}
        >
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.6rem", letterSpacing: "0.25em", textTransform: "uppercase", color: "#FF9D42" }}>
            07 — Tools &amp; Technologies
          </span>
          <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: "clamp(1.8rem, 4vw, 3rem)", color: "#F3F2F0", margin: "12px 0 0", letterSpacing: "-0.02em" }}>
            Built With{" "}
            <span style={{ color: "#FF9D42" }}>Production Tools</span>
          </h2>
        </motion.div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: 12 }}>
          {TECH_STACK.map((tech, i) => (
            <motion.div
              key={tech.name}
              initial={{ y: 20, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.05 }}
              whileHover={{ borderColor: "rgba(255,157,66,0.3)", backgroundColor: "rgba(255,157,66,0.03)" }}
              style={{
                padding: "20px",
                backgroundColor: "#11131B",
                border: "1px solid rgba(255,157,66,0.1)",
                borderRadius: 4,
                transition: "all 0.2s",
                cursor: "default",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: "1.4rem" }}>{tech.icon}</span>
                  <div>
                    <div style={{ fontFamily: "Inter, sans-serif", fontWeight: 700, fontSize: "0.9rem", color: "#F3F2F0" }}>
                      {tech.name}
                    </div>
                    <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.58rem", color: "#FF9D42", letterSpacing: "0.05em" }}>
                      v{tech.version}
                    </div>
                  </div>
                </div>
                <span
                  style={{
                    fontFamily: "JetBrains Mono, monospace",
                    fontSize: "0.55rem",
                    letterSpacing: "0.08em",
                    color: "#A0A2AD",
                    backgroundColor: "rgba(160,162,173,0.08)",
                    padding: "2px 7px",
                    borderRadius: 2,
                    textTransform: "uppercase",
                    whiteSpace: "nowrap",
                  }}
                >
                  {tech.category}
                </span>
              </div>
              <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.78rem", color: "#A0A2AD", lineHeight: 1.5 }}>
                {tech.desc}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
