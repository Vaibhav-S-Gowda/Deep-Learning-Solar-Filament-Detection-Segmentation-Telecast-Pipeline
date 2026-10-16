"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, ExternalLink } from "lucide-react";

const NAV_LINKS = [
  { label: "Overview", href: "#overview" },
  { label: "Pipeline", href: "#pipeline" },
  { label: "Architecture", href: "#architecture" },
  { label: "Results", href: "#results" },
  { label: "API", href: "#api" },
  { label: "Docker", href: "#docker" },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [active, setActive] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive("#" + e.target.id);
        });
      },
      { rootMargin: "-40% 0px -55% 0px" }
    );
    document.querySelectorAll("section[id]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const scrollTo = (href: string) => {
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: "smooth" });
    setMenuOpen(false);
  };

  return (
    <motion.nav
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        backgroundColor: scrolled ? "rgba(8,9,13,0.92)" : "transparent",
        backdropFilter: scrolled ? "blur(12px)" : "none",
        borderBottom: scrolled ? "1px solid rgba(255,157,66,0.1)" : "none",
        transition: "all 0.3s ease",
      }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 64 }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 28, height: 28, borderRadius: "50%", background: "radial-gradient(circle at 40% 40%, #FF9D42, #FF5C35 60%, #08090D)", border: "1px solid rgba(255,157,66,0.4)" }} />
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "0.7rem", letterSpacing: "0.2em", color: "#A0A2AD" }}>
              SOLAR / ML SYSTEMS
            </span>
          </div>

          {/* Desktop Links */}
          <div style={{ display: "flex", gap: 32, alignItems: "center" }} className="desktop-nav">
            {NAV_LINKS.map((link) => (
              <button
                key={link.href}
                onClick={() => scrollTo(link.href)}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "0.72rem",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  color: active === link.href ? "#FF9D42" : "#A0A2AD",
                  transition: "color 0.2s",
                  padding: "4px 0",
                  borderBottom: active === link.href ? "1px solid #FF9D42" : "1px solid transparent",
                }}
              >
                {link.label}
              </button>
            ))}
            <a
              href="https://github.com/"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "6px 14px",
                border: "1px solid rgba(255,157,66,0.4)",
                borderRadius: 4,
                color: "#FF9D42",
                textDecoration: "none",
                fontSize: "0.72rem",
                fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: "0.1em",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "rgba(255,157,66,0.1)")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              <ExternalLink size={14} />
              GitHub
            </a>
          </div>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            style={{ background: "none", border: "none", cursor: "pointer", color: "#F3F2F0", display: "none" }}
            className="mobile-menu-btn"
          >
            {menuOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            style={{ backgroundColor: "#11131B", borderTop: "1px solid rgba(255,157,66,0.1)" }}
          >
            <div style={{ padding: "16px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
              {NAV_LINKS.map((link) => (
                <button
                  key={link.href}
                  onClick={() => scrollTo(link.href)}
                  style={{
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "0.8rem",
                    letterSpacing: "0.1em",
                    color: active === link.href ? "#FF9D42" : "#A0A2AD",
                    textAlign: "left",
                    padding: "8px 0",
                  }}
                >
                  {link.label}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        @media (max-width: 768px) {
          .desktop-nav { display: none !important; }
          .mobile-menu-btn { display: block !important; }
        }
      `}</style>
    </motion.nav>
  );
}
