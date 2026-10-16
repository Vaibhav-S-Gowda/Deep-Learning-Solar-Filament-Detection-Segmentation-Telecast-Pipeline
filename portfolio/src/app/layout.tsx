import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Solar Filament ML Pipeline",
  description:
    "Production-ready deep learning pipeline for solar filament detection and segmentation using PyTorch, FastAPI, MLflow, and Docker.",
  keywords: ["solar filament", "deep learning", "PyTorch", "FastAPI", "MLflow", "GONG", "space weather"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body style={{ backgroundColor: "#08090D", color: "#F3F2F0" }}>
        {children}
      </body>
    </html>
  );
}
