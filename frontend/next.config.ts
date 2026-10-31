import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // Proxy all /api/** requests to the FastAPI backend.
  // This avoids CORS issues and lets the frontend use relative URLs in
  // production while keeping the FastAPI URL only on the server side.
  async rewrites() {
    const backendUrl =
      process.env.BACKEND_INTERNAL_URL?.replace(/\/+$/, "") ||
      "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
