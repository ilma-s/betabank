import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    // Hardcode the API URL for production
    NEXT_PUBLIC_API_URL: "http://ec2-35-179-171-39.eu-west-2.compute.amazonaws.com",
  },
  async headers() {
    return [
      {
        // Allow CORS for API routes
        source: "/api/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Access-Control-Allow-Methods", value: "GET,POST,PUT,DELETE,OPTIONS" },
          { key: "Access-Control-Allow-Headers", value: "Content-Type, Authorization" },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://ec2-35-179-171-39.eu-west-2.compute.amazonaws.com/:path*",
      },
    ];
  },
};

export default nextConfig;
