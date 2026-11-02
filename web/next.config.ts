import type { NextConfig } from "next";

// Multithreaded WebAssembly needs cross-origin isolation. `credentialless` still allows map tiles
// and species photos from other origins. Browsers that ignore it simply run on a single thread.
const isolation = [
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
];

const nextConfig: NextConfig = {
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  images: {
    // Vercel's free tier limits image transformations. Sources already provide suitable sizes.
    unoptimized: true,
  },
  async headers() {
    return [
      { source: "/identify", headers: isolation },
      { source: "/camera-trap", headers: isolation },
      // Worker scripts must carry these too, or an isolated page's browser refuses to start them.
      {
        source: "/_next/static/:path*",
        headers: [
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
          { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
