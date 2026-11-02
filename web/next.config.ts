import type { NextConfig } from "next";

// Multithreaded WebAssembly needs cross-origin isolation. `credentialless` still allows map tiles
// and species photos from other origins. Browsers that ignore it simply run on a single thread.
const isolation = [
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
];

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
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
    ];
  },
};

export default nextConfig;
