import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  // JSX is compiled by Vite's built-in transform, so no React plugin is needed for unit tests.
  oxc: { jsx: { runtime: "automatic" } },
  resolve: {
    alias: { "@": fileURLToPath(new URL("./", import.meta.url)) },
  },
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.test.{ts,tsx}"],
  },
});
