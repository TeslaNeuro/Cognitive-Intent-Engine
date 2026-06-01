import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": { target: "ws://localhost:8000", ws: true, changeOrigin: true },
      "/video.mjpg": { target: "http://localhost:8000", changeOrigin: true },
      "/frame.jpg":  { target: "http://localhost:8000", changeOrigin: true },
      "/baseline":   { target: "http://localhost:8000", changeOrigin: true },
      "/events":     { target: "http://localhost:8000", changeOrigin: true },
      "/config":     { target: "http://localhost:8000", changeOrigin: true },
      "/latest":     { target: "http://localhost:8000", changeOrigin: true },
      "/healthz":    { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
