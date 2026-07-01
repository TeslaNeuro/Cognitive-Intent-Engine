import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import type { IncomingMessage } from "http";
import type { Socket } from "net";

/** Suppress benign EPIPE errors when a WS client disconnects mid-proxy. */
function silenceProxyEpipe(proxy: {
  on: (event: string, listener: (...args: unknown[]) => void) => void;
}) {
  const ignoreEpipe = (err: unknown) => {
    if (err && typeof err === "object" && "code" in err && err.code === "EPIPE") {
      return;
    }
    console.error("[vite] ws proxy error:", err);
  };
  proxy.on("error", ignoreEpipe);
  proxy.on("proxyReqWs", (_proxyReq: unknown, _req: IncomingMessage, socket: Socket) => {
    socket.on("error", ignoreEpipe);
  });
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
        configure: silenceProxyEpipe,
      },
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
