import { useEffect, useRef, useState } from "react";
import type { FusedFrame } from "../types";

type Status = "connecting" | "open" | "closed" | "error";

/**
 * Connects to /ws and exposes the latest frame + a rolling history.
 * - Reconnects automatically with exponential backoff.
 * - History is *flushed at animation-frame cadence* to avoid blowing up
 *   React/Recharts when frames arrive at 10+ Hz.
 * - A fresh array reference is published on every flush so memoized chart
 *   components correctly detect changes.
 */
export function useFusedFrameStream(historySize = 600): {
  status: Status;
  latest: FusedFrame | null;
  history: FusedFrame[];
} {
  const [status, setStatus] = useState<Status>("connecting");
  const [latest, setLatest] = useState<FusedFrame | null>(null);
  const [history, setHistory] = useState<FusedFrame[]>([]);

  // We accumulate incoming frames in a ref and flush them at most once per
  // animation frame to throttle React updates.
  const pendingRef = useRef<FusedFrame[]>([]);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retryDelay = 500;
    let retryTimer: number | null = null;
    let closed = false;

    const flush = () => {
      rafRef.current = null;
      const pending = pendingRef.current;
      if (pending.length === 0) return;
      pendingRef.current = [];
      setHistory((prev) => {
        const next = prev.concat(pending);
        if (next.length > historySize) next.splice(0, next.length - historySize);
        return next;
      });
    };

    const connect = () => {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      // In dev, connect directly to the backend to avoid Vite proxy EPIPE noise
      // when React StrictMode mounts/unmounts quickly.
      const url = import.meta.env.DEV
        ? `ws://localhost:8000/ws`
        : `${proto}//${window.location.host}/ws`;
      setStatus("connecting");
      ws = new WebSocket(url);

      ws.onopen = () => {
        retryDelay = 500;
        setStatus("open");
      };

      ws.onmessage = (ev) => {
        try {
          const frame = JSON.parse(ev.data) as FusedFrame;
          setLatest(frame);
          pendingRef.current.push(frame);
          if (rafRef.current === null) {
            rafRef.current = requestAnimationFrame(flush);
          }
        } catch {
          /* ignore malformed frames */
        }
      };

      ws.onerror = () => setStatus("error");
      ws.onclose = () => {
        setStatus("closed");
        if (closed) return;
        retryTimer = window.setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 5000);
      };
    };

    connect();

    return () => {
      closed = true;
      if (retryTimer) window.clearTimeout(retryTimer);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      if (ws && ws.readyState <= 1) ws.close();
    };
  }, [historySize]);

  return { status, latest, history };
}
