import { useEffect, useState } from "react";
import type { FusedFrame } from "../types";
import { EMOTION_COLORS } from "../types";

interface Props {
  frame: FusedFrame | null;
}

/**
 * Live MJPEG view of the annotated webcam, plus an emotion badge overlay.
 * Falls back to a single-frame poll if the MJPEG stream is unavailable.
 */
export function VideoPanel({ frame }: Props) {
  const [mjpegOk, setMjpegOk] = useState(true);
  const [fallbackUrl, setFallbackUrl] = useState("/frame.jpg?ts=0");

  useEffect(() => {
    if (mjpegOk) return;
    const id = setInterval(() => setFallbackUrl(`/frame.jpg?ts=${Date.now()}`), 100);
    return () => clearInterval(id);
  }, [mjpegOk]);

  const color = frame ? EMOTION_COLORS[frame.emotion] ?? "#fff" : "#888";

  return (
    <div className="panel video-panel">
      <div className="panel-header">
        <span>Camera</span>
        <span className="muted">
          {frame?.face_detected ? "face ok" : "no face"} ·{" "}
          {frame?.has_voice ? "voice ok" : "silence"}
        </span>
      </div>
      <div className="video-wrap">
        {mjpegOk ? (
          <img
            src="/video.mjpg"
            alt="webcam"
            onError={() => setMjpegOk(false)}
          />
        ) : (
          <img src={fallbackUrl} alt="webcam" />
        )}
        {frame && (
          <div className="badge-row">
            <div className="badge" style={{ borderColor: color, color }}>
              {frame.emotion}
              <span className="badge-pct">{Math.round(frame.confidence * 100)}%</span>
            </div>
            <div className="badge subtle">
              {frame.cognitive_state}
            </div>
            <div className="badge subtle">{frame.intent}</div>
          </div>
        )}
      </div>
    </div>
  );
}
