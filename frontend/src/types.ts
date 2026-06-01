// Mirrors backend/utils/schemas.py FusedFrame.

export interface Event {
  type: string;
  severity: number;
  ts: number;
  detail?: string | null;
}

export interface FusedFrame {
  ts: number;

  emotion: string;
  confidence: number;
  probs: Record<string, number>;
  source_weights: Record<string, number>;

  cognitive_state: string;
  intent: string;
  attention: string;
  trend: string;

  stress: number;
  engagement: number;
  fatigue: number;
  cognitive_load: number;

  events: Event[];
  anomaly_score: number;
  explanation: string[];
  adaptive_action: string | null;
  calibration: { samples?: number; ready?: boolean };
  latency_ms: number;
  has_voice: boolean;
  face_detected: boolean;
}

export const EMOTION_LABELS = [
  "happy",
  "sad",
  "angry",
  "neutral",
  "frustrated",
] as const;

export const EMOTION_COLORS: Record<string, string> = {
  happy: "#22c55e",
  sad: "#3b82f6",
  angry: "#ef4444",
  neutral: "#94a3b8",
  frustrated: "#a855f7",
};
