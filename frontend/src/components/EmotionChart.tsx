import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import type { FusedFrame } from "../types";
import { EMOTION_COLORS, EMOTION_LABELS } from "../types";

interface Props {
  history: FusedFrame[];
}

export function EmotionChart({ history }: Props) {
  if (history.length === 0) {
    return null;
  }
  const t0 = history[0].ts;
  const rows = history.map((f) => {
    const row: Record<string, number> = { t: +(f.ts - t0).toFixed(1) };
    for (const e of EMOTION_LABELS) {
      row[e] = f.probs?.[e] ?? 0;
    }
    return row;
  });

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Emotion probabilities</span>
        <span className="muted">5-class softmax</span>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
            <XAxis dataKey="t" stroke="#94a3b8" fontSize={11} />
            <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={11} />
            <Tooltip
              contentStyle={{
                background: "#0f172a",
                border: "1px solid #334155",
                borderRadius: 8,
                color: "#e2e8f0",
              }}
            />
            {EMOTION_LABELS.map((e) => (
              <Line
                key={e}
                type="monotone"
                dataKey={e}
                stroke={EMOTION_COLORS[e]}
                dot={false}
                isAnimationActive={false}
                strokeWidth={1.5}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        {EMOTION_LABELS.map((e) => (
          <span key={e} style={{ marginRight: 12 }}>
            <span
              className="dot"
              style={{ background: EMOTION_COLORS[e] }}
            />
            {e}
          </span>
        ))}
      </div>
    </div>
  );
}
