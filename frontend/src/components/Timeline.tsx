import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Event, FusedFrame } from "../types";

interface Props {
  history: FusedFrame[];
}

interface ChartRow {
  t: number;
  stress: number;
  engagement: number;
  anomaly: number;
  events: Event[];
}

export function Timeline({ history }: Props) {
  if (history.length === 0) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>Timeline</span>
          <span className="muted">waiting for data…</span>
        </div>
        <div className="timeline-empty">No frames yet.</div>
      </div>
    );
  }

  const t0 = history[0].ts;
  const data: ChartRow[] = history.map((f) => ({
    t: +(f.ts - t0).toFixed(1),
    stress: f.stress,
    engagement: f.engagement,
    anomaly: f.anomaly_score,
    events: f.events,
  }));

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Timeline</span>
        <span className="muted">last {Math.round(data[data.length - 1].t)} s</span>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="g-stress" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="g-eng" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22c55e" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="g-an" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#a855f7" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#a855f7" stopOpacity={0} />
              </linearGradient>
            </defs>
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
            <Area
              type="monotone"
              dataKey="stress"
              stroke="#ef4444"
              strokeWidth={2}
              fill="url(#g-stress)"
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="engagement"
              stroke="#22c55e"
              strokeWidth={2}
              fill="url(#g-eng)"
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="anomaly"
              stroke="#a855f7"
              strokeWidth={1.5}
              fill="url(#g-an)"
              isAnimationActive={false}
            />
            {data
              .filter((d) => d.events && d.events.length > 0)
              .slice(-30)
              .map((d, i) => (
                <ReferenceLine
                  key={i}
                  x={d.t}
                  stroke="#fbbf24"
                  strokeDasharray="2 4"
                />
              ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <Legend />
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      <span className="dot dot-red" /> stress
      <span className="dot dot-green" /> engagement
      <span className="dot dot-purple" /> anomaly
      <span className="dot dot-yellow" /> event
    </div>
  );
}
