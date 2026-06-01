import type { FusedFrame } from "../types";

interface Props {
  frame: FusedFrame | null;
}

function Gauge({
  label,
  value,
  hue,
}: {
  label: string;
  value: number;
  hue: number;
}) {
  const pct = Math.max(0, Math.min(1, value));
  const display = (pct * 100).toFixed(0);
  return (
    <div className="gauge">
      <div className="gauge-label">{label}</div>
      <div className="gauge-bar">
        <div
          className="gauge-fill"
          style={{
            width: `${pct * 100}%`,
            background: `hsl(${hue} 80% 55%)`,
          }}
        />
      </div>
      <div className="gauge-value">{display}</div>
    </div>
  );
}

export function Gauges({ frame }: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span>Live signals</span>
        <span className="muted">
          {frame ? `${frame.latency_ms.toFixed(0)} ms` : "—"}
        </span>
      </div>
      <div className="gauges">
        <Gauge label="Stress" value={frame?.stress ?? 0} hue={0} />
        <Gauge label="Engagement" value={frame?.engagement ?? 0} hue={140} />
        <Gauge label="Fatigue" value={frame?.fatigue ?? 0} hue={30} />
        <Gauge label="Cognitive load" value={frame?.cognitive_load ?? 0} hue={260} />
        <Gauge label="Anomaly" value={frame?.anomaly_score ?? 0} hue={320} />
      </div>
    </div>
  );
}
