import type { FusedFrame } from "../types";

interface Props {
  frame: FusedFrame | null;
}

export function ReasoningPanel({ frame }: Props) {
  return (
    <div className="panel reasoning-panel">
      <div className="panel-header">
        <span>Reasoning</span>
        <span className="muted">
          {frame?.calibration?.ready
            ? `baseline · ${frame?.calibration?.samples ?? 0} samples`
            : `calibrating · ${frame?.calibration?.samples ?? 0} samples`}
        </span>
      </div>
      <div className="reasoning-state">
        <div>
          <span className="kv-key">Trend</span>
          <span className="kv-val">{frame?.trend ?? "—"}</span>
        </div>
        <div>
          <span className="kv-key">Attention</span>
          <span className="kv-val">{frame?.attention ?? "—"}</span>
        </div>
        <div>
          <span className="kv-key">Action</span>
          <span className="kv-val kv-action">{frame?.adaptive_action ?? "—"}</span>
        </div>
      </div>
      <ul className="reasoning-list">
        {(frame?.explanation ?? []).map((line, i) => (
          <li key={i}>{line}</li>
        ))}
        {(!frame || frame.explanation.length === 0) && (
          <li className="muted">Waiting for signal…</li>
        )}
      </ul>
      <div className="weights">
        <div>audio w: {(frame?.source_weights?.audio ?? 0).toFixed(2)}</div>
        <div>vision w: {(frame?.source_weights?.vision ?? 0).toFixed(2)}</div>
      </div>
    </div>
  );
}
