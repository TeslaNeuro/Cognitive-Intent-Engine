import { useEffect, useState } from "react";
import type { Event, FusedFrame } from "../types";

interface Props {
  latest: FusedFrame | null;
}

const TYPE_COLORS: Record<string, string> = {
  frustration_spike: "#a855f7",
  attention_drop: "#fbbf24",
  disengagement: "#94a3b8",
  fatigue_onset: "#f97316",
  stress_rising: "#ef4444",
};

export function EventLog({ latest }: Props) {
  const [events, setEvents] = useState<Event[]>([]);

  useEffect(() => {
    if (!latest || !latest.events?.length) return;
    setEvents((prev) => {
      const merged = [...latest.events, ...prev];
      // Dedupe by (type, rounded ts second).
      const seen = new Set<string>();
      const out: Event[] = [];
      for (const e of merged) {
        const key = `${e.type}:${Math.round(e.ts)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(e);
      }
      return out.slice(0, 50);
    });
  }, [latest]);

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Events</span>
        <span className="muted">last 50</span>
      </div>
      <div className="event-list">
        {events.length === 0 && <div className="muted">No events yet.</div>}
        {events.map((e, i) => (
          <div className="event" key={i}>
            <span
              className="event-dot"
              style={{ background: TYPE_COLORS[e.type] ?? "#94a3b8" }}
            />
            <span className="event-type">{e.type.replace(/_/g, " ")}</span>
            <span className="event-sev">{Math.round(e.severity * 100)}%</span>
            {e.detail && <span className="event-detail">{e.detail}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
