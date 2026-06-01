import { EmotionChart } from "./components/EmotionChart";
import { EventLog } from "./components/EventLog";
import { Gauges } from "./components/Gauges";
import { ReasoningPanel } from "./components/ReasoningPanel";
import { Timeline } from "./components/Timeline";
import { VideoPanel } from "./components/VideoPanel";
import { useFusedFrameStream } from "./hooks/useWebSocket";

export default function App() {
  const { status, latest, history } = useFusedFrameStream(600);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          Cognitive State &amp; Intent Engine
        </div>
        <div className={`status status-${status}`}>{status}</div>
      </header>

      <main className="grid">
        <section className="col col-left">
          <VideoPanel frame={latest} />
          <Gauges frame={latest} />
        </section>
        <section className="col col-center">
          <Timeline history={history} />
          <EmotionChart history={history} />
        </section>
        <section className="col col-right">
          <ReasoningPanel frame={latest} />
          <EventLog latest={latest} />
        </section>
      </main>

      <footer className="footer">
        <span>
          face: {latest?.face_detected ? "✓" : "✗"} · voice:{" "}
          {latest?.has_voice ? "✓" : "✗"} · samples:{" "}
          {latest?.calibration?.samples ?? 0}
        </span>
        <span>v0.1.0 · realtime multimodal</span>
      </footer>
    </div>
  );
}
