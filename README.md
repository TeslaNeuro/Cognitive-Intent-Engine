# 🧠 Cognitive State & Intent Engine

A modular multimodal AI system that combines real-time **computer vision** and **microphone audio** to infer:

- **Emotion** — happy, sad, angry, neutral, frustrated
- **Cognitive state** — focused, confused, overloaded, fatigued
- **Intent** — problem-solving, asking for help, idle, exploring
- **Behaviour trends** — improving, deteriorating, disengaging
- **Stress / engagement / attention** — continuous-valued, with anomaly + event detection

It exposes results both as a **structured JSON stream** and through a live **React dashboard** with charts, gauges, the webcam feed, and an event log.

---

## ✨ Highlights

- **Three-level fusion**: feature-level, decision-level (calibrated, learnable weights) and context-level (temporal + behaviour-aware).
- **Temporal LSTM/GRU** over the last *N* seconds to predict trend direction and stress trajectory.
- **Hybrid reasoning engine**: transparent rule-based layer + optional learned classifier, with explanations attached to every output.
- **Per-user calibration**: rolling baseline of pitch, energy, neutral face, gaze, head pose. All affect-scores are z-normalized against this baseline.
- **Event detection**: frustration spike, attention drop, disengagement, fatigue onset, anomaly burst.
- **Adaptive responder**: the system reacts to user state (e.g. suggests a break, lowers UI density, surfaces help).
- **Self-supervised refinement**: rolling pseudo-label memory used to fine-tune the decision fuser online.
- **Anomaly detection**: streaming z-score + IsolationForest over the fused feature vector.
- **Edge-ready**: PyTorch → ONNX → TFLite conversion script, INT8 quantization, Jetson notes.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Runtime** | Python 3.10+ |
| **ML / inference** | PyTorch, torchvision, torchaudio, scikit-learn, NumPy, SciPy |
| **Vision** | OpenCV, MediaPipe, Pillow |
| **Audio** | librosa, sounddevice, soundfile |
| **API** | FastAPI, Uvicorn, WebSockets, Pydantic, PyYAML |
| **Frontend** | React 18, TypeScript, Vite, Recharts |
| **Edge export** (optional) | ONNX, ONNX Runtime; TensorFlow / TFLite for Jetson deployment |
| **Dev / test** | pytest, rich, loguru, tqdm |

---

## 🏗️ Architecture

```
┌──────────────────────────┐    ┌──────────────────────────┐
│ Webcam (OpenCV)          │    │ Microphone (sounddevice) │
└───────────┬──────────────┘    └──────────────┬───────────┘
            │ frames                           │ audio blocks
            ▼                                  ▼
  ┌─────────────────────┐            ┌─────────────────────┐
  │ Vision Pipeline     │            │ Audio Pipeline      │
  │ MediaPipe FaceMesh  │            │ librosa: MFCC, F0,  │
  │ EAR, brow, mouth,   │            │ RMS, speech rate,   │
  │ head pose, gaze     │            │ pause ratio         │
  │ FER CNN (PyTorch)   │            │ SER CNN (PyTorch)   │
  └──────────┬──────────┘            └──────────┬──────────┘
             │ emotion + features               │ emotion + features
             ▼                                  ▼
            ┌──────────────────────────────────────┐
            │ Level-1 Feature Fusion (concat+norm) │
            └────────────────┬─────────────────────┘
                             │
            ┌────────────────▼─────────────────────┐
            │ Level-2 Decision Fusion (weighted +  │
            │ calibrated, optionally learned)      │
            └────────────────┬─────────────────────┘
                             ▼
            ┌──────────────────────────────────────┐
            │ Temporal Model (LSTM, 5–10 s window) │
            │ stress / engagement / attention      │
            └────────────────┬─────────────────────┘
                             ▼
            ┌──────────────────────────────────────┐
            │ Level-3 Context Fusion + Reasoning   │
            │ rule engine + ML classifier + XAI    │
            └─────┬───────────────┬────────────────┘
                  │               │
                  ▼               ▼
          Event Detector    Adaptive Responder
                  │               │
                  └───────┬───────┘
                          ▼
                  FastAPI WebSocket
                          ▼
                  React Dashboard
```

---

## 📁 Layout

```
cognitive-state-engine/
├── backend/                # Python: capture, models, fusion, reasoning, API
│   ├── pipelines/          # audio + vision capture & feature extraction
│   ├── models/             # CNNs, LSTM, wrappers
│   ├── fusion/             # 3-level fusion
│   ├── reasoning/          # rules, ML classifier, explainability
│   ├── calibration/        # per-user baseline
│   ├── events/             # event/anomaly detection
│   ├── adaptive/           # adaptive responder
│   ├── state/              # rolling state store
│   ├── utils/              # timing, logging, schemas
│   └── main.py             # FastAPI + WebSocket entry point
├── frontend/               # React + Vite + TypeScript dashboard
├── training/               # training scripts for audio / vision / temporal
├── scripts/                # tflite export, dev launchers
├── config/                 # YAML configuration
├── tests/                  # unit tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 🐍 Backend (Python 3.10+)

```bash
cd cognitive-state-engine
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Run the engine + API server (auto-opens webcam & microphone):

```bash
python -m backend.main
# server on http://localhost:8000
# websocket on ws://localhost:8000/ws
```

Run **headless** (no API, console only) for quick smoke-testing:

```bash
python -m backend.main --headless
```

### ⚛️ Frontend (Node 18+)

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

### 🎓 Training (optional)

The system **ships with lightweight pretrained-style heads** (initialized with sensible weights) so it runs end-to-end immediately. To improve accuracy on your domain, train your own:

```bash
# Vision emotion (FER-2013 style 48x48 grayscale)
python -m training.train_vision_emotion --data ./data/fer2013

# Audio emotion (RAVDESS / CREMA-D / IEMOCAP)
python -m training.train_audio_emotion --data ./data/ravdess

# Temporal trend model
python -m training.train_temporal --data ./logs/sessions
```

Trained weights are saved into `backend/models/weights/` and picked up automatically.

### 📦 Edge deployment

```bash
python -m scripts.convert_to_tflite --model vision_emotion --quantize int8
python -m scripts.convert_to_tflite --model audio_emotion  --quantize int8
```

See [`scripts/README_jetson.md`](scripts/README_jetson.md) for Jetson Nano notes.

---

## 📤 Output format

Every ~100 ms the backend emits a structured frame on the WebSocket:

```json
{
  "ts": 1717000000.123,
  "emotion": {
    "label": "frustrated",
    "confidence": 0.78,
    "probs": {"happy":0.02,"sad":0.05,"angry":0.10,"neutral":0.05,"frustrated":0.78},
    "source_weights": {"audio": 0.42, "vision": 0.58}
  },
  "cognitive_state": "overloaded",
  "intent": "problem-solving",
  "attention": "low",
  "stress": 0.71,
  "engagement": 0.34,
  "fatigue": 0.22,
  "trend": "deteriorating",
  "events": [{"type":"frustration_spike","severity":0.8,"ts":1717000000.0}],
  "anomaly_score": 0.61,
  "explanation": [
    "pitch z-score +1.9 (above baseline)",
    "brow tension +1.2 (above baseline)",
    "rule: pitch↑ + brow↑ → frustration"
  ],
  "adaptive_action": "suggest_short_break",
  "calibration": {"samples": 312, "ready": true}
}
```

---

## ⚙️ Configuration

`config/default.yaml` controls cameras, sample rates, model paths, thresholds, fusion weights and adaptive responses. CLI flags and env vars override.

---

## ⚡ Performance

- Vision pipeline: ~25–30 FPS on a modern laptop CPU (MediaPipe).
- Audio pipeline: ~10 Hz feature extraction on 1 s rolling buffer.
- Fusion + temporal model: <5 ms per tick.
- End-to-end latency target: **<200 ms**.
- All pipelines run on dedicated threads with bounded queues — no blocking.

---

## 📊 Status

This is a **professional prototype**: every module is in place, runs end-to-end, and the model heads are designed to be replaced with your own trained weights. See `training/README.md` for what to train next.

---

## 👤 Author

Created and maintained by **Arshia Keshvari** (`@TeslaNeuro`).

## 📜 License

**[`MIT`](./LICENSE)**
