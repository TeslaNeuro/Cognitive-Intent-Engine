# Jetson Nano (or similar edge) notes

This system was designed to run on a 2 GB Jetson Nano with USB webcam +
USB microphone. The defaults here assume Python 3.10+, but on JetPack 4.x
you'll often have Python 3.8 — most of this still works.

## 1. System packages

```bash
sudo apt update
sudo apt install -y \
    python3-pip libsndfile1 portaudio19-dev \
    libatlas-base-dev libjpeg-dev libavcodec-extra \
    cmake build-essential
```

## 2. Python deps

Install in an `nv-python3.10` venv if you have it, otherwise system Python:

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
# For Jetson, replace the generic torch wheel with the NVIDIA build:
#   https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

OpenCV, MediaPipe and librosa all have ARM64 wheels.

## 3. Models

Use the TFLite versions for vision + audio:

```bash
python -m scripts.convert_to_tflite --model vision_emotion --quantize int8
python -m scripts.convert_to_tflite --model audio_emotion  --quantize int8
```

Then in `config/default.yaml` set:

```yaml
vision:
  emotion_model: backend/models/weights/vision_emotion.tflite
audio:
  emotion_model: backend/models/weights/audio_emotion.tflite
```

…and swap the inference wrappers to load TFLite (left as a 30-line change
in `vision_emotion.py` / `audio_emotion.py` once you have a working
quantized graph; see `tflite_runtime.interpreter.Interpreter`).

## 4. Performance tips

- Set `vision.fps: 15` and `app.tick_hz: 5` for the Nano.
- Set `audio.feature_hop_s: 0.2` to halve audio compute.
- Pin the orchestrator thread to one CPU and the audio callback to another
  (`taskset`).
- Disable `vision.draw_overlays` if you don't need the MJPEG stream.
- Run the frontend on a different machine and point it at the Jetson:
  `frontend/vite.config.ts` proxy → `http://<jetson-ip>:8000`.

## 5. Headless service

A simple systemd unit:

```ini
# /etc/systemd/system/cse.service
[Unit]
Description=Cognitive State & Intent Engine
After=network.target

[Service]
ExecStart=/usr/bin/python3 -m backend.main --headless
WorkingDirectory=/home/jetson/cognitive-state-engine
Restart=on-failure
User=jetson

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now cse
journalctl -u cse -f
```
