# Training

All three trainable components ship with a sensible default architecture and
fall back to deterministic / uniform behaviour when no weights are present.
You can train any subset incrementally.

## 1. Vision emotion CNN

```bash
python -m training.train_vision_emotion \
    --data ./data/fer2013 \
    --epochs 30 \
    --batch-size 128 \
    --out backend/models/weights/vision_emotion.pt
```

Expected layout: `data/fer2013/{train,val}/<emotion>/<image>.png` with
48x48 grayscale faces. The 7 FER-2013 classes are mapped to our 5
canonical labels (`disgust` and `surprise` are folded into `angry` and
`happy` respectively — adjust in the script if you want a different
mapping).

## 2. Audio emotion CNN

```bash
python -m training.train_audio_emotion \
    --data ./data/ravdess \
    --epochs 40 \
    --batch-size 64 \
    --out backend/models/weights/audio_emotion.pt
```

Expected layout: `data/<dataset>/<emotion>/<file>.wav`. Each clip is
resampled to 16 kHz, trimmed to ~3 s, converted to log-mel
(64 × 96) and fed to the same small CNN used at inference time.

## 3. Temporal trend model

The temporal head needs *session* recordings, not isolated clips. Run the
backend with `--headless --record-sessions` (see `scripts/record_session.py`),
then:

```bash
python -m training.train_temporal \
    --data ./logs/sessions \
    --window-sec 8 \
    --epochs 20 \
    --out backend/models/weights/temporal.pt
```

The script reads the per-tick FusedFrame JSONL files written by the engine,
recreates a feature window for each tick, and trains the GRU to predict the
*self-reported* trend label that you can annotate in
`logs/sessions/<id>/labels.csv` (`ts_start,ts_end,trend`).

> Tip: even a handful of 5-minute sessions noticeably improves the trend
> head over the fallback estimator.
