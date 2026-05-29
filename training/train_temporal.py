"""Train the temporal trend GRU on recorded session JSONL files.

Each session is a directory containing:
    frames.jsonl   one FusedFrame per line
    labels.csv     ts_start,ts_end,trend           (optional; can be auto-labeled)

If labels.csv is missing, we auto-label using the *current rule-based fallback*
(it's noisy but useful as a warm-start for the GRU).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from backend.fusion.feature_fusion import FEATURE_DIM, FEATURE_INDEX, FEATURE_NAMES
from backend.models.temporal import TemporalTrendModel
from backend.utils.schemas import TREND_LABELS


def load_session(session_dir: Path) -> Tuple[np.ndarray, List[float], List[str]]:
    """Return (X[T,F], timestamps, per-tick labels)."""
    frames_path = session_dir / "frames.jsonl"
    if not frames_path.exists():
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), [], []
    feats: List[np.ndarray] = []
    ts: List[float] = []
    fallback_labels: List[str] = []
    with frames_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vec = np.zeros(FEATURE_DIM, dtype=np.float32)
            for name, idx in FEATURE_INDEX.items():
                if name in obj:
                    vec[idx] = float(obj[name])
            feats.append(vec)
            ts.append(float(obj.get("ts", 0.0)))
            fallback_labels.append(str(obj.get("trend", "stable")))
    X = np.stack(feats) if feats else np.zeros((0, FEATURE_DIM), dtype=np.float32)

    # Optional manual labels override.
    labels_path = session_dir / "labels.csv"
    labels = list(fallback_labels)
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t0 = float(row["ts_start"]); t1 = float(row["ts_end"])
                lbl = row["trend"]
                for i, t in enumerate(ts):
                    if t0 <= t <= t1:
                        labels[i] = lbl
    return X, ts, labels


class WindowDataset(Dataset):
    def __init__(self, sessions: List[Path], window: int):
        self.window = window
        self.samples: List[Tuple[np.ndarray, int]] = []
        for s in sessions:
            X, _, labels = load_session(s)
            for i in range(window, X.shape[0]):
                lbl = labels[i]
                if lbl not in TREND_LABELS:
                    continue
                self.samples.append((X[i - window : i], TREND_LABELS.index(lbl)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x.astype(np.float32)), y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Directory of session subdirs")
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--tick-hz", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="backend/models/weights/temporal.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    root = Path(args.data)
    sessions = [d for d in root.iterdir() if d.is_dir() and (d / "frames.jsonl").exists()]
    if not sessions:
        print(f"No sessions in {root}")
        return 1
    window = int(args.window_sec * args.tick_hz)
    ds = WindowDataset(sessions, window=window)
    print(f"sessions={len(sessions)} windows={len(ds)} features={FEATURE_DIM}")

    val_n = max(1, len(ds) // 10)
    train, val = torch.utils.data.random_split(ds, [len(ds) - val_n, val_n])
    train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=args.batch_size)

    model = TemporalTrendModel(input_dim=FEATURE_DIM, hidden_size=64).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_trend = nn.CrossEntropyLoss()
    loss_scal = nn.MSELoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    stress_idx = FEATURE_INDEX.get("pitch_z", 0)
    engagement_idx = FEATURE_INDEX.get("attention_score", 0)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(args.device); y = torch.as_tensor(y).long().to(args.device)
            t_logits, scalars = model(x)
            # Approximate scalar targets from features (self-supervised).
            target_stress = torch.sigmoid(x[:, -1, stress_idx]).unsqueeze(-1)
            target_eng = x[:, -1, engagement_idx].clamp(0, 1).unsqueeze(-1)
            target_attn = target_eng
            target = torch.cat([target_stress, target_eng, target_attn], dim=-1)
            loss = loss_trend(t_logits, y) + 0.2 * loss_scal(scalars, target)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in val_dl:
                x = x.to(args.device); y = torch.as_tensor(y).long().to(args.device)
                logits, _ = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        print(f"epoch {epoch+1:02d}/{args.epochs}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
