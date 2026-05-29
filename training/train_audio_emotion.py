"""Train the audio-emotion CNN on log-mel spectrograms.

Layout expected:
    data/
        <class>/<file>.wav
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from backend.models.audio_emotion import AudioEmotionCNN, _fit_mel
from backend.utils.schemas import EMOTION_LABELS


# Common public-dataset emotion names -> canonical taxonomy.
AUDIO_MAP = {
    "happy": "happy", "happiness": "happy", "joy": "happy",
    "sad": "sad", "sadness": "sad",
    "angry": "angry", "anger": "angry",
    "neutral": "neutral", "calm": "neutral",
    "fear": "frustrated", "frustration": "frustrated", "frustrated": "frustrated",
    "disgust": "angry", "surprise": "happy",
}


class WAVEmotionDataset(Dataset):
    def __init__(self, files: List[Tuple[Path, int]], sr: int = 16000) -> None:
        self.files = files
        self.sr = sr

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path, label = self.files[idx]
        y, _ = librosa.load(str(path), sr=self.sr, mono=True)
        # Take a random 3-second window if longer, pad if shorter.
        target = 3 * self.sr
        if len(y) > target:
            start = random.randint(0, len(y) - target)
            y = y[start : start + target]
        elif len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        # Optional small augmentation
        if random.random() < 0.3:
            y = y * random.uniform(0.7, 1.3)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=64, n_fft=1024, hop_length=160, fmin=20, fmax=self.sr // 2,
        )
        log_mel = librosa.power_to_db(mel + 1e-10)
        x = _fit_mel(log_mel, n_mels=64, frames=96)
        return torch.from_numpy(x).unsqueeze(0), label


def gather(root: Path) -> List[Tuple[Path, int]]:
    out: List[Tuple[Path, int]] = []
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        canonical = AUDIO_MAP.get(cls_dir.name.lower())
        if canonical is None:
            continue
        label = EMOTION_LABELS.index(canonical)
        for wav in cls_dir.rglob("*.wav"):
            out.append((wav, label))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--out", default="backend/models/weights/audio_emotion.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    files = gather(Path(args.data))
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_split))
    val = files[:n_val]
    train = files[n_val:]
    print(f"train={len(train)} val={len(val)}")

    train_ds = WAVEmotionDataset(train)
    val_ds = WAVEmotionDataset(val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = AudioEmotionCNN(num_classes=len(EMOTION_LABELS)).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_dl:
            x = x.to(args.device); y = torch.as_tensor(y).long().to(args.device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in val_dl:
                x = x.to(args.device); y = torch.as_tensor(y).long().to(args.device)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        print(f"epoch {epoch+1:02d}/{args.epochs}  loss={run_loss / max(1, len(train)):.4f}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ saved to {out_path}")

    print(f"Done. best val_acc={best_acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
