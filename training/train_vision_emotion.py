"""Train the facial-emotion CNN.

Layout expected:
    data/
        train/<class>/<image>.png
        val/<class>/<image>.png

Where <class> is one of FER-2013 labels. They are mapped to the canonical
5-class taxonomy used by the engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from backend.models.vision_emotion import VisionEmotionCNN
from backend.utils.schemas import EMOTION_LABELS


# FER-2013 -> canonical mapping. Tweak to your liking.
FER_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
    "fear": "frustrated",
    "disgust": "angry",
    "surprise": "happy",
}


class FERDataset(Dataset):
    def __init__(self, root: Path, transform=None) -> None:
        self.items: list[tuple[Path, int]] = []
        self.transform = transform
        if not root.exists():
            raise FileNotFoundError(root)
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            canonical = FER_MAP.get(sub.name.lower())
            if canonical is None or canonical not in EMOTION_LABELS:
                continue
            label = EMOTION_LABELS.index(canonical)
            for img in sub.iterdir():
                if img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.items.append((img, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("L").resize((48, 48), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img, label


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dir with train/ and val/ subdirs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="backend/models/weights/vision_emotion.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    root = Path(args.data)
    tfm_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.ToTensor()
    train_ds = FERDataset(root / "train", tfm_train)
    val_ds = FERDataset(root / "val", tfm_val)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = VisionEmotionCNN(num_classes=len(EMOTION_LABELS)).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_dl:
            x = x.to(args.device); y = y.to(args.device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in val_dl:
                x = x.to(args.device); y = y.to(args.device)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        print(f"epoch {epoch+1:02d}/{args.epochs}  loss={run_loss / max(1, len(train_ds)):.4f}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ saved to {out_path}")

    print(f"Done. best val_acc={best_acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
