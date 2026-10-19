"""Phase 1 training stub: EfficientNet-B0 classifier over crop-disease images.

Run locally:
    python ml/training/train.py --data ml/data --epochs 5

Produces ml/training/artifacts/model.pt and a metrics json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


def build_model(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    data_dir = Path(args.data)
    classes = sorted(p.name for p in (data_dir / "train").iterdir() if p.is_dir())
    num_classes = len(classes)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(data_dir / "train", transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = build_model(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch+1} loss={total/len(train_dl):.4f}")

    out_dir = Path("ml/training/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"classes": classes, "epochs": args.epochs}, f)
    print(f"saved {out_dir / 'model.pt'} ({num_classes} classes)")


if __name__ == "__main__":
    main()
