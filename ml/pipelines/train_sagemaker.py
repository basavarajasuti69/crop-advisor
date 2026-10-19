"""SageMaker training entry point for crop disease classifier."""

from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    ap.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args = ap.parse_args()

    train_dir = Path(args.train)
    classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    num_classes = len(classes)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = ImageFolder(train_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = build_model(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, y in dl:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch+1} loss={total/len(dl):.4f}")

    # Save model
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")

    # Save class list for inference
    with open(model_dir / "classes.json", "w") as f:
        json.dump({"classes": classes}, f)

    print(f"Training complete. {num_classes} classes saved to {args.model_dir}")


if __name__ == "__main__":
    main()
