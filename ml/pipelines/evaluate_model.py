"""Model evaluation step for SageMaker Pipeline.

Evaluates the trained model on field photos and writes evaluation_report.json
with field_photo_accuracy metric used by the approval gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def main() -> None:
    model_dir = Path("/opt/ml/processing/model")
    test_dir = Path("/opt/ml/processing/input/test")
    output_dir = Path("/opt/ml/processing/output/evaluation_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class names from model artifact metadata if available
    classes = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    num_classes = len(classes)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    state_path = model_dir / "model.pth"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = ImageFolder(test_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=32)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total if total else 0.0
    report = {
        "metrics": {
            "field_photo_accuracy": round(accuracy, 4),
            "num_classes": num_classes,
            "total_images": total,
        }
    }

    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
