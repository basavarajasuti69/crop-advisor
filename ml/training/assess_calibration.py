"""Assess probability calibration of a PlantDoc checkpoint.

Uses the held-out validation split (the one the checkpoint was selected
on) to compute Expected Calibration Error (ECE) and a reliability
table. Does NOT touch the test set for model selection.

Usage:
  python ml/training/assess_calibration.py \\
      --experiment plantdoc_27class_v9 \\
      --out ml/training/experiments/plantdoc_v9_audit/calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml/training"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import models, transforms  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from train_plantdoc import MappedSubset, build_split, resolve_classes  # noqa: E402

EXPERIMENTS = ROOT / "ml/training/experiments"
DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v3/data"


def ece(confidences, correct, n_bins=10):
    import numpy as np
    conf = np.asarray(confidences)
    corr = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    ece_val = 0.0
    table = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n == 0:
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                          "avg_confidence": None, "accuracy": None, "gap": None})
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(corr[mask].mean())
        gap = abs(avg_conf - acc)
        ece_val += (n / total) * gap
        table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n,
                      "avg_confidence": round(avg_conf, 4),
                      "accuracy": round(acc, 4),
                      "gap": round(gap, 4)})
    return float(ece_val), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="plantdoc_27class_v9")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    exp = EXPERIMENTS / args.experiment
    data = Path(args.data)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    classes = json.loads((exp / "classes.json").read_text()) if False else None
    import json as _json
    classes = _json.loads((exp / "classes.json").read_text())
    if isinstance(classes, dict):
        classes = classes.get("classes", classes)
    classes = list(classes)


    train_ds = ImageFolder(data / "train")
    test_ds = ImageFolder(data / "test")
    resolved = resolve_classes(train_ds.classes, test_ds.classes)
    assert resolved == classes, (resolved, classes)
    split = build_split(train_ds.samples, train_ds.classes,
                        test_ds.samples, test_ds.classes, resolved)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    val_dataset = MappedSubset(
        ImageFolder(data / "train", transform=eval_tf),
        split["validation_indices"], split["class_to_new"],
    )

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(exp / "best_model.pt", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()


    confidences = []
    correct = []
    with torch.no_grad():
        for images, labels in DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0):
            images = images.to(device)
            labels = labels.to(device)
            probs = torch.softmax(model(images), dim=1)
            conf, pred = probs.max(dim=1)
            confidences.extend(conf.cpu().tolist())
            correct.extend((pred == labels).cpu().tolist())

    ece_val, table = ece(confidences, correct, n_bins=10)

    # Per-class calibration: ECE is dominated by confident classes.
    # Need per-sample labels; re-run tracking labels.
    confidences2 = []
    correct2 = []
    labels2 = []
    with torch.no_grad():
        for images, labels in DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0):
            images = images.to(device)
            labels = labels.to(device)
            probs = torch.softmax(model(images), dim=1)
            conf, pred = probs.max(dim=1)
            confidences2.extend(conf.cpu().tolist())
            correct2.extend((pred == labels).cpu().tolist())
            labels2.extend(labels.cpu().tolist())

    per_class_ece: dict = {}
    for idx, name in enumerate(classes):
        mask = [i for i, lbl in enumerate(labels2) if lbl == idx]
        if not mask:
            per_class_ece[name] = None
            continue
        c = [confidences2[i] for i in mask]
        r = [correct2[i] for i in mask]
        e, _ = ece(c, r, n_bins=5)
        per_class_ece[name] = round(e, 4)

    report = {
        "experiment": args.experiment,
        "split": "validation (held out of train, NOT the test set)",
        "n_samples": len(confidences),
        "accuracy": round(sum(correct) / len(correct), 4),
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "ece_10bin": round(ece_val, 4),
        "reliability_table": table,
        "per_class_ece_5bin": per_class_ece,
        "interpretation": (
            "ECE measures how much the stated confidence differs from the "
            "empirical accuracy in each confidence bin. A well-calibrated "
            "model has ECE near 0. This model has NOT been calibrated "
            "(no temperature scaling), so confidence thresholds in the API "
            "(0.70 / 0.85) are engineering cut-offs, not calibrated "
            "probabilities of disease."
        ),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
