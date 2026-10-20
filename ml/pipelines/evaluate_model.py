"""Evaluate the classifier on a labelled folder of images.

Reports overall and per-class accuracy, a confusion matrix, and - the
number that matters for the app - how many images the API would
actually diagnose rather than answer "retake the photo".

The evaluation uses the repo's own preprocessing
(``agents.graph._onnx_preprocess`` / ``_torch_transform``), so it
cannot drift from what the API does at runtime.

Examples
--------
Evaluate the model that ships (ONNX):

    python ml/pipelines/evaluate_model.py \\
        --onnx ml/export_edge/artifacts/classifier.onnx \\
        --data ml/data/test

Evaluate the PyTorch checkpoint before exporting it:

    python ml/pipelines/evaluate_model.py \\
        --model ml/training/artifacts/model.pt \\
        --metrics ml/training/artifacts/metrics.json \\
        --data ml/data/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.graph import CONFIDENCE_MIN, HEALTHY_MIN  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _classes_from_metrics(metrics_path: Path) -> list[str]:
    return json.loads(metrics_path.read_text())["classes"]


def _classes_from_onnx(onnx_path: Path) -> list[str]:
    classes_path = onnx_path.parent / "classes.json"
    if not classes_path.exists():
        raise SystemExit(
            f"{classes_path} is missing - it lists the class names in order."
        )
    return json.loads(classes_path.read_text())["classes"]


def _load_images(data_dir: Path) -> list[tuple[Path, str]]:
    """Every image under data_dir, labelled with its class folder name."""
    if not data_dir.exists():
        raise SystemExit(f"{data_dir} does not exist")
    found: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for image in sorted(class_dir.rglob("*")):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                found.append((image, class_dir.name))
    if not found:
        raise SystemExit(f"no images found under {data_dir}")
    return found


def _predict_onnx(onnx_path: Path, images: list[Path]) -> tuple[list[str], list[float]]:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from agents.graph import _onnx_preprocess

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions: list[str] = []
    scores: list[float] = []
    for image in images:
        with Image.open(image) as handle:
            tensor = _onnx_preprocess(handle.convert("RGB"))
        logits = session.run([output_name], {input_name: tensor})[0].reshape(-1)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        predictions.append(str(int(probs.argmax())))
        scores.append(float(probs.max()))
    return predictions, scores


def _predict_torch(model_path: Path, images: list[Path]) -> tuple[list[str], list[float]]:
    import torch
    from torchvision import models

    from agents.graph import _torch_transform

    checkpoint = torch.load(model_path, map_location="cpu")
    num_classes = len(checkpoint.get("classes", [])) or 2
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    transform = _torch_transform()
    predictions: list[str] = []
    scores: list[float] = []
    with torch.no_grad():
        for image in images:
            from PIL import Image

            with Image.open(image) as handle:
                tensor = transform(handle.convert("RGB")).unsqueeze(0)
            probs = torch.softmax(model(tensor), dim=1)[0]
            top = int(probs.argmax())
            predictions.append(str(top))
            scores.append(float(probs[top]))
    return predictions, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="folder of class sub-folders")
    parser.add_argument("--onnx", help="the ONNX model to evaluate")
    parser.add_argument("--model", help="the PyTorch checkpoint to evaluate")
    parser.add_argument("--metrics", help="metrics.json with the class list")
    parser.add_argument("--output", help="where to write the JSON report")
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_MIN,
        help="confidence above which the API answers instead of asking for a retake",
    )
    args = parser.parse_args()

    if not args.onnx and not args.model:
        raise SystemExit("pass --onnx or --model")

    classes = (
        _classes_from_onnx(Path(args.onnx))
        if args.onnx
        else _classes_from_metrics(Path(args.metrics))
    )
    labelled = _load_images(Path(args.data))
    images = [image for image, _ in labelled]
    truth = [label for _, label in labelled]

    scores: list[float] = []
    if args.onnx:
        predicted_idx, scores = _predict_onnx(Path(args.onnx), images)
        predicted = [classes[int(i)] for i in predicted_idx]
    else:
        predicted_idx, scores = _predict_torch(Path(args.model), images)
        predicted = [classes[int(i)] for i in predicted_idx]

    total = len(truth)
    correct = sum(1 for want, got in zip(truth, predicted) if want == got)
    above = sum(1 for score in scores if score >= args.threshold)
    diagnosed = sum(
        1
        for want, got, score in zip(truth, predicted, scores)
        if want == got and score >= args.threshold
    )

    per_class: dict[str, dict[str, float]] = {}
    for name in classes:
        wanted = [i for i, label in enumerate(truth) if label == name]
        if not wanted:
            per_class[name] = {"support": 0, "accuracy": 0.0}
            continue
        hits = sum(1 for i in wanted if predicted[i] == name)
        per_class[name] = {
            "support": len(wanted),
            "accuracy": round(hits / len(wanted), 4),
        }

    matrix: dict[str, dict[str, int]] = {}
    for name in classes:
        row = {
            other: sum(
                1
                for i, label in enumerate(truth)
                if label == name and predicted[i] == other
            )
            for other in classes
        }
        matrix[name] = row

    report = {
        "data": str(args.data),
        "model": str(args.onnx or args.model),
        "classes": classes,
        "metrics": {
            "accuracy": round(correct / total, 4) if total else 0.0,
            "correct": correct,
            "total_images": total,
            "above_threshold": above,
            "above_threshold_rate": round(above / total, 4) if total else 0.0,
            "correct_and_above_threshold": diagnosed,
            "would_diagnose_rate": round(diagnosed / total, 4) if total else 0.0,
        },
        "per_class": per_class,
        "confusion_matrix": matrix,
        "notes": (
            "would_diagnose_rate is the share of images the API would answer "
            f"for rather than ask for a retake (threshold {args.threshold}, "
            f"healthy results also need {HEALTHY_MIN})."
        ),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
