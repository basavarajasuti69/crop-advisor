"""Export trained PyTorch EfficientNet-B0 to ONNX for offline edge inference.

Run:
    python ml/export_edge/export_onnx.py \
        --weights ml/training/artifacts/model.pt \
        --metrics ml/training/artifacts/metrics.json \
        --output ml/export_edge/artifacts/classifier.onnx

Produces:
    ml/export_edge/artifacts/classifier.onnx   - ONNX model
    ml/export_edge/artifacts/classes.json      - class label list
    ml/export_edge/artifacts/manifest.json     - export metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import models


def export_onnx(
    weights: Path,
    metrics: Path,
    output: Path,
    dataset: str | None = None,
    notes: str | None = None,
) -> None:
    payload = json.loads(metrics.read_text())
    classes = payload.get("classes") or payload
    num_classes = len(classes)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()

    output.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output / "classifier.onnx"),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    (output / "classes.json").write_text(json.dumps({"classes": classes}, indent=2))
    manifest = {
        "model": "efficientnet_b0",
        "num_classes": num_classes,
        "classes": classes,
        "input_size": [224, 224],
        "format": "onnx",
        "opset": 13,
        "source": str(weights),
        # The dataset the model was actually trained on, and the metrics
        # that back it up. "dataset" is None when nothing states it, so
        # the service cannot claim a dataset it does not know.
        "dataset": dataset or payload.get("dataset"),
        "metrics": {
            key: payload.get(key)
            for key in (
                "best_epoch",
                "best_validation_accuracy",
                "best_validation_macro_f1",
                "test_accuracy",
                "test_macro_f1",
                "training_images",
                "validation_images",
                "test_images",
                "lr",
                "batch_size",
                "label_smoothing",
                "class_weighting",
                "seed",
                "data",
            )
            if key in payload
        },
        # The softmax output of this model has not been calibrated, so the
        # score must not be presented as a disease probability.
        "confidence_is_calibrated": bool(
            payload.get("confidence_is_calibrated", False)
        ),
    }
    if notes:
        manifest["notes"] = notes
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Exported ONNX model to {output / 'classifier.onnx'} ({num_classes} classes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to model.pt")
    ap.add_argument("--metrics", required=True, help="Path to metrics.json")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument(
        "--dataset",
        help="The dataset the model was trained on, for the manifest.",
    )
    ap.add_argument(
        "--notes",
        help="A short note for the manifest, e.g. what the model is not.",
    )
    args = ap.parse_args()
    export_onnx(
        Path(args.weights),
        Path(args.metrics),
        Path(args.output),
        dataset=args.dataset,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
