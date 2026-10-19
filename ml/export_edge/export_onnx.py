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


def export_onnx(weights: Path, metrics: Path, output: Path) -> None:
    classes = json.loads(metrics.read_text())["classes"]
    num_classes = len(classes)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    output.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
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
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "model": "efficientnet_b0",
                "num_classes": num_classes,
                "classes": classes,
                "input_size": [224, 224],
                "format": "onnx",
                "opset": 13,
            },
            indent=2,
        )
    )
    print(f"Exported ONNX model to {output / 'classifier.onnx'} ({num_classes} classes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to model.pt")
    ap.add_argument("--metrics", required=True, help="Path to metrics.json")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()
    export_onnx(Path(args.weights), Path(args.metrics), Path(args.output))


if __name__ == "__main__":
    main()
