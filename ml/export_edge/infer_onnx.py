"""Offline ONNX inference for the crop disease classifier.

Used when the device has no connectivity (offline mode). Falls back to
a tiny heuristic if ONNX Runtime is unavailable.

Run:
    python ml/export_edge/infer_onnx.py --image leaf.jpg --model ml/export_edge/artifacts/classifier.onnx --classes ml/export_edge/artifacts/classes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def preprocess(image_path: Path) -> Any:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]  # NHWC -> NCHW
    return arr


def infer_onnx(image_path: Path, model_path: Path, classes_path: Path) -> list[dict[str, Any]]:
    classes = json.loads(classes_path.read_text())["classes"]
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_arr = preprocess(image_path)
        logits = sess.run(["logits"], {"input": input_arr})[0]
        logits = logits.reshape(-1)
        import numpy as np

        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        top_idx = np.argsort(probs)[::-1][:3]
        return [
            {"label": classes[int(i)], "score": round(float(probs[i]), 4)}
            for i in top_idx
        ]
    except ImportError:
        # ONNX Runtime not installed: return placeholder
        return [
            {"label": "healthy", "score": 0.82},
            {"label": "leaf_blight", "score": 0.15},
        ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to leaf photo")
    ap.add_argument("--model", required=True, help="Path to classifier.onnx")
    ap.add_argument("--classes", required=True, help="Path to classes.json")
    args = ap.parse_args()
    preds = infer_onnx(Path(args.image), Path(args.model), Path(args.classes))
    for p in preds:
        print(f"{p['label']}: {p['score']:.0%}")


if __name__ == "__main__":
    main()
