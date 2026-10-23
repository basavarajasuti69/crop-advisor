"""Leaf-disease inference.

This module owns everything that turns image bytes into predictions:
the preprocessing, the ONNX Runtime and PyTorch backends, the class-name
mapping and the leaf-likeness check. It is deliberately free of LangGraph
and of any training code, so the serving path can be read - and tested -
on its own.

The preprocessing here must stay identical to the transform the model was
trained with (``Resize(256) + CenterCrop(224) + ImageNet normalisation``).
``tests/test_classifier.py`` asserts the numpy and PyTorch versions agree,
and ``ml/export_edge/infer_onnx.py`` uses the same one.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image  # noqa: F401

from agents.label_map import (
    is_plantdoc_label,
    plantdoc_crop,
    plantdoc_disease,
    plantdoc_is_healthy,
    plantdoc_kb_section,
)

__all__ = [
    "SUPPORTED_CROPS",
    "ImageUnreadable",
    "ModelUnavailable",
    "crop_from_label",
    "disease_from_label",
    "is_healthy_label",
    "kb_section_for_label",
    "load_class_names",
    "load_model_metadata",
    "looks_like_leaf",
    "predict",
    "predict_onnx",
]


class ModelUnavailable(Exception):
    """The classifier model file is missing or could not be loaded."""


class ImageUnreadable(Exception):
    """The uploaded image could not be opened or decoded."""

# The ImageNet statistics used by the training transform.
_ONNX_MEAN = (0.485, 0.456, 0.406)
_ONNX_STD = (0.229, 0.224, 0.225)

# Crop-specific classifier labels the model can produce.
SUPPORTED_CROPS = {"rice", "wheat", "cotton", "maize", "tomato"}


def _repo_root() -> Path:
    """Repo root, so paths do not depend on the current working directory."""
    return Path(__file__).resolve().parent.parent


def _artifacts_dir() -> Path:
    """Where the deployed model lives.

    ``CA_MODEL_DIR`` points the service at a different export, which is
    how a new model is tried without overwriting the one that ships.
    Without it the module-level ``_ONNX_DIR`` is used, which the tests
    override.
    """
    override = os.getenv("CA_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    return _ONNX_DIR


_ONNX_DIR = _repo_root() / "ml/export_edge/artifacts"


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------


def crop_from_label(label: str) -> str | None:
    """Return the crop encoded in a classifier label.

    Handles both label schemes in this repository: the crop-specific
    ``rice_healthy`` form and the PlantDoc form such as
    ``Tomato mold leaf``, where the crop has to be looked up.
    """
    if is_plantdoc_label(label):
        return plantdoc_crop(label)

    prefix, separator, _disease = label.partition("_")

    if separator and prefix in SUPPORTED_CROPS:
        return prefix

    return None


def disease_from_label(label: str) -> str:
    """Normalize a classifier label for retrieval and localization."""
    if is_plantdoc_label(label):
        return plantdoc_disease(label) or label

    crop = crop_from_label(label)

    if crop is None:
        return label

    return label[len(crop) + 1:]


def kb_section_for_label(label: str) -> str | None:
    """The knowledge-base section for a label, or None when there is none.

    A PlantDoc label such as ``Potato leaf early blight`` has no
    section: the knowledge base does not cover potato. Returning None
    lets the service say so rather than retrieving rice guidance.
    """
    if is_plantdoc_label(label):
        return plantdoc_kb_section(label)
    return None


def is_healthy_label(label: str) -> bool:
    """Recognize both legacy and crop-specific healthy labels."""
    if is_plantdoc_label(label):
        return plantdoc_is_healthy(label)
    return label == "healthy" or disease_from_label(label) == "healthy"


def load_class_names(model_dir: Path | None = None) -> list[str]:
    """The class names of the deployed model, in the order it was trained.

    Reads ``classes.json`` next to the ONNX export. Falls back to the
    legacy two-class list so the service still starts if the export has
    not been produced yet.
    """
    directory = Path(model_dir) if model_dir else _artifacts_dir()
    path = directory / "classes.json"
    if path.exists():
        classes = json.loads(path.read_text())
        if isinstance(classes, dict):
            classes = classes.get("classes", classes)
        if isinstance(classes, list) and classes:
            return list(classes)
    return ["healthy", "leaf_blight"]


def load_model_metadata(model_dir: Path | None = None) -> dict[str, Any]:
    """What the deployed model actually is, read from its own manifest.

    The service describes the model from the export rather than from a
    hardcoded string, so the description cannot drift away from the
    artifact that is actually being served.
    """
    directory = Path(model_dir) if model_dir else _artifacts_dir()
    manifest_path = directory / "manifest.json"
    metadata: dict[str, Any] = {
        "classes": load_class_names(directory),
        "manifest": None,
        "dataset": None,
        "num_classes": None,
        "source": "unknown",
    }
    if not manifest_path.exists():
        return metadata

    manifest = json.loads(manifest_path.read_text())
    metadata["manifest"] = manifest
    metadata["num_classes"] = manifest.get("num_classes")
    metadata["dataset"] = manifest.get("dataset")
    metadata["source"] = manifest.get("source", "export")
    metadata["metrics"] = manifest.get("metrics")
    metadata["confidence_is_calibrated"] = bool(
        manifest.get("confidence_is_calibrated", False)
    )
    return metadata


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _torch_transform():
    """The training/eval transform: Resize(256) + CenterCrop(224) + Normalize."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(list(_ONNX_MEAN), list(_ONNX_STD)),
    ])


def onnx_preprocess(img: Image.Image):
    """NumPy equivalent of ``_torch_transform()`` - must stay byte-identical."""
    import numpy as np

    width, height = img.size
    if width <= height:
        new_w, new_h = 256, int(256 * height / width)
    else:
        new_h, new_w = 256, int(256 * width / height)
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = int(round((new_w - 224) / 2.0))
    top = int(round((new_h - 224) / 2.0))
    cropped = resized.crop((left, top, left + 224, top + 224))
    arr = np.asarray(cropped, dtype="float32") / 255.0
    arr = (arr - np.array(_ONNX_MEAN, dtype="float32")) / np.array(
        _ONNX_STD, dtype="float32"
    )
    return arr.transpose(2, 0, 1)[None, ...]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_ONNX_SESSION = None
_ONNX_CLASSES: list[str] = []
_TORCH_MODEL = None
_TORCH_CLASSES: list[str] = []


def load_onnx():
    """Load the ONNX Runtime session. Returns (session, classes) or None."""
    global _ONNX_SESSION, _ONNX_CLASSES
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION, _ONNX_CLASSES
    directory = _artifacts_dir()
    model_path = directory / "classifier.onnx"
    classes_path = directory / "classes.json"
    if not (model_path.exists() and classes_path.exists()):
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    try:
        _ONNX_CLASSES = json.loads(classes_path.read_text())["classes"]
        _ONNX_SESSION = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        # The file may be an LFS pointer, truncated, or corrupted.
        # Log and fall back to the torch backend instead of crashing.
        import logging

        logging.getLogger(__name__).warning(
            "ONNX model could not be loaded from %s: %s", model_path, exc
        )
        return None
    return _ONNX_SESSION, _ONNX_CLASSES


def load_torch(weights_path: str | None = None, metrics_path: str | None = None):
    """Load the PyTorch checkpoint. Paths are injectable for tests."""
    global _TORCH_MODEL, _TORCH_CLASSES
    if (
        weights_path is None
        and metrics_path is None
        and _TORCH_MODEL is not None
    ):
        return _TORCH_MODEL, _TORCH_CLASSES
    root = _repo_root()
    if weights_path is None and metrics_path is None:
        # Prefer the export directory so a PlantDoc export is picked up
        # automatically; fall back to the legacy training artifacts.
        directory = _artifacts_dir()
        candidates = [
            (directory / "model.pt", directory / "classes.json"),
            (
                root / "ml/training/artifacts/model.pt",
                root / "ml/training/artifacts/metrics.json",
            ),
        ]
        _w, _m = next(
            (w, m) for w, m in candidates if w.exists() and m.exists()
        )
        weights_path = str(_w)
        metrics_path = str(_m)
    w: Path = Path(weights_path) if weights_path else root / "ml/training/artifacts/model.pt"
    metrics_path_obj: Path = Path(metrics_path) if metrics_path else root / "ml/training/artifacts/metrics.json"
    if not (w.exists() and metrics_path_obj.exists()):
        raise ModelUnavailable("classifier model file missing")
    import torch
    from torchvision import models

    payload = json.loads(metrics_path_obj.read_text())
    classes = payload.get("classes") or payload
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, len(classes)
    )
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
    model.eval()
    if weights_path is None and metrics_path is None:
        _TORCH_MODEL = model
        _TORCH_CLASSES = classes
    return model, classes


def predict_onnx(image_bytes: bytes, top_k: int = 3) -> list[dict[str, Any]]:
    """Classify with ONNX Runtime. No torch required."""
    import numpy as np

    loaded = load_onnx()
    if loaded is None:
        raise ModelUnavailable("onnx model not available")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ImageUnreadable("could not open image") from exc
    session, classes = loaded
    logits = session.run(
        ["logits"], {"input": onnx_preprocess(img)}
    )[0].reshape(-1)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    order = np.argsort(probs)[::-1][: min(top_k, len(classes))]
    return [
        {"label": classes[int(i)], "score": round(float(probs[i]), 4)}
        for i in order
    ]


def predict(
    image_bytes: bytes,
    top_k: int = 3,
    model=None,
    classes=None,
) -> list[dict[str, Any]]:
    """Classify an image, preferring ONNX Runtime over the torch checkpoint.

    Raises ``ImageUnreadable`` when the image cannot be opened and
    ``ModelUnavailable`` when no model is installed: the service must
    never return a made-up prediction.
    """
    try:
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ImageUnreadable("could not open image") from exc
    if model is None and classes is None:
        if load_onnx() is not None:
            return predict_onnx(image_bytes, top_k)
        try:
            model, classes = load_torch()
        except ModelUnavailable:
            raise
        except FileNotFoundError as exc:
            raise ModelUnavailable(str(exc)) from exc
    elif model is None or classes is None:
        raise ValueError("model and classes must be provided together")

    import torch

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _torch_transform()(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, min(top_k, len(classes)))
    return [
        {"label": classes[int(i)], "score": round(float(p), 4)}
        for i, p in zip(top.indices, top.values)
    ]


# ---------------------------------------------------------------------------
# Is this even a leaf photo?
# ---------------------------------------------------------------------------


# Thresholds calibrated on the PlantDoc images in this repository.
# Measured over 500 real leaf photos: the vegetation fraction never fell
# below 0.20 and the edge density stayed between 3.2 and 26.7.
_MIN_VEGETATION = 0.15
_FLAT_EDGE_MAX = 1.5
_NOISE_EDGE_MIN = 40.0


def looks_like_leaf(image_bytes: bytes) -> bool:
    """A conservative check that the photo plausibly shows a leaf.

    The classifier will happily assign a disease to a picture of a
    wallet, so the service checks for vegetation-like colour and for
    texture before trusting the model.

    Deliberately conservative: it only rejects images that are clearly
    flat, clearly noise, or clearly not plant-coloured, so a real leaf -
    including a dry or yellow one - is never turned away. Measured
    against the 2,546 images in this repository it rejects none of them.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False

    import numpy as np

    small = img.resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype="float32")

    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]
    maximum = arr.max(axis=2)
    minimum = arr.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)

    # Vegetation is green, or yellow-brown when it is dying: green must
    # lead, or the colour must be a warm one with visible saturation.
    green_lead = (green >= red) & (green >= blue)
    warm = (red > green) & (red > blue) & (saturation > 0.25)
    vegetation = float((green_lead | warm).mean())

    # Texture: the mean absolute difference between neighbouring pixels.
    edges = float(
        (
            np.abs(np.diff(arr, axis=0)).mean()
            + np.abs(np.diff(arr, axis=1)).mean()
        )
        / 2.0
    )

    if edges < _FLAT_EDGE_MAX:
        return False  # a flat colour, not a photograph
    if edges > _NOISE_EDGE_MIN:
        return False  # noise, not a photograph
    return vegetation >= _MIN_VEGETATION
