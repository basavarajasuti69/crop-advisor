from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, ".")

from agents import classifier  # noqa: E402
from agents.classifier import (  # noqa: E402
    ImageUnreadable,
    ModelUnavailable,
    crop_from_label,
    disease_from_label,
    is_healthy_label,
    load_class_names,
    looks_like_leaf,
    predict,
    predict_onnx,
)

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "ml/export_edge/artifacts/classifier.onnx"


def _leaf_image_bytes(size=(300, 300), seed=0) -> bytes:
    """A green image with some texture, so it passes the leaf check."""
    import random

    rng = random.Random(seed)
    img = Image.new("RGB", size, color=(34, 139, 34))
    pixels = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            jitter = rng.randint(-25, 25)
            pixels[x, y] = (
                max(0, min(255, 34 + jitter)),
                max(0, min(255, 139 + jitter)),
                max(0, min(255, 34 + jitter)),
            )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- class name mapping ----------

def test_crop_from_label():
    assert crop_from_label("maize_leaf_blight") == "maize"
    assert crop_from_label("tomato_blight") == "tomato"
    assert crop_from_label("cotton_leaf_curl_virus") == "cotton"
    # A legacy label with no crop prefix.
    assert crop_from_label("leaf_blight") is None
    # An unknown prefix is not a crop.
    assert crop_from_label("grape_black_rot") is None


def test_disease_from_label():
    assert disease_from_label("maize_leaf_blight") == "leaf_blight"
    assert disease_from_label("tomato_blight") == "blight"
    assert disease_from_label("leaf_blight") == "leaf_blight"


def test_is_healthy_label():
    assert is_healthy_label("healthy") is True
    assert is_healthy_label("maize_healthy") is True
    assert is_healthy_label("maize_leaf_blight") is False


def test_load_class_names_matches_the_deployed_export():
    names = load_class_names()
    assert names
    if ONNX_PATH.exists():
        classes_file = ONNX_PATH.parent / "classes.json"
        if classes_file.exists():
            stored = json.loads(classes_file.read_text())
            if isinstance(stored, dict):
                stored = stored["classes"]
            assert names == stored


# ---------- preprocessing ----------

def test_onnx_preprocess_matches_the_torch_transform():
    """The serving path and the training path must agree exactly."""
    from agents.classifier import _torch_transform

    img = Image.new("RGB", (517, 389), color=(40, 150, 60))
    expected = _torch_transform()(img).numpy()
    # onnx_preprocess adds the batch dimension the ONNX session expects.
    actual = classifier.onnx_preprocess(img)[0]
    assert actual.shape == expected.shape
    assert np.abs(expected - actual).max() < 1e-6


def test_onnx_preprocess_handles_portrait_and_landscape():
    from agents.classifier import _torch_transform

    for size in ((300, 500), (500, 300), (224, 224)):
        img = Image.new("RGB", size, color=(40, 150, 60))
        expected = _torch_transform()(img).numpy()
        actual = classifier.onnx_preprocess(img)[0]
        assert np.abs(expected - actual).max() < 1e-6, size


# ---------- prediction ----------

def test_predict_rejects_a_file_that_is_not_an_image():
    with pytest.raises(ImageUnreadable):
        predict(b"this is not an image")


def test_predict_uses_an_injected_model():
    class _FakeModel:
        def __call__(self, x):
            import torch

            return torch.tensor([[0.7, 0.3]])

    predictions = predict(
        _leaf_image_bytes(),
        model=_FakeModel(),
        classes=["healthy", "leaf_blight"],
    )
    assert predictions[0]["label"] == "healthy"
    assert 0.0 <= predictions[0]["score"] <= 1.0
    # The order must be descending.
    assert [p["score"] for p in predictions] == sorted(
        [p["score"] for p in predictions], reverse=True
    )


def test_predict_requires_both_model_and_classes():
    class _FakeModel:
        def __call__(self, x):
            import torch

            return torch.tensor([[0.7, 0.3]])

    with pytest.raises(ValueError):
        predict(_leaf_image_bytes(), model=_FakeModel())


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="the ONNX export is not in git")
def test_predict_onnx_returns_a_valid_ranking():
    predictions = predict_onnx(_leaf_image_bytes(), top_k=2)
    assert len(predictions) == 2
    assert predictions[0]["label"] in load_class_names()
    assert predictions[0]["score"] >= predictions[1]["score"]


def test_predict_onnx_without_a_model_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(classifier, "_ONNX_DIR", tmp_path)
    monkeypatch.setattr(classifier, "_ONNX_SESSION", None)
    with pytest.raises(ModelUnavailable):
        predict_onnx(_leaf_image_bytes())


# ---------- the leaf check ----------

def test_leaf_check_accepts_a_leaf_photo():
    assert looks_like_leaf(_leaf_image_bytes()) is True


def test_leaf_check_rejects_a_flat_colour():
    img = Image.new("RGB", (400, 300), color=(120, 90, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    assert looks_like_leaf(buf.getvalue()) is False


def test_leaf_check_rejects_a_blue_sky():
    img = Image.new("RGB", (400, 300), color=(90, 140, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    assert looks_like_leaf(buf.getvalue()) is False


def test_leaf_check_rejects_noise():
    import random

    rng = random.Random(0)
    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
            )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    assert looks_like_leaf(buf.getvalue()) is False


def test_leaf_check_rejects_something_unreadable():
    assert looks_like_leaf(b"not an image") is False
