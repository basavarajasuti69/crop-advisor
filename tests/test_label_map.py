"""Tests for the label mapping and the serving path around it.

The label map is the piece that decides which knowledge-base section a
prediction is allowed to reach, so it is tested directly rather than
only through the graph.

Run with::

    python -m pytest tests/test_label_map.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import label_map  # noqa: E402
from agents.classifier import (  # noqa: E402
    crop_from_label,
    disease_from_label,
    is_healthy_label,
    kb_section_for_label,
    load_model_metadata,
)

KB_INDEX = ROOT / "rag/index.jsonl"
SHIPPED_MANIFEST = ROOT / "ml/export_edge/artifacts/manifest.json"


# ---------- the label map itself ----------

def test_the_map_covers_the_27_plantdoc_classes():
    assert len(label_map.PLANTDOC_CLASSES) == 27


def test_every_class_has_a_crop_and_a_disease():
    for name, entry in label_map.PLANTDOC_CLASSES.items():
        assert entry["crop"], name
        assert entry["disease"], name
        assert isinstance(entry["healthy"], bool), name


def test_only_tomato_and_corn_reach_a_knowledge_base_section():
    with_section = {
        name: entry["kb"]
        for name, entry in label_map.PLANTDOC_CLASSES.items()
        if entry["kb"]
    }
    assert set(with_section) == {
        "Corn leaf blight",
        "Tomato Early blight leaf",
        "Tomato leaf",
        "Tomato leaf late blight",
    }
    assert set(with_section.values()) == {
        "Maize - Leaf Blight",
        "Tomato - Blight",
        "Tomato - Healthy",
    }


def test_every_referenced_section_exists_in_the_knowledge_base():
    """A mapping to a section that is not in the KB would retrieve nothing."""
    if not KB_INDEX.exists():
        pytest.skip("rag/index.jsonl is not built")
    text = KB_INDEX.read_text()
    for name, entry in label_map.PLANTDOC_CLASSES.items():
        section = entry["kb"]
        if not section:
            continue
        assert f"## {section}" in text, f"{name} -> {section} is not in the KB"


def test_the_ten_kb_sections_are_the_ones_in_the_source_document():
    source = ROOT / "rag/source_docs/crop_guide_en.md"
    if not source.exists():
        pytest.skip("knowledge base source is not present")
    headings = {
        line[3:].strip()
        for line in source.read_text().splitlines()
        if line.startswith("## ")
    }
    assert headings == set(label_map.KB_SECTIONS)


def test_the_knowledge_base_does_not_mention_rice_in_the_plantdoc_map():
    """Rice is in the knowledge base but not in the PlantDoc classes.

    A plantdoc label must never be routed to a rice section.
    """
    assert "rice" not in label_map.PLANTDOC_CROPS
    assert "Rice - Leaf Blight" not in {
        entry["kb"] for entry in label_map.PLANTDOC_CLASSES.values()
    }


# ---------- the classifier uses the map ----------

def test_crop_from_label_handles_the_plantdoc_scheme():
    assert crop_from_label("Tomato mold leaf") == "tomato"
    assert crop_from_label("Potato leaf early blight") == "potato"
    assert crop_from_label("Corn Gray leaf spot") == "corn"


def test_crop_from_label_still_handles_the_crop_specific_scheme():
    assert crop_from_label("rice_healthy") == "rice"
    assert crop_from_label("tomato_blight") == "tomato"
    assert crop_from_label("leaf_blight") is None


def test_crop_from_label_returns_none_for_an_unknown_label():
    assert crop_from_label("Not A Class") is None


def test_disease_from_label_uses_the_friendly_name():
    assert disease_from_label("Tomato mold leaf") == "leaf mold"
    assert disease_from_label("Tomato leaf bacterial spot") == "bacterial spot"
    assert disease_from_label("Squash Powdery mildew leaf") == "powdery mildew"


def test_disease_from_label_keeps_the_crop_specific_form():
    assert disease_from_label("rice_healthy") == "healthy"
    assert disease_from_label("healthy") == "healthy"


def test_is_healthy_label_for_both_schemes():
    assert is_healthy_label("Tomato leaf") is True
    assert is_healthy_label("Tomato mold leaf") is False
    assert is_healthy_label("healthy") is True
    assert is_healthy_label("rice_healthy") is True
    assert is_healthy_label("leaf_blight") is False


def test_kb_section_for_label():
    assert kb_section_for_label("Tomato leaf") == "Tomato - Healthy"
    assert kb_section_for_label("Corn leaf blight") == "Maize - Leaf Blight"
    # No potato section exists, so there must be no answer.
    assert kb_section_for_label("Potato leaf early blight") is None
    assert kb_section_for_label("Cherry leaf") is None
    assert kb_section_for_label("rice_healthy") is None


# ---------- the deployed model describes itself ----------

def test_the_shipped_manifest_describes_the_plantdoc_v8_model():
    """The shipped export is the 27-class PlantDoc V8 model.

    The manifest is the source of truth for what the deployed model
    actually is; the service reads it rather than hard-coding a dataset
    name.
    """
    manifest = json.loads(SHIPPED_MANIFEST.read_text())
    classes = json.loads((SHIPPED_MANIFEST.parent / "classes.json").read_text())["classes"]
    assert manifest["num_classes"] == 27
    assert manifest["num_classes"] == len(classes)
    assert manifest["classes"] == classes
    assert manifest.get("dataset") == "PlantDoc"
    assert "plantdoc" in (manifest.get("dataset") or "").lower()
    # Confidence is not calibrated until temperature scaling is performed.
    assert manifest.get("confidence_is_calibrated") is False
    # The class ordering must be consistent between manifest and classes.json.
    assert manifest["classes"] == classes


def test_load_model_metadata_reads_the_manifest():
    metadata = load_model_metadata()
    classes = json.loads((SHIPPED_MANIFEST.parent / "classes.json").read_text())["classes"]
    assert metadata["num_classes"] == 27
    assert metadata["classes"] == classes
    assert metadata["confidence_is_calibrated"] is False
    assert metadata["dataset"] == "PlantDoc"


def test_load_model_metadata_survives_a_missing_manifest(tmp_path):
    metadata = load_model_metadata(tmp_path)
    assert metadata["classes"] == ["healthy", "leaf_blight"]
    assert metadata["dataset"] is None
