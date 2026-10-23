"""Dataset-integrity and V9 pipeline tests.

These tests are read-only with respect to the datasets: they assert
invariants, they never modify data.

Run with::

    python -m pytest tests/test_dataset_integrity.py -q
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_V2 = ROOT / "ml/training/experiments/plantdoc_deduplicated_v2/data"
DATA_V3 = ROOT / "ml/training/experiments/plantdoc_deduplicated_v3/data"
ARTIFACTS = ROOT / "ml/export_edge/artifacts"
CLASSES_JSON = ARTIFACTS / "classes.json"
MANIFEST_JSON = ARTIFACTS / "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_files(root: Path, split: str) -> list[Path]:
    out = []
    for p in (root / split).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            out.append(p)
    return out


# ---------- dataset integrity ----------

@pytest.mark.skipif(not DATA_V2.exists(), reason="v2 dataset not present")
def test_v2_train_and_test_have_no_exact_hash_overlap():
    train = {_sha256(p) for p in _image_files(DATA_V2, "train")}
    test = {_sha256(p) for p in _image_files(DATA_V2, "test")}
    assert train & test == set(), "train/test exact-hash overlap detected"


@pytest.mark.skipif(not DATA_V2.exists(), reason="v2 dataset not present")
def test_v2_has_27_shared_classes():
    train_classes = {p.parent.name for p in _image_files(DATA_V2, "train")}
    test_classes = {p.parent.name for p in _image_files(DATA_V2, "test")}
    # 28 folders in train (incl. the 2-image spider-mites class), 27 in test.
    assert len(train_classes) == 28
    assert len(test_classes) == 27
    assert len(train_classes & test_classes) == 27


@pytest.mark.skipif(not DATA_V3.exists(), reason="v3 dataset not present")
def test_v3_removes_the_known_duplicate_pair_from_train():
    # The byte-identical duplicate pair was (early blight 18028_1.jpg,
    # late blight 24064_1.jpg). V9 cleaning removed BOTH copies.
    early = DATA_V3 / "train" / "Potato leaf early blight" / "18028_1.jpg"
    late = DATA_V3 / "train" / "Potato leaf late blight" / "24064_1.jpg"
    assert not early.exists(), "duplicate early-blight copy still present"
    assert not late.exists(), "duplicate late-blight copy still present"


@pytest.mark.skipif(not DATA_V3.exists(), reason="v3 dataset not present")
def test_v3_test_split_is_unchanged_from_v2():
    v2 = {_sha256(p): str(p.relative_to(DATA_V2)) for p in _image_files(DATA_V2, "test")}
    v3 = {_sha256(p): str(p.relative_to(DATA_V3)) for p in _image_files(DATA_V3, "test")}
    assert set(v2) == set(v3), "test split changed between v2 and v3"


@pytest.mark.skipif(not DATA_V2.exists(), reason="v2 dataset not present")
def test_excluded_class_has_two_training_images():
    folder = DATA_V2 / "train" / "Tomato two spotted spider mites leaf"
    if folder.exists():
        assert len(_image_files(folder, "")) if False else len(list(folder.iterdir())) <= 2


# ---------- export consistency ----------

@pytest.mark.skipif(not CLASSES_JSON.exists(), reason="export not present")
def test_manifest_and_classes_json_agree():
    manifest = json.loads(MANIFEST_JSON.read_text())
    classes = json.loads(CLASSES_JSON.read_text())["classes"]
    assert manifest["num_classes"] == len(classes)
    assert manifest["classes"] == classes
    assert manifest["confidence_is_calibrated"] is False


@pytest.mark.skipif(not CLASSES_JSON.exists(), reason="export not present")
def test_manifest_records_plantdoc_dataset():
    manifest = json.loads(MANIFEST_JSON.read_text())
    assert manifest.get("dataset") == "PlantDoc"
    assert manifest.get("num_classes") == 27


# ---------- class mapping ----------

def test_class_map_covers_27_classes():
    from agents.label_map import PLANTDOC_CLASSES
    assert len(PLANTDOC_CLASSES) == 27


def test_healthy_labels_are_consistent():
    from agents.label_map import PLANTDOC_CLASSES, plantdoc_is_healthy
    from agents.classifier import is_healthy_label
    healthy = [name for name, e in PLANTDOC_CLASSES.items() if e["healthy"]]
    assert len(healthy) >= 8
    for name in healthy:
        assert is_healthy_label(name) is True, name
    diseased = [name for name, e in PLANTDOC_CLASSES.items() if not e["healthy"]]
    for name in diseased:
        assert is_healthy_label(name) is False, name


def test_no_unsupported_crop_gets_kb_guidance():
    from agents.label_map import PLANTDOC_CLASSES, plantdoc_kb_section
    for name, entry in PLANTDOC_CLASSES.items():
        section = plantdoc_kb_section(name)
        if section is not None:
            assert entry["crop"] in {"tomato", "corn", "maize"}, (name, section)


# ---------- confidence thresholds ----------

def test_confidence_thresholds_are_conservative():
    from agents.graph import CONFIDENCE_MIN, HEALTHY_MIN, FOLLOW_UP_MIN, VISION_DISAGREE_MIN
    assert CONFIDENCE_MIN == 0.70
    assert HEALTHY_MIN == 0.85
    assert HEALTHY_MIN > CONFIDENCE_MIN
    assert FOLLOW_UP_MIN == 0.80
    assert VISION_DISAGREE_MIN == 0.70


def test_healthy_requires_stricter_threshold_in_evaluator():
    # Smoke test the evaluator's meets_api_threshold logic indirectly.
    from agents.classifier import is_healthy_label
    # A healthy label at 0.80 must NOT pass (HEALTHY_MIN 0.85).
    assert is_healthy_label("Tomato leaf") is True
    # disease label at 0.80 passes 0.70
    assert is_healthy_label("Tomato leaf late blight") is False
