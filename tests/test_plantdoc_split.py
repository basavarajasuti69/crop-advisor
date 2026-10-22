"""Tests for the PlantDoc training split and its metrics helpers.

The split is the thing that has to stay exactly the same between a
training run and any later evaluation of its checkpoint, so it is
pinned here rather than left to a comment in the training script.

Run with::

    python -m pytest tests/test_plantdoc_split.py -q
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml/training"))

from train_plantdoc import (  # noqa: E402
    EXCLUDED_CLASS,
    MappedSubset,
    build_split,
    per_class_metrics,
    resolve_classes,
)


def _fake_samples(class_counts: dict[str, int]):
    """Stand-in for ImageFolder.samples: (path, class index) pairs."""
    classes = sorted(class_counts)
    samples = []
    for index, name in enumerate(classes):
        for number in range(class_counts[name]):
            samples.append((f"/img/{name}/{number}.jpg", index))
    return samples, classes


# ---------- resolve_classes ----------

def test_resolve_classes_is_the_sorted_intersection():
    train = ["b", "a", "c", "d"]
    test = ["c", "a", "b", "z"]
    assert resolve_classes(train, test) == ["a", "b", "c"]


def test_resolve_classes_drops_the_excluded_class():
    train = ["a", EXCLUDED_CLASS]
    test = ["a", EXCLUDED_CLASS]
    assert resolve_classes(train, test) == ["a"]
    assert EXCLUDED_CLASS not in resolve_classes(train, test)


def test_resolve_classes_defaults_to_the_plantdoc_exclusion():
    assert EXCLUDED_CLASS == "Tomato two spotted spider mites leaf"


# ---------- build_split ----------

def test_build_split_is_deterministic_for_a_given_seed():
    samples, classes = _fake_samples({"a": 20, "b": 30})
    first = build_split(samples, classes, samples, classes, classes)
    second = build_split(samples, classes, samples, classes, classes)
    assert first["train_indices"] == second["train_indices"]
    assert first["validation_indices"] == second["validation_indices"]
    assert first["test_indices"] == second["test_indices"]


def test_build_split_changes_with_the_seed():
    samples, classes = _fake_samples({"a": 40, "b": 40})
    first = build_split(samples, classes, samples, classes, classes, seed=42)
    second = build_split(samples, classes, samples, classes, classes, seed=7)
    assert first["validation_indices"] != second["validation_indices"]


def test_train_and_validation_are_disjoint_and_cover_every_sample():
    samples, classes = _fake_samples({"a": 23, "b": 31, "c": 9})
    split = build_split(samples, classes, samples, classes, classes)
    train = set(split["train_indices"])
    validation = set(split["validation_indices"])
    assert train & validation == set()
    assert train | validation == set(range(len(samples)))


def test_validation_takes_about_fifteen_percent_of_each_class():
    samples, classes = _fake_samples({"a": 100, "b": 40})
    split = build_split(samples, classes, samples, classes, classes)
    labels = [samples[i][1] for i in split["validation_indices"]]
    counts = Counter(labels)
    assert counts[0] == 15
    assert counts[1] == 6


def test_every_class_keeps_at_least_one_validation_image():
    samples, classes = _fake_samples({"a": 1, "b": 2})
    split = build_split(samples, classes, samples, classes, classes)
    # "a" has 1 image (all of it validation) and "b" has 2 (1 of each).
    assert len(split["validation_indices"]) == 2
    assert len(split["train_indices"]) == 1


def test_the_split_is_per_class_stratified():
    """Every class appears in both the train and the validation half."""
    samples, classes = _fake_samples({"a": 30, "b": 30, "c": 30})
    split = build_split(samples, classes, samples, classes, classes)
    train_labels = {samples[i][1] for i in split["train_indices"]}
    validation_labels = {samples[i][1] for i in split["validation_indices"]}
    assert train_labels == {0, 1, 2}
    assert validation_labels == {0, 1, 2}


def test_the_label_maps_point_at_the_shared_class_order():
    train_samples, train_classes = _fake_samples({"b": 10, "a": 10})
    test_samples, test_classes = _fake_samples({"a": 5, "b": 5})
    classes = resolve_classes(train_classes, test_classes)
    assert classes == ["a", "b"]

    split = build_split(
        train_samples, train_classes, test_samples, test_classes, classes
    )
    # train_classes is ["a", "b"], so label 0 is "a" -> shared index 0.
    assert split["class_to_new"] == {0: 0, 1: 1}
    assert split["test_class_to_new"] == {0: 0, 1: 1}


# ---------- MappedSubset ----------

class _StubDataset:
    def __init__(self, samples):
        self.samples = samples
        self.classes = sorted({name for name, _ in samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return f"image:{path}", label


def test_mapped_subset_remaps_labels_without_shifting_the_images():
    dataset = _StubDataset(
        [("/img/a.jpg", 0), ("/img/b.jpg", 1), ("/img/c.jpg", 2)]
    )
    subset = MappedSubset(dataset, [2, 0], {0: 5, 2: 9})
    assert subset[0] == ("image:/img/c.jpg", 9)
    assert subset[1] == ("image:/img/a.jpg", 5)
    assert len(subset) == 2


def test_mapped_subset_exposes_the_samples_it_was_built_from():
    dataset = _StubDataset(
        [("/img/a.jpg", 0), ("/img/b.jpg", 1), ("/img/c.jpg", 2)]
    )
    subset = MappedSubset(dataset, [2, 0], {0: 5, 2: 9})
    assert subset.samples == [
        ("/img/c.jpg", 9),
        ("/img/a.jpg", 5),
    ]


# ---------- per_class_metrics ----------

def test_per_class_metrics_matches_a_hand_computed_example():
    # "a" is the true label of 4 images, 3 of them predicted correctly.
    # "b" is predicted 3 times: twice correctly and once wrongly.
    confusion = Counter({
        ("a", "a"): 3,
        ("a", "b"): 1,
        ("b", "b"): 2,
    })
    metrics = per_class_metrics(confusion, ["a", "b"])

    assert metrics["a"]["support"] == 4
    assert metrics["a"]["precision"] == pytest.approx(1.0)
    assert metrics["a"]["recall"] == pytest.approx(0.75)
    assert metrics["a"]["f1"] == pytest.approx(6 / 7)

    assert metrics["b"]["support"] == 2
    assert metrics["b"]["precision"] == pytest.approx(2 / 3)
    assert metrics["b"]["recall"] == pytest.approx(1.0)
    assert metrics["b"]["f1"] == pytest.approx(0.8)


def test_per_class_metrics_handles_a_class_with_no_support():
    confusion = Counter({("a", "a"): 1})
    metrics = per_class_metrics(confusion, ["a", "b"])
    assert metrics["b"]["support"] == 0
    assert metrics["b"]["precision"] == 0.0
    assert metrics["b"]["recall"] == 0.0
    assert metrics["b"]["f1"] == 0.0
