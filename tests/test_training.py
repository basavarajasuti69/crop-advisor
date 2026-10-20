from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from ml.training.train_plantdoc import MappedSubset  # noqa: E402


class _FakeImageFolder:
    """Stands in for an ImageFolder: a list of (path, label) samples."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return f"image:{path}", label


def _dataset():
    # The excluded class sits in the middle, not last - this is the case that
    # broke the old "filter the samples, then reuse the indices" approach.
    return _FakeImageFolder(
        [
            ("/img/healthy_0.jpg", 0),
            ("/img/healthy_1.jpg", 0),
            ("/img/spider_mites.jpg", 2),
            ("/img/rust_0.jpg", 1),
            ("/img/rust_1.jpg", 1),
            ("/img/blight_0.jpg", 3),
        ]
    )


def test_mapped_subset_keeps_the_original_samples():
    dataset = _dataset()
    before = list(dataset.samples)
    subset = MappedSubset(dataset, [3, 4, 5], {0: 0, 1: 1, 3: 2})
    assert len(subset) == 3
    # The dataset is not mutated, so the indices still refer to the same rows.
    assert dataset.samples == before


def test_mapped_subset_maps_the_label():
    subset = MappedSubset(_dataset(), [3, 4, 5], {0: 0, 1: 1, 3: 2})
    labels = [subset[position][1] for position in range(len(subset))]
    # rust (original label 1) -> shared index 1, blight (3) -> shared index 2
    assert labels == [1, 1, 2]


def test_filtering_the_samples_would_shift_the_indices():
    """Why the old approach was wrong: dropping a class renumbers the rest."""
    dataset = _dataset()
    filtered = [(p, l) for p, l in dataset.samples if l != 2]
    # Original index 3 is rust_0, but after filtering it is rust_1.
    assert dataset.samples[3][0] == "/img/rust_0.jpg"
    assert filtered[3][0] == "/img/rust_1.jpg"


def test_mapped_subset_reads_the_right_sample():
    subset = MappedSubset(_dataset(), [3, 4, 5], {0: 0, 1: 1, 3: 2})
    images = [subset[position][0] for position in range(len(subset))]
    assert images == [
        "image:/img/rust_0.jpg",
        "image:/img/rust_1.jpg",
        "image:/img/blight_0.jpg",
    ]


def test_mapped_subset_needs_a_label_for_every_index_it_reads():
    subset = MappedSubset(_dataset(), [3], {1: 1})
    assert subset[0][1] == 1
    with pytest.raises(KeyError):
        # Label 3 is not in the map, so this sample cannot be mapped.
        MappedSubset(_dataset(), [5], {1: 1})[0]
