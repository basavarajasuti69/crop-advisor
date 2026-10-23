"""Tests for the V9 dataset-cleaning builder and review rules.

Run with::

    python -m pytest tests/test_v9_builder.py -q
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "ml/training/build_v9_dataset.py"


def _make_dataset(root: Path) -> Path:
    src = root / "src"
    (src / "train" / "A").mkdir(parents=True, exist_ok=True)
    (src / "train" / "B").mkdir(parents=True, exist_ok=True)
    (src / "test" / "A").mkdir(parents=True, exist_ok=True)
    (src / "train" / "A" / "a.jpg").write_bytes(b"img-a")
    (src / "train" / "B" / "b.jpg").write_bytes(b"img-b")
    (src / "test" / "A" / "t.jpg").write_bytes(b"img-t")
    return src


def _write_manifest(path: Path, rows: list[dict]) -> None:
    cols = ["split", "current_label", "file", "reviewer_decision", "reviewer_new_label"]
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _run(src, manifest, out, extra=()):
    return subprocess.run(
        [sys.executable, str(BUILDER), "--source", str(src),
         "--manifest", str(manifest), "--out", str(out), *extra],
        capture_output=True, text=True,
    )


def test_rejects_invalid_decision(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [{"split": "train", "current_label": "A",
                         "file": "train/A/a.jpg", "reviewer_decision": "MAYBE",
                         "reviewer_new_label": ""}])
    r = _run(src, m, tmp_path / "out")
    assert r.returncode != 0
    assert "Invalid reviewer_decision" in (r.stdout + r.stderr)


def test_rejects_relabel_without_target(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [{"split": "train", "current_label": "A",
                         "file": "train/A/a.jpg", "reviewer_decision": "RELABEL",
                         "reviewer_new_label": ""}])
    r = _run(src, m, tmp_path / "out")
    assert r.returncode != 0
    assert "RELABEL without reviewer_new_label" in (r.stdout + r.stderr)


def test_rejects_relabel_to_nonexistent_class(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [{"split": "train", "current_label": "A",
                         "file": "train/A/a.jpg", "reviewer_decision": "RELABEL",
                         "reviewer_new_label": "Z"}])
    r = _run(src, m, tmp_path / "out")
    assert r.returncode != 0
    assert "is not an existing class folder" in (r.stdout + r.stderr)


def test_strict_rejects_pending(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [{"split": "train", "current_label": "A",
                         "file": "train/A/a.jpg", "reviewer_decision": "PENDING",
                         "reviewer_new_label": ""}])
    r = _run(src, m, tmp_path / "out", extra=["--strict"])
    assert r.returncode != 0
    assert "Unresolved PENDING" in (r.stdout + r.stderr)


def test_remove_excludes_image(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [{"split": "train", "current_label": "A",
                         "file": "train/A/a.jpg", "reviewer_decision": "REMOVE",
                         "reviewer_new_label": ""}])
    out = tmp_path / "out"
    r = _run(src, m, out)
    assert r.returncode == 0, r.stderr
    assert not (out / "train" / "A" / "a.jpg").exists()
    assert (out / "train" / "B" / "b.jpg").exists()
    assert (out / "test" / "A" / "t.jpg").exists()


def test_relabel_moves_to_existing_target(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [{"split": "train", "current_label": "A",
                         "file": "train/A/a.jpg", "reviewer_decision": "RELABEL",
                         "reviewer_new_label": "B"}])
    out = tmp_path / "out"
    r = _run(src, m, out)
    assert r.returncode == 0, r.stderr
    assert not (out / "train" / "A" / "a.jpg").exists()
    assert (out / "train" / "B" / "a.jpg").exists()


def test_unlisted_files_are_kept(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [])  # empty manifest
    out = tmp_path / "out"
    r = _run(src, m, out)
    assert r.returncode == 0, r.stderr
    assert (out / "train" / "A" / "a.jpg").exists()
    assert (out / "train" / "B" / "b.jpg").exists()
    assert (out / "test" / "A" / "t.jpg").exists()


def test_source_dataset_is_not_modified(tmp_path):
    src = _make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    m = tmp_path / "m.csv"
    _write_manifest(m, [{"split": "train", "current_label": "A",
                         "file": "train/A/a.jpg", "reviewer_decision": "REMOVE",
                         "reviewer_new_label": ""}])
    _run(src, m, tmp_path / "out")
    after = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    assert before == after


def test_dry_run_does_not_create_output(tmp_path):
    src = _make_dataset(tmp_path)
    m = tmp_path / "m.csv"
    _write_manifest(m, [])
    out = tmp_path / "out"
    r = _run(src, m, out, extra=["--dry-run"])
    assert r.returncode == 0, r.stderr
    assert not out.exists()
