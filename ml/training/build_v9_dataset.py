"""Build a cleaned PlantDoc dataset from a reviewed V9 cleaning manifest.

Reads a manifest CSV produced by `audit_v9_dataset.py` (and edited by a
human reviewer), copies the approved subset into a new output directory,
and leaves the source dataset untouched. The cleaning is reproducible
from the manifest alone.

Manifest columns:
  split, current_label, file, suggested_label, reason,
  reviewer_decision (PENDING/REMOVE/KEEP/RELABEL),
  reviewer_new_label, reviewer_notes

Rules:
  - Rows with reviewer_decision == REMOVE are excluded.
  - Rows with reviewer_decision == RELABEL are copied into the folder
    named by reviewer_new_label (must be an existing class folder).
  - Rows with reviewer_decision == KEEP or PENDING are kept as-is.
  - Files not listed in the manifest are always kept.

Usage:
  python ml/training/build_v9_dataset.py \\
      --source ml/training/experiments/plantdoc_deduplicated_v2/data \\
      --manifest ml/training/experiments/plantdoc_v9_audit/v9_cleaning_manifest.csv \\
      --out ml/training/experiments/plantdoc_deduplicated_v4/data
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source dataset root (train/, test/)")
    parser.add_argument("--manifest", required=True, help="reviewed manifest CSV")
    parser.add_argument("--out", required=True, help="output dataset root")
    parser.add_argument("--dry-run", action="store_true", help="print actions, copy nothing")
    parser.add_argument("--strict", action="store_true",
                        help="reject PENDING or unresolved decisions (use for a reviewed manifest)")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if not (source / "train").is_dir() or not (source / "test").is_dir():
        raise SystemExit(f"{source} must contain train/ and test/")

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output is not empty: {out}. Choose a new --out.")

    # Load manifest decisions, validating the schema and values.
    VALID_DECISIONS = {"PENDING", "KEEP", "REMOVE", "RELABEL"}
    decisions: dict[str, dict] = {}
    required_cols = {"split", "current_label", "file", "reviewer_decision",
                     "reviewer_new_label"}
    with Path(args.manifest).open(newline="") as h:
        reader = csv.DictReader(h)
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise SystemExit(
                f"Manifest is missing required columns: "
                f"{sorted(required_cols - set(reader.fieldnames or []))}"
            )
        for row in reader:
            if not row.get("file"):
                raise SystemExit("Manifest contains a row with an empty 'file'.")
            action = (row.get("reviewer_decision") or "KEEP").upper()
            if action not in VALID_DECISIONS:
                raise SystemExit(
                    f"Invalid reviewer_decision '{row.get('reviewer_decision')}' "
                    f"for {row['file']}; expected one of {sorted(VALID_DECISIONS)}."
                )
            if action == "RELABEL" and not (row.get("reviewer_new_label") or "").strip():
                raise SystemExit(f"RELABEL without reviewer_new_label: {row['file']}")
            if args.strict and action in ("PENDING",):
                raise SystemExit(
                    f"Unresolved PENDING decision in strict mode: {row['file']}"
                )
            decisions[row["file"]] = row

    removed = 0
    relabeled = 0
    kept = 0
    for split in ("train", "test"):
        src_split = source / split
        dst_split = out / split
        if not args.dry_run:
            dst_split.mkdir(parents=True, exist_ok=True)
        for class_dir in sorted(p for p in src_split.iterdir() if p.is_dir()):
            dst_class = dst_split / class_dir.name
            if not args.dry_run:
                dst_class.mkdir(parents=True, exist_ok=True)
            for f in sorted(class_dir.iterdir()):
                if not f.is_file():
                    continue
                rel = f"{split}/{class_dir.name}/{f.name}"
                dec = decisions.get(rel, {})
                action = (dec.get("reviewer_decision") or "KEEP").upper()
                if action == "REMOVE":
                    removed += 1
                    if not args.dry_run:
                        pass  # skip copy
                    print(f"REMOVE  {rel}")
                    continue
                if action == "RELABEL":
                    new_label = (dec.get("reviewer_new_label") or "").strip()
                    if not new_label:
                        raise SystemExit(f"RELABEL without reviewer_new_label: {rel}")
                    # The target must be an existing class folder in the source
                    # split. We never invent a new class from a free-text label.
                    if not (src_split / new_label).is_dir():
                        raise SystemExit(
                            f"RELABEL target '{new_label}' is not an existing "
                            f"class folder in {src_split}: {rel}"
                        )
                    target = dst_split / new_label
                    if not args.dry_run:
                        target.mkdir(parents=True, exist_ok=True)
                    relabeled += 1
                    if not args.dry_run:
                        shutil.copy2(f, target / f.name)
                    print(f"RELABEL {rel} -> {new_label}/{f.name}")
                    continue
                # KEEP or PENDING -> copy as-is
                kept += 1
                if not args.dry_run:
                    shutil.copy2(f, dst_class / f.name)

    print(f"\nSummary: kept={kept} removed={removed} relabeled={relabeled}")
    if args.dry_run:
        print("(dry run: nothing was copied)")
    else:
        print(f"Cleaned dataset written to {out}")


if __name__ == "__main__":
    main()
