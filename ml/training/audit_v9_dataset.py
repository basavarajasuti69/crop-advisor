"""V9 dataset audit: mislabeled-image review + leakage audit.

Read-only with respect to every dataset. Writes a review report and a
leakage report into a new audit directory. No image is moved, renamed or
deleted; no label is changed. Every candidate is a suggestion for a
human reviewer.

Outputs (under --out):
  label_review_report.csv   one row per candidate image
  leakage_audit.json        exact-hash and perceptual near-duplicates
  v9_cleaning_manifest.csv  manifest a human edits to approve removals
  AUDIT.md                 summary

Usage:
  python ml/training/audit_v9_dataset.py \\
      --data ml/training/experiments/plantdoc_deduplicated_v2/data \\
      --out  ml/training/experiments/plantdoc_v9_audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Disease-specific filename tokens -> the class they belong to.
TOKENS = {
    "early_blight": ["early blight", "early_blight", "earlyblight", "early-blight"],
    "late_blight": ["late blight", "late_blight", "lateblight", "late-blight"],
    "septoria": ["septoria"],
    "bacterial_spot": ["bacterial spot", "bacterial_spot", "bacterial-spot", "bacspot"],
    "mosaic_virus": ["mosaic virus", "mosaic_virus", "mosaicvirus", "tmv", "tobacco mosaic"],
    "yellow_leaf_curl": ["tylc", "yellow leaf curl", "yellow_leaf_curl", "yellowleafcurl"],
    "leaf_mold": ["leaf mold", "leaf_mold", "leafmold", "passalora", "fulvia"],
    "gray_mold_botrytis": ["botrytis", "grey mold", "gray mold", "grey-mould", "gray-mould"],
    "rust": ["rust"],
    "powdery_mildew": ["powdery mildew", "powdery_mildew", "powderymildew"],
    "black_rot": ["black rot", "black_rot", "blackrot"],
    "gray_leaf_spot": ["gray leaf spot", "grey leaf spot", "gray_leaf_spot",
                       "grayleafspot", "greyleafspot"],
    "scab": ["scab"],
    "spider_mites": ["spider mite", "spider_mites", "two spotted", "twospotted"],
    "edema": ["edema"],
    "leaf_roll": ["leaf roll", "leaf_roll", "leafroll"],
    "healthy": ["healthy"],
}

CLASS_OF_TOKEN = {
    "early_blight": "Tomato Early blight leaf",
    "late_blight": "Tomato leaf late blight",
    "septoria": "Tomato Septoria leaf spot",
    "bacterial_spot": "Tomato leaf bacterial spot",
    "mosaic_virus": "Tomato leaf mosaic virus",
    "yellow_leaf_curl": "Tomato leaf yellow virus",
    "leaf_mold": "Tomato mold leaf",
    "gray_mold_botrytis": "Tomato mold leaf",
    "powdery_mildew": "Squash Powdery mildew leaf",
    "black_rot": "grape leaf black rot",
    "gray_leaf_spot": "Corn Gray leaf spot",
    "scab": "Apple Scab Leaf",
    "spider_mites": "Tomato two spotted spider mites leaf",
}

# Classes the user cares about for V9.
PRIORITY_CROPS = {"corn", "potato", "tomato"}
PRIORITY_CLASSES = {
    "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf",
    "Potato leaf early blight", "Potato leaf late blight",
    "Tomato Early blight leaf", "Tomato Septoria leaf spot", "Tomato leaf",
    "Tomato leaf bacterial spot", "Tomato leaf late blight",
    "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato mold leaf",
}


def tokens_in(name: str) -> set[str]:
    import re
    n = name.lower().replace("%20", " ").replace("+", " ").replace("_", " ")
    n = re.sub(r"[^a-z0-9]+", " ", n)
    found = set()
    for key, pats in TOKENS.items():
        for p in pats:
            p2 = re.sub(r"[^a-z0-9]+", " ", p)
            if p2 in n:
                found.add(key)
                break
    return found


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels(path: Path) -> int:
    try:
        with Image.open(path) as h:
            return h.size[0] * h.size[1]
    except Exception:
        return 0


def phash(path: Path, size: int = 16) -> str | None:
    """A tiny perceptual hash: resize to size x size, threshold on mean."""
    try:
        with Image.open(path) as h:
            gray = h.convert("L").resize((size, size))
            vals = list(gray.getdata())
            mean = sum(vals) / len(vals)
            bits = "".join("1" if v > mean else "0" for v in vals)
            return bits
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/training/experiments/plantdoc_deduplicated_v2/data")
    parser.add_argument("--out", default="ml/training/experiments/plantdoc_v9_audit")
    parser.add_argument("--phash-threshold", type=int, default=8,
                        help="Hamming distance <= this counts as near-duplicate.")
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_dir = data / "train"
    test_dir = data / "test"

    # ------------------------------------------------------------------
    # 1. Label review candidates (corn / potato / tomato priority)
    # ------------------------------------------------------------------
    rows = []
    for split, root in (("train", train_dir), ("test", test_dir)):
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(folder.iterdir()):
                if not f.is_file() or f.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                found = tokens_in(f.name)
                for key in found:
                    target = CLASS_OF_TOKEN.get(key)
                    if (target and target != folder.name
                            and target in {d.name for d in root.iterdir() if d.is_dir()}):
                        # Only report priority crops.
                        if folder.name.split()[0].lower() not in {"corn", "potato", "tomato"} \
                                and not folder.name.lower().startswith("tomato"):
                            # still report if the suggested class is priority
                            if target not in PRIORITY_CLASSES:
                                continue
                        rows.append({
                            "split": split,
                            "current_label": folder.name,
                            "file": str(f.relative_to(data)),
                            "absolute_path": str(f.resolve()),
                            "filename_token": key,
                            "suggested_label": target,
                            "priority_crop": "yes" if folder.name.split()[0].lower() in PRIORITY_CROPS
                                             or folder.name.lower().startswith("tomato") else "no",
                            "reason": (f"filename states '{key}' but file is filed under "
                                       f"'{folder.name}'"),
                            "action": "MANUAL_REVIEW",
                        })

    # Deduplicate by absolute path (a file can match multiple tokens).
    seen = set()
    uniq = []
    for r in rows:
        if r["absolute_path"] in seen:
            continue
        seen.add(r["absolute_path"])
        uniq.append(r)
    rows = uniq

    # Sort: priority crops first, then by current label.
    rows.sort(key=lambda r: (r["priority_crop"] != "yes", r["current_label"], r["file"]))

    csv_path = out / "label_review_report.csv"
    with csv_path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------
    # 2. Leakage audit: exact hashes + perceptual near-duplicates
    # ------------------------------------------------------------------
    def collect(split_root):
        out_list = []
        for folder in sorted(p for p in split_root.iterdir() if p.is_dir()):
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                    out_list.append((folder.name, f))
        return out_list

    train_items = collect(train_dir)
    test_items = collect(test_dir)

    # Exact hash overlap train<->test
    train_hashes = {sha256_file(f): (label, f) for label, f in train_items}
    test_hashes = {sha256_file(f): (label, f) for label, f in test_items}
    exact_overlap = []
    for hsh in set(train_hashes) & set(test_hashes):
        tl, tf = train_hashes[hsh]
        el, ef = test_hashes[hsh]
        exact_overlap.append({
            "hash": hsh,
            "train_label": tl,
            "train_file": str(tf.relative_to(data)),
            "test_label": el,
            "test_file": str(ef.relative_to(data)),
        })

    # Exact duplicates within train
    within_train = defaultdict(list)
    for label, f in train_items:
        within_train[sha256_file(f)].append((label, f))
    train_dup_groups = [
        [{"label": lbl, "file": str(f.relative_to(data))} for lbl, f in v]
        for v in within_train.values() if len(v) > 1
    ]

    # Exact duplicates within test
    within_test = defaultdict(list)
    for label, f in test_items:
        within_test[sha256_file(f)].append((label, f))
    test_dup_groups = [
        [{"label": lbl, "file": str(f.relative_to(data))} for lbl, f in v]
        for v in within_test.values() if len(v) > 1
    ]

    # Perceptual near-duplicates across splits
    train_ph = [(label, f, phash(f)) for label, f in train_items]
    test_ph = [(label, f, phash(f)) for label, f in test_items]
    near_dup = []
    for tlabel, tf, th in train_ph:
        if th is None:
            continue
        for elabel, ef, eh in test_ph:
            if eh is None:
                continue
            dist = hamming(th, eh)
            if dist <= args.phash_threshold:
                near_dup.append({
                    "hamming_distance": dist,
                    "train_label": tlabel,
                    "train_file": str(tf.relative_to(data)),
                    "test_label": elabel,
                    "test_file": str(ef.relative_to(data)),
                })
    near_dup.sort(key=lambda r: r["hamming_distance"])

    # Source/capture-session grouping by filename stem (strip trailing numbers).
    import re
    def session_key(name: str) -> str:
        stem = Path(name).stem.lower()
        # strip trailing digits / common counters
        return re.sub(r"[\d_\-]+$", "", stem).strip("_-")[:40]

    # Group test images by session key to find same-source clusters.
    test_sessions = defaultdict(list)
    for label, f in test_items:
        test_sessions[session_key(f.name)].append((label, str(f.relative_to(data))))
    multi_session = {k: v for k, v in test_sessions.items() if len(v) > 1}

    leakage = {
        "exact_hash_train_test_overlap": len(exact_overlap),
        "exact_hash_train_test_pairs": exact_overlap,
        "exact_duplicates_within_train_groups": len(train_dup_groups),
        "exact_duplicates_within_train": train_dup_groups,
        "exact_duplicates_within_test_groups": len(test_dup_groups),
        "exact_duplicates_within_test": test_dup_groups,
        "perceptual_near_duplicates_across_splits": len(near_dup),
        "perceptual_near_duplicate_pairs": near_dup[:50],
        "test_images_grouped_by_filename_stem_clusters": len(multi_session),
        "note": ("Near-duplicate pairs are candidates only; a low Hamming "
                 "distance can be a real similarity or a re-saved copy. "
                 "Verify visually before removing anything."),
    }
    (out / "leakage_audit.json").write_text(json.dumps(leakage, indent=2) + "\n")

    # ------------------------------------------------------------------
    # 3. V9 cleaning manifest (human edits this to approve removals)
    # ------------------------------------------------------------------
    manifest_rows = []
    for r in rows:
        manifest_rows.append({
            "split": r["split"],
            "current_label": r["current_label"],
            "file": r["file"],
            "suggested_label": r["suggested_label"],
            "reason": r["reason"],
            "reviewer_decision": "PENDING",   # PENDING / REMOVE / KEEP / RELABEL
            "reviewer_new_label": "",
            "reviewer_notes": "",
        })
    # Add the known byte-identical duplicate pair.
    manifest_rows.append({
        "split": "train",
        "current_label": "Potato leaf early blight",
        "file": "train/Potato leaf early blight/18028_1.jpg",
        "suggested_label": "Potato leaf late blight",
        "reason": ("byte-identical to Potato leaf late blight/24064_1.jpg; "
                   "the same image under two labels"),
        "reviewer_decision": "PENDING",
        "reviewer_new_label": "",
        "reviewer_notes": "",
    })
    manifest_rows.append({
        "split": "train",
        "current_label": "Potato leaf late blight",
        "file": "train/Potato leaf late blight/24064_1.jpg",
        "suggested_label": "Potato leaf early blight",
        "reason": ("byte-identical to Potato leaf early blight/18028_1.jpg; "
                   "the same image under two labels"),
        "reviewer_decision": "PENDING",
        "reviewer_new_label": "",
        "reviewer_notes": "",
    })

    # Deduplicate manifest by file.
    seenf = set()
    uniqm = []
    for r in manifest_rows:
        if r["file"] in seenf:
            continue
        seenf.add(r["file"])
        uniqm.append(r)
    manifest_rows = uniqm

    mpath = out / "v9_cleaning_manifest.csv"
    with mpath.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # ------------------------------------------------------------------
    # 4. AUDIT.md summary
    # ------------------------------------------------------------------
    priority_rows = [r for r in rows if r["priority_crop"] == "yes"]
    # corn is "Corn", potato "Potato", tomato "Tomato"
    crop_counts = Counter()
    for r in priority_rows:
        cl = r["current_label"].lower()
        if cl.startswith("corn"):
            crop_counts["corn"] += 1
        elif cl.startswith("potato"):
            crop_counts["potato"] += 1
        elif cl.startswith("tomato"):
            crop_counts["tomato"] += 1

    md = f"""# V9 dataset audit

Generated by `ml/training/audit_v9_dataset.py` from
`{data}`.

This is a **review document**, not a cleaning action. No image was
moved, renamed or deleted and no label was changed. Every row below is a
candidate for a human reviewer to confirm or reject.

## 1. Label-review candidates

Total candidates: **{len(rows)}** (unique files: {len(seen)}).
Priority-crop candidates (corn / potato / tomato): **{len(priority_rows)}**.

| Crop | Candidates |
|---|---|
| corn | {crop_counts['corn']} |
| potato | {crop_counts['potato']} |
| tomato | {crop_counts['tomato']} |

Top confusion directions among priority crops:

"""
    top = Counter((r["current_label"], r["suggested_label"]) for r in priority_rows)
    for (a, b), n in top.most_common(15):
        md += f"- `{a}` -> `{b}`: {n}\n"

    md += f"""
See `label_review_report.csv` for the full list (file paths, current
label, suggested label, reason).

## 2. Leakage audit

- Exact-hash train/test overlap: **{leakage['exact_hash_train_test_overlap']}** pairs
- Exact duplicates within train: **{leakage['exact_duplicates_within_train_groups']}** groups
- Exact duplicates within test: **{leakage['exact_duplicates_within_test_groups']}** groups
- Perceptual near-duplicates across splits (Hamming <= {args.phash_threshold}): **{leakage['perceptual_near_duplicates_across_splits']}** pairs
- Test images sharing a filename stem cluster: **{leakage['test_images_grouped_by_filename_stem_clusters']}** clusters

See `leakage_audit.json` for the pairs.

## 3. How to produce the cleaned V9 dataset

1. A human reviewer edits `v9_cleaning_manifest.csv` and sets
   `reviewer_decision` to one of `REMOVE`, `KEEP`, `RELABEL` (with
   `reviewer_new_label` filled in for RELABEL).
2. `ml/training/build_v9_dataset.py --manifest <edited csv> --out ...`
   copies the approved subset into a new directory, leaving the
   original dataset untouched.
3. Retrain with `ml/training/train_plantdoc.py --data <cleaned> ...`.

The cleaning is reproducible from the manifest: anyone can rebuild the
exact same dataset by re-running the builder on the same manifest.

## 4. Limitations

- Filename-based review is a *hint*, not proof. Every candidate needs a
  visual check before removal.
- The perceptual hash is coarse (16x16); near-duplicate pairs are
  candidates, not findings.
- This audit does not add new images or collect new data; classes with
  few examples stay small.
"""
    (out / "AUDIT.md").write_text(md)

    print(f"Wrote {csv_path}  ({len(rows)} candidates)")
    print(f"Wrote {out/'leakage_audit.json'}")
    print(f"Wrote {mpath}  ({len(manifest_rows)} manifest rows)")
    print(f"Wrote {out/'AUDIT.md'}")
    print(f"Priority-crop candidates: {len(priority_rows)}")
    print(f"Exact train/test overlap: {leakage['exact_hash_train_test_overlap']}")
    print(f"Perceptual near-dups across splits: {leakage['perceptual_near_duplicates_across_splits']}")


if __name__ == "__main__":
    main()
