"""Phase 3 - analyse the classes the PlantDoc models get wrong.

Read-only. Reads the comparison artifacts and the dataset, writes a
structured report and a set of contact sheets into a new directory.
Nothing in an experiment directory is touched.

What it answers, per hard class:

* how many images it has in train / validation / test;
* what each model recalls on it, and what it is confused with instead;
* whether the models that get it wrong are *confidently* wrong
  (a visual confusion) or *unsure* (the model knows it is guessing);
* whether the four models agree with each other on the class, which
  separates "this class is intrinsically hard" from "this checkpoint is
  bad".

Example
-------
    python ml/training/analyse_plantdoc_hard_classes.py \\
        --out plantdoc_hard_classes_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/training"))

from train_plantdoc import (  # noqa: E402
    build_split,
    resolve_classes,
)

EXPERIMENTS_DIR = ROOT / "ml/training/experiments"
DATA_DIR = EXPERIMENTS_DIR / "plantdoc_deduplicated_v2/data"

# The classes the mission asks about, in priority order.
HARD_CLASSES = [
    "Tomato mold leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf mosaic virus",
    "Cherry leaf",
    "Corn leaf blight",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Corn Gray leaf spot",
]

# Pairs that are expected to be visually close.
CONFUSION_PAIRS = [
    ("Corn leaf blight", "Corn Gray leaf spot"),
    ("Potato leaf early blight", "Potato leaf late blight"),
    ("Tomato leaf mosaic virus", "Tomato leaf yellow virus"),
    ("Tomato mold leaf", "Tomato leaf late blight"),
    ("Tomato leaf bacterial spot", "Tomato leaf late blight"),
    ("Tomato Early blight leaf", "Potato leaf early blight"),
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def pixels(folder: Path) -> list[int]:
    out = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                with Image.open(path) as handle:
                    out.append(handle.size[0] * handle.size[1])
            except Exception:
                pass
    return out


def contact_sheet(
    rows: list[dict],
    path: Path,
    cols: int = 4,
    tile: int = 210,
    caption_h: int = 54,
) -> None:
    """A grid of images with a caption under each one."""
    if not rows:
        return
    thumb = tile - 8
    rows_count = math.ceil(len(rows) / cols)
    width = cols * tile
    height = rows_count * (thumb + caption_h)
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, row in enumerate(rows):
        column = index % cols
        line = index // cols
        x = column * tile
        y = line * (thumb + caption_h)
        try:
            with Image.open(row["path"]) as handle:
                copy = handle.convert("RGB")
                copy.thumbnail((thumb, thumb))
                sheet.paste(copy, (x + (thumb - copy.width) // 2, y))
        except Exception:
            draw.rectangle(
                [x + 4, y + 4, x + thumb - 4, y + thumb - 4],
                outline=(200, 0, 0),
            )
        label = row.get("caption", "")
        # Wrap the caption onto two short lines.
        words = label.split()
        line_one: list = []
        line_two: list = []
        for word in words:
            if len(" ".join(line_one + [word])) <= 26:
                line_one.append(word)
            else:
                line_two.append(word)
        draw.text(
            (x + 4, y + thumb + 2),
            " ".join(line_one)[:34],
            fill=(0, 0, 0),
            font=font,
        )
        if line_two:
            draw.text(
                (x + 4, y + thumb + 18),
                " ".join(line_two)[:34],
                fill=(0, 0, 0),
                font=font,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        default="plantdoc_v4_v6_comparison_v2",
        help="Comparison directory under ml/training/experiments/.",
    )
    parser.add_argument(
        "--out",
        default="plantdoc_hard_classes_v1",
        help="Output directory under ml/training/experiments/.",
    )
    args = parser.parse_args()

    comparison = EXPERIMENTS_DIR / args.comparison
    out_dir = EXPERIMENTS_DIR / args.out
    if not comparison.is_dir():
        raise SystemExit(f"Comparison directory not found: {comparison}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {out_dir}. Choose a new --out."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(
        (comparison / "comparison_summary.json").read_text()
    )
    experiments = list(summary["experiments"].keys())
    classes = summary["classes"]

    # ---- class support -----------------------------------------------------
    train_folder = DATA_DIR / "train"
    test_folder = DATA_DIR / "test"

    train_counts = {
        folder.name: len([
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ])
        for folder in sorted(train_folder.iterdir()) if folder.is_dir()
    }
    test_counts = {
        folder.name: len([
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ])
        for folder in sorted(test_folder.iterdir()) if folder.is_dir()
    }

    # Rebuild the validation split with the training code.
    from torchvision.datasets import ImageFolder

    train_ds = ImageFolder(train_folder)
    test_ds = ImageFolder(test_folder)
    resolved = resolve_classes(train_ds.classes, test_ds.classes)
    split = build_split(
        train_ds.samples, train_ds.classes,
        test_ds.samples, test_ds.classes,
        resolved,
    )
    shared_label = split["shared_label"]
    validation_counts = Counter(
        shared_label[train_ds.classes[train_ds.samples[i][1]]]
        for i in split["validation_indices"]
    )
    train_split_counts = Counter(
        shared_label[train_ds.classes[train_ds.samples[i][1]]]
        for i in split["train_indices"]
    )

    support_rows = []
    for name in classes:
        index = shared_label[name]
        support_rows.append({
            "class": name,
            "train_total": train_counts.get(name, 0),
            "train_used": train_split_counts.get(index, 0),
            "validation": validation_counts.get(index, 0),
            "test": test_counts.get(name, 0),
            "train_to_test_ratio": (
                round(train_counts.get(name, 0) / test_counts.get(name, 0), 2)
                if test_counts.get(name) else ""
            ),
        })

    with (out_dir / "class_support.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(support_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(support_rows)

    # ---- per-model per-class metrics --------------------------------------
    per_model = {}
    for name in experiments:
        rows = read_csv(
            comparison / name / "per_class_metrics.csv"
        )
        per_model[name] = {row["class"]: row for row in rows}

    # ---- confusion matrices ------------------------------------------------
    confusions = {}
    for name in experiments:
        path = comparison / name / "confusion_matrix.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            grid = list(csv.reader(handle))
        header = grid[0][1:]
        matrix = {}
        for line in grid[1:]:
            actual = line[0]
            matrix[actual] = {
                predicted: int(value)
                for predicted, value in zip(header, line[1:])
            }
        confusions[name] = matrix

    # ---- hard class report -------------------------------------------------
    hard_report = []
    for hard in HARD_CLASSES:
        if hard not in classes:
            continue
        support = next(r for r in support_rows if r["class"] == hard)

        # What does each model get confused with on this class?
        per_model_detail = {}
        for name in experiments:
            row = per_model[name].get(hard, {})
            others = [
                (predicted, count)
                for predicted, count in confusions[name][hard].items()
                if predicted != hard and count > 0
            ]
            others.sort(key=lambda item: item[1], reverse=True)
            per_model_detail[name] = {
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f1": row.get("f1"),
                "support": row.get("support"),
                "top_confusions": [
                    {"predicted": predicted, "count": count}
                    for predicted, count in others[:4]
                ],
            }

        hard_report.append({
            "class": hard,
            "support": support,
            "per_model": per_model_detail,
        })

    # ---- confidence of the errors -----------------------------------------
    confidence_report = {}
    for name in experiments:
        mistakes = read_csv(
            comparison / name / "misclassified_images.csv"
        )
        hard_confidences = []
        all_confidences = []
        for row in mistakes:
            confidence = float(row["confidence"])
            all_confidences.append(confidence)
            if row["actual_label"] in HARD_CLASSES:
                hard_confidences.append({
                    "actual": row["actual_label"],
                    "predicted": row["predicted_label"],
                    "confidence": round(confidence, 4),
                    "path": row["source_path"],
                })
        hard_confidences.sort(key=lambda item: item["confidence"], reverse=True)
        confidence_report[name] = {
            "all_mistakes_mean_confidence": round(
                sum(all_confidences) / len(all_confidences), 4
            ) if all_confidences else 0.0,
            "all_mistakes_median_confidence": round(
                median(all_confidences), 4
            ),
            "hard_class_mistakes": hard_confidences,
        }

    # ---- cross-model agreement on the hard classes ------------------------
    # For each test image of a hard class, how many of the four models
    # classified it correctly?
    correct_sets = {}
    for name in experiments:
        rows = read_csv(comparison / name / "correct_images.csv")
        correct_sets[name] = {row["source_path"] for row in rows}

    agreement = []
    for hard in HARD_CLASSES:
        folder = test_folder / hard
        if not folder.is_dir():
            continue
        images = [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        buckets: Counter = Counter()
        for image in images:
            key = str(image.resolve())
            hits = sum(
                1 for name in experiments if key in correct_sets.get(name, set())
            )
            buckets[hits] += 1
        agreement.append({
            "class": hard,
            "test_images": len(images),
            "models_correct_histogram": {
                f"{k}_of_{len(experiments)}": v
                for k, v in sorted(buckets.items())
            },
            "never_correct_by_any_model": buckets.get(0, 0),
        })

    # ---- resolution shift --------------------------------------------------
    resolution = []
    for name in classes:
        train_px = [float(x) for x in pixels(train_folder / name)] if (train_folder / name).is_dir() else []
        test_px = [float(x) for x in pixels(test_folder / name)] if (test_folder / name).is_dir() else []
        train_median = median(train_px)
        test_median = median(test_px)
        resolution.append({
            "class": name,
            "train_median_px": int(train_median),
            "test_median_px": int(test_median),
            "ratio": round(test_median / train_median, 2) if train_median else None,
        })
    resolution.sort(
        key=lambda row: (row["ratio"] is None, -(row["ratio"] or 0))
    )

    # ---- confusion pair matrix --------------------------------------------
    pair_rows = []
    for source, target in CONFUSION_PAIRS:
        if source not in classes or target not in classes:
            continue
        for name in experiments:
            pair_rows.append({
                "experiment": name,
                "actual": source,
                "predicted": target,
                "count": confusions[name][source][target],
                "source_support": sum(confusions[name][source].values()),
            })
            pair_rows.append({
                "experiment": name,
                "actual": target,
                "predicted": source,
                "count": confusions[name][target][source],
                "source_support": sum(confusions[name][target].values()),
            })

    # ---- write the structured report --------------------------------------
    report = {
        "comparison": str(comparison),
        "experiments": experiments,
        "hard_classes": hard_report,
        "cross_model_agreement": agreement,
        "confidence": confidence_report,
        "resolution_shift": resolution,
        "note": (
            "The test set is reused across every experiment, so these "
            "figures are an exploratory comparison. The confidences are "
            "uncalibrated softmax outputs, not disease probabilities."
        ),
    }
    (out_dir / "hard_class_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    with (out_dir / "resolution_shift.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(resolution[0].keys())
        )
        writer.writeheader()
        writer.writerows(resolution)

    with (out_dir / "confusion_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(pair_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    # ---- contact sheets ----------------------------------------------------
    best_model = experiments[-1]
    mistakes = read_csv(
        comparison / best_model / "misclassified_images.csv"
    )
    by_actual = defaultdict(list)
    for row in mistakes:
        by_actual[row["actual_label"]].append(row)

    for hard in HARD_CLASSES:
        folder = test_folder / hard
        if not folder.is_dir():
            continue
        images = [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        # Every test image of the class, with what the best model said.
        rows = []
        for image in images:
            key = str(image.resolve())
            said = "not scored by the best model"
            for name in experiments:
                for row in read_csv(
                    comparison / name / "misclassified_images.csv"
                ):
                    if row["source_path"] == key:
                        said = f"{name}: {row['predicted_label']}"
                        break
                else:
                    continue
                break
            rows.append({
                "path": str(image),
                "caption": f"TEST {hard} | {said}",
            })
        contact_sheet(
            rows, out_dir / "contact_sheets" / f"test_{hard}.jpg"
        )

        # The training images the model finds hardest: the ones it puts
        # in the other classes.
        hard_mistakes = by_actual.get(hard, [])
        if hard_mistakes:
            contact_sheet(
                [
                    {
                        "path": row["source_path"],
                        "caption": (
                            f"TEST {row['actual_label']} -> "
                            f"{row['predicted_label']} "
                            f"({float(row['confidence']):.0%})"
                        ),
                    }
                    for row in hard_mistakes
                ],
                out_dir / "contact_sheets" / f"errors_{hard}.jpg",
            )

    print(f"Report written to {out_dir}")
    print(f"  class_support.csv          {len(support_rows)} classes")
    print(f"  hard_class_report.json     {len(hard_report)} hard classes")
    print(f"  resolution_shift.csv       {len(resolution)} classes")
    print(f"  confusion_pairs.csv        {len(pair_rows)} rows")
    print(f"  contact_sheets/            {len(list((out_dir / 'contact_sheets').glob('*.jpg')))} images")


if __name__ == "__main__":
    main()
