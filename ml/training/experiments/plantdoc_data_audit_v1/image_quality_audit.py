from pathlib import Path
from collections import Counter
import csv
import json

import numpy as np
from PIL import Image, ImageStat

ROOT = Path.cwd()
DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v1/data"
OUT = ROOT / "ml/training/experiments/plantdoc_data_audit_v1"

CSV_PATH = OUT / "image_quality_review.csv"
SUMMARY_PATH = OUT / "image_quality_summary.json"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_image(path):
    result = {
        "image_path": str(path.relative_to(ROOT)),
        "split": path.parts[-4],
        "class_label": path.parent.name,
        "width": "",
        "height": "",
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "blur_score": "",
        "contrast": "",
        "flags": "",
        "status": "NO_AUTOMATED_FLAGS",
        "error": "",
    }

    flags = []

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            gray = image.convert("L")
            contrast = ImageStat.Stat(gray).stddev[0]
            pixels = np.asarray(gray, dtype=np.float32)

            laplacian = (
                -4 * pixels
                + np.roll(pixels, 1, axis=0)
                + np.roll(pixels, -1, axis=0)
                + np.roll(pixels, 1, axis=1)
                + np.roll(pixels, -1, axis=1)
            )
            blur_score = float(laplacian.var())

        result.update({
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 2),
            "contrast": round(contrast, 2),
        })

        if width < 150 or height < 150:
            flags.append("small_dimensions")
        if result["file_size_kb"] < 10:
            flags.append("small_file_size")
        if blur_score < 25:
            flags.append("possible_blur_or_low_detail")
        if contrast < 15:
            flags.append("low_contrast")
        if width / height > 3 or width / height < 0.33:
            flags.append("unusual_aspect_ratio")

    except Exception as exc:
        flags.append("unreadable_or_processing_error")
        result["error"] = str(exc)

    result["flags"] = ";".join(flags)
    if flags:
        result["status"] = "REVIEW"

    return result


def main():
    if not DATA.is_dir():
        raise SystemExit(f"Dataset not found: {DATA}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    flag_counts = Counter()
    split_counts = Counter()

    for split in ("train", "test"):
        files = sorted(
            p for p in (DATA / split).rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        print(f"Inspecting {split}: {len(files)} images")

        for path in files:
            row = inspect_image(path)
            row["split"] = split
            rows.append(row)
            split_counts[split] += 1
            flag_counts.update(filter(None, row["flags"].split(";")))

    fields = [
        "image_path", "split", "class_label", "width", "height",
        "file_size_kb", "blur_score", "contrast", "flags", "status", "error"
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    flagged = sum(row["status"] == "REVIEW" for row in rows)

    summary = {
        "images_inspected": len(rows),
        "images_by_split": dict(split_counts),
        "images_flagged_for_review": flagged,
        "flag_counts": dict(flag_counts),
        "warning": (
            "Automated flags are heuristics, not proof of poor quality "
            "or incorrect labels. Review flagged images manually."
        ),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nIMAGE QUALITY AUDIT")
    print("=" * 40)
    print("Images inspected:", len(rows))
    print("Images flagged for review:", flagged)
    print("\nFlag counts:")
    for name, count in flag_counts.most_common():
        print(f"- {name}: {count}")

    print("\nCSV report:", CSV_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Dataset and production files were not modified.")


if __name__ == "__main__":
    main()
