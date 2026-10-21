import argparse
import csv
import math
from pathlib import Path
from urllib.parse import unquote

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description="Create contact sheets from a misclassified-images CSV."
    )
    parser.add_argument(
        "--experiment",
        default="plantdoc_27class_v1",
        help="Experiment directory name under ml/training/experiments/",
    )
    parser.add_argument(
        "--csv",
        default="misclassified_images.csv",
        help="CSV filename inside the experiment directory.",
    )
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    if args.cols < 1 or args.batch_size < 1:
        parser.error("--cols and --batch-size must be positive integers")

    experiment = ROOT / "ml/training/experiments" / args.experiment
    csv_path = experiment / args.csv

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No misclassified images found in {csv_path}")

    required = {"actual_label", "predicted_label", "confidence"}
    missing_columns = required - set(rows[0].keys())
    if missing_columns:
        raise SystemExit(
            f"CSV is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    font = ImageFont.load_default()
    thumb_w, thumb_h, text_h = 260, 190, 85

    def find_image(row):
        # Support both V1's copied-image CSV and V2's source_path CSV.
        filename = row.get("image", "").strip()
        if filename:
            candidate = experiment / "error_images" / unquote(filename)
            if candidate.is_file():
                return candidate

        source_path = row.get("source_path", "").strip()
        if source_path:
            # Try the path exactly as recorded, then URL-decode it.
            raw_source = Path(source_path)
            decoded_source = Path(unquote(source_path))

            candidates = [raw_source, decoded_source]
            for source in candidates:
                if not source.is_absolute():
                    source = ROOT / source
                if source.is_file():
                    return source

        return None

    images = []
    missing = 0

    for row in rows:
        path = find_image(row)
        if path is None:
            missing += 1
            continue

        try:
            with Image.open(path) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((thumb_w, thumb_h))
            images.append((row, thumb))
        except Exception as exc:
            print(f"Could not open {path.name}: {exc}")
            missing += 1

    if not images:
        raise SystemExit(
            "No images could be opened. Check the CSV paths and dataset files."
        )

    pages = math.ceil(len(images) / args.batch_size)

    for page in range(pages):
        batch = images[page * args.batch_size:(page + 1) * args.batch_size]
        rows_needed = math.ceil(len(batch) / args.cols)
        cell_h = thumb_h + text_h

        sheet = Image.new(
            "RGB",
            (args.cols * thumb_w, rows_needed * cell_h),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for i, (row, thumb) in enumerate(batch):
            x = (i % args.cols) * thumb_w
            y = (i // args.cols) * cell_h
            image_x = x + (thumb_w - thumb.width) // 2
            image_y = y + (thumb_h - thumb.height) // 2
            sheet.paste(thumb, (image_x, image_y))

            try:
                confidence = float(row["confidence"])
                confidence_text = f"{confidence:.1%}"
            except (TypeError, ValueError):
                confidence_text = str(row["confidence"])

            label = (
                f'Actual: {row["actual_label"]}\n'
                f'Pred: {row["predicted_label"]}\n'
                f'Confidence: {confidence_text}'
            )
            draw.multiline_text(
                (x + 5, y + thumb_h + 3),
                label,
                fill="black",
                font=font,
                spacing=3,
            )

        output_path = experiment / (
            f"misclassified_contact_sheet_{page + 1}.jpg"
        )
        sheet.save(output_path, quality=90)
        print(f"Saved: {output_path}")

    print(f"\nImages included: {len(images)}")
    print(f"Images unavailable or unreadable: {missing}")
    print(f"Contact sheets created: {pages}")


if __name__ == "__main__":
    main()
