import csv
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = (
    ROOT / "ml/training/experiments/plantdoc_27class_v1"
)

CSV_PATH = EXPERIMENT / "misclassified_images.csv"
OUTPUT_PATH = EXPERIMENT / "misclassified_contact_sheet.jpg"

THUMB_W = 220
THUMB_H = 180
TEXT_H = 72
COLS = 4
BATCH_SIZE = 24

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("No misclassified images found in the CSV.")

font = ImageFont.load_default()

def find_image(row):
    # The CSV contains the copied filename. Resolve it safely.
    filename = unquote(row["image"])
    candidate = EXPERIMENT / "error_images" / filename

    if candidate.is_file():
        return candidate

    # Fallback: use the original source path recorded in the CSV.
    source = Path(row.get("source_path", ""))
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
            thumb.thumbnail((THUMB_W, THUMB_H))

        images.append((row, thumb))
    except Exception as exc:
        print(f"Could not open {path.name}: {exc}")
        missing += 1

if not images:
    raise SystemExit(
        "No images could be opened. Check the error_images directory."
    )

pages = math.ceil(len(images) / BATCH_SIZE)

for page in range(pages):
    batch = images[page * BATCH_SIZE:(page + 1) * BATCH_SIZE]
    rows_needed = math.ceil(len(batch) / COLS)

    cell_h = THUMB_H + TEXT_H
    sheet = Image.new(
        "RGB",
        (COLS * THUMB_W, rows_needed * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for i, (row, thumb) in enumerate(batch):
        x = (i % COLS) * THUMB_W
        y = (i // COLS) * cell_h

        image_x = x + (THUMB_W - thumb.width) // 2
        image_y = y + (THUMB_H - thumb.height) // 2

        sheet.paste(thumb, (image_x, image_y))

        confidence = float(row["confidence"])
        label = (
            f'Actual: {row["actual_label"]}\n'
            f'Pred: {row["predicted_label"]}\n'
            f'Confidence: {confidence:.1%}'
        )

        draw.multiline_text(
            (x + 5, y + THUMB_H + 3),
            label,
            fill="black",
            font=font,
            spacing=3,
        )

    page_path = EXPERIMENT / f"misclassified_contact_sheet_{page + 1}.jpg"
    sheet.save(page_path, quality=90)

    print(f"Saved: {page_path}")

print(f"\nImages included: {len(images)}")
print(f"Images unavailable or unreadable: {missing}")
print(f"Contact sheets created: {pages}")
