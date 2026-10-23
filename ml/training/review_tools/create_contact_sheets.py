import csv
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path.cwd()

MANIFEST = ROOT / (
    "ml/training/experiments/plantdoc_v9_audit/"
    "v9_cleaning_manifest.csv"
)

DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v2/data"

OUT = ROOT / "ml/training/experiments/plantdoc_v9_audit/contact_sheets"
OUT.mkdir(parents=True, exist_ok=True)

THUMB_W = 190
THUMB_H = 155
LABEL_H = 58
COLS = 5
BATCH_SIZE = 50

with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

groups: dict = {}

for row in rows:
    split = row.get("split", "").strip()
    current = row.get("current_label", "").strip()
    filename = row.get("file", "").strip()

    if not split or not filename:
        continue

    path = DATA / split / filename

    if not path.is_file():
        # Some manifests store paths relative to the split directory.
        path = DATA / filename

    key = (split, current)
    groups.setdefault(key, []).append((row, path))

created = 0

for (split, label), items in groups.items():
    safe_label = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in label
    )

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        rows_count = math.ceil(len(batch) / COLS)

        sheet = Image.new(
            "RGB",
            (COLS * THUMB_W, rows_count * (THUMB_H + LABEL_H)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for i, (record, path) in enumerate(batch):
            x = (i % COLS) * THUMB_W
            y = (i // COLS) * (THUMB_H + LABEL_H)

            if path.is_file():
                try:
                    with Image.open(path) as img:
                        img = ImageOps.exif_transpose(img)
                        if img is None:
                            raise ValueError("could not transpose")
                        img = img.convert("RGB")
                        img.thumbnail((THUMB_W - 10, THUMB_H - 10))
                        sheet.paste(img, (x + 5, y + 5))
                except Exception:
                    draw.text((x + 5, y + 5), "IMAGE ERROR", fill="red")
            else:
                draw.text((x + 5, y + 5), "FILE NOT FOUND", fill="red")

            filename = record.get("file", "")
            suggested = record.get("suggested_label", "")

            draw.text(
                (x + 5, y + THUMB_H),
                filename[:27],
                fill="black",
            )
            draw.text(
                (x + 5, y + THUMB_H + 18),
                ("Suggested: " + suggested)[:27],
                fill="blue",
            )

        start_num = batch_start + 1
        end_num = batch_start + len(batch)

        output = OUT / (
            f"{split}_{safe_label}_{start_num}-{end_num}.jpg"
        )

        sheet.save(output, quality=90)
        created += 1
        print("Created:", output.relative_to(ROOT))

print(f"\nDone. Created {created} contact sheets.")
print("Folder:", OUT)
