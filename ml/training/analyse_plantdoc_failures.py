"""Phase 3 - inspect the hardest classes.

Read-only. Looks at what the model gets wrong and what the images look
like, so the failure is described rather than guessed at.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v2/data"
COMPARISON = ROOT / "ml/training/experiments/plantdoc_v4_v6_comparison"

HARD = [
    "Tomato mold leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf mosaic virus",
    "Cherry leaf",
    "Corn leaf blight",
    "Potato leaf early blight",
]


def image_stats(folder: Path) -> list[tuple[str, tuple[int, int]]]:
    from PIL import Image

    out = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as handle:
                out.append((path.name, handle.size))
        except Exception:
            out.append((path.name, (0, 0)))
    return out


def main() -> None:
    print("=" * 72)
    print("TRAIN vs TEST IMAGE SIZES FOR THE HARD CLASSES")
    print("=" * 72)
    for name in HARD:
        train_folder = DATA / "train" / name
        test_folder = DATA / "test" / name
        train_stats = image_stats(train_folder) if train_folder.is_dir() else []
        test_stats = image_stats(test_folder) if test_folder.is_dir() else []
        if not train_stats and not test_stats:
            print(f"{name}: not present")
            continue
        train_px = [w * h for _, (w, h) in train_stats]
        test_px = [w * h for _, (w, h) in test_stats]
        small_train = sum(1 for px in train_px if px < 200 * 200)
        small_test = sum(1 for px in test_px if px < 200 * 200)
        print(f"\n{name}")
        print(
            f"  train: {len(train_stats):>4} images, "
            f"median {int(sorted(train_px)[len(train_px)//2]) if train_px else 0:>8} px, "
            f"{small_train} under 200x200"
        )
        print(
            f"  test : {len(test_stats):>4} images, "
            f"median {int(sorted(test_px)[len(test_px)//2]) if test_px else 0:>8} px, "
            f"{small_test} under 200x200"
        )

    print()
    print("=" * 72)
    print("CLASSES WHERE EVERY MODEL FAILS (V4, V5-A, V5-B, V6)")
    print("=" * 72)
    rows = list(
        csv.DictReader(open(COMPARISON / "per_class_all_models.csv"))
    )
    models = sorted({r["experiment"] for r in rows})
    by_class: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_class.setdefault(row["class"], {})[row["experiment"]] = row

    never_right = []
    for class_name, per_model in by_class.items():
        support = int(next(iter(per_model.values()))["support"])
        if support == 0:
            continue
        hits = {
            model: int(per_model[model]["tp"]) for model in models
        }
        if sum(hits.values()) == 0:
            never_right.append((class_name, support, hits))

    print(f"{'class':34} {'sup':>4}  " + "  ".join(m[-3:] for m in models))
    for class_name, support, hits in sorted(
        never_right, key=lambda row: row[1]
    ):
        print(
            f"{class_name:34} {support:>4}  "
            + "  ".join(f"{hits[m]:>5}" for m in models)
        )
    print(
        f"\n{len(never_right)} of {len(by_class)} classes are never predicted "
        "correctly by any of the four models."
    )

    print()
    print("=" * 72)
    print("TEST CLASSES WITH VERY FEW IMAGES (noisy per-class accuracy)")
    print("=" * 72)
    small = sorted(
        (
            (int(next(iter(per_model.values()))["support"]), class_name)
            for class_name, per_model in by_class.items()
        )
    )
    for support, class_name in small[:8]:
        print(f"  {support:>3} images  {class_name}")
    print(
        "\nWith this many images, one image is "
        f"{100 // max(small[0][0], 1)}% of the class's accuracy, so the "
        "per-class numbers for the small classes are not stable."
    )


if __name__ == "__main__":
    main()
