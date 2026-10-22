"""Phase 3 - resolution shift and significance of the model differences."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v2/data"
COMPARISON = ROOT / "ml/training/experiments/plantdoc_v4_v6_comparison"

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pixels(folder: Path) -> list[int]:
    out = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            try:
                with Image.open(path) as handle:
                    out.append(handle.size[0] * handle.size[1])
            except Exception:
                pass
    return out


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def main() -> None:
    print("=" * 72)
    print("TRAIN vs TEST RESOLUTION, ALL CLASSES")
    print("=" * 72)
    train_px = pixels(DATA / "train")
    test_px = pixels(DATA / "test")
    print(f"  train: {len(train_px)} images, median {median(train_px):,.0f} px")
    print(f"  test : {len(test_px)} images, median {median(test_px):,.0f} px")
    print(
        f"  ratio test/train median = {median(test_px) / median(train_px):.2f}x"
    )

    print()
    print("  Per class (test median / train median):")
    rows = []
    for train_folder in sorted((DATA / "train").iterdir()):
        if not train_folder.is_dir():
            continue
        name = train_folder.name
        test_folder = DATA / "test" / name
        train_values = pixels(train_folder)
        test_values = pixels(test_folder) if test_folder.is_dir() else []
        if not train_values or not test_values:
            continue
        train_median = median(train_values)
        test_median = median(test_values)
        rows.append((test_median / train_median, name, train_median, test_median))
    rows.sort(reverse=True)
    print(f"    {'class':34} {'train px':>12} {'test px':>12} {'ratio':>7}")
    for ratio, name, train_median, test_median in rows[:6]:
        print(
            f"    {name:34} {train_median:>12,.0f} "
            f"{test_median:>12,.0f} {ratio:>6.2f}x"
        )
    print("    ...")
    for ratio, name, train_median, test_median in rows[-3:]:
        print(
            f"    {name:34} {train_median:>12,.0f} "
            f"{test_median:>12,.0f} {ratio:>6.2f}x"
        )
    worse = sum(1 for r in rows if r[0] > 1.2)
    print(
        f"\n  {worse} of {len(rows)} classes have test images that are more "
        "than 20% larger than their training images."
    )

    print()
    print("=" * 72)
    print("ARE THE MODEL DIFFERENCES SIGNIFICANT?")
    print("=" * 72)
    summary = json.loads(
        (COMPARISON / "comparison_summary.json").read_text()
    )

    # Per-image correctness, joined on the image path.
    per_model: dict[str, dict[str, bool]] = {}
    for name in summary["experiments"]:
        correct: dict[str, bool] = {}
        for filename in ("correct_images.csv", "misclassified_images.csv"):
            path = COMPARISON / name / filename
            for row in csv.DictReader(open(path)):
                correct[row["source_path"]] = (
                    filename == "correct_images.csv"
                )
        per_model[name] = correct

    names = list(summary["experiments"])
    total = len(per_model[names[0]])

    for name in names:
        correct = sum(per_model[name].values())
        n = total
        p = correct / n
        # Wilson 95% interval.
        z = 1.96
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        margin = (
            z
            * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            / denominator
        )
        print(
            f"  {name:22} {correct:>3}/{n} = {p:.4f} "
            f"[{centre - margin:.4f}, {centre + margin:.4f}]"
        )

    print()
    print("  Paired McNemar test (exact binomial) between the models:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            only_a = sum(
                1
                for key in per_model[a]
                if per_model[a][key] and not per_model[b][key]
            )
            only_b = sum(
                1
                for key in per_model[a]
                if per_model[b][key] and not per_model[a][key]
            )
            if only_a + only_b == 0:
                print(f"    {a} vs {b}: identical predictions")
                continue
            n = only_a + only_b
            # Two-sided exact binomial p-value at p=0.5.
            k = min(only_a, only_b)
            tail = sum(
                math.comb(n, j) for j in range(0, k + 1)
            ) / (2 ** n)
            p_value = min(1.0, 2 * tail)
            verdict = (
                "significant" if p_value < 0.05 else "NOT significant"
            )
            print(
                f"    {a[:14]:14} vs {b[:14]:14} "
                f"{only_a} vs {only_b} disagreements (n={n}) "
                f"p={p_value:.3f} -> {verdict}"
            )


if __name__ == "__main__":
    main()
