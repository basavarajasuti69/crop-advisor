from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "ml/training/experiments/plantdoc_deduplicated_v1/data/test"
)

V2 = ROOT / "ml/training/experiments/plantdoc_27class_v2"
V3 = ROOT / "ml/training/experiments/plantdoc_27class_v3"

OUT = ROOT / "ml/training/experiments/plantdoc_v2_v3_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def load_errors(experiment, model_name):
    path = experiment / "misclassified_images.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "source_path",
        "actual_label",
        "predicted_label",
        "confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} report is missing columns: {missing}"
        )

    def relative_path(value):
        full_path = Path(str(value)).resolve()

        try:
            return full_path.relative_to(DATA.resolve()).as_posix()
        except ValueError:
            raise ValueError(
                f"Image path is outside the shared test dataset: "
                f"{value}"
            )

    df["image_key"] = df["source_path"].map(relative_path)

    if df["image_key"].duplicated().any():
        duplicates = df.loc[
            df["image_key"].duplicated(keep=False),
            "image_key",
        ].tolist()

        raise ValueError(
            f"Duplicate image paths in {model_name}: {duplicates}"
        )

    # Verify that each reported image really belongs to its actual class.
    for _, row in df.iterrows():
        expected_class = row["image_key"].split("/", 1)[0]

        if expected_class != row["actual_label"]:
            raise ValueError(
                f"{model_name}: class/path mismatch for "
                f"{row['image_key']}: "
                f"CSV says {row['actual_label']!r}"
            )

    return df


v2 = load_errors(V2, "V2")
v3 = load_errors(V3, "V3")

merged = v2.merge(
    v3,
    on="image_key",
    how="outer",
    suffixes=("_v2", "_v3"),
    indicator=True,
    validate="one_to_one",
)

both_wrong = merged[merged["_merge"] == "both"].copy()
v2_only_wrong = merged[merged["_merge"] == "left_only"].copy()
v3_only_wrong = merged[merged["_merge"] == "right_only"].copy()

# Check that matched errors have consistent ground-truth labels.
matched = merged[merged["_merge"] == "both"]

if not (
    matched["actual_label_v2"] == matched["actual_label_v3"]
).all():
    raise ValueError(
        "Ground-truth labels disagree for some matched images."
    )

both_wrong.to_csv(OUT / "wrong_in_both.csv", index=False)
v2_only_wrong.to_csv(OUT / "v2_wrong_v3_correct.csv", index=False)
v3_only_wrong.to_csv(OUT / "v2_correct_v3_wrong.csv", index=False)

# Compare against every image in the test directory.
test_images = [
    path.relative_to(DATA).as_posix()
    for path in DATA.rglob("*")
    if path.is_file()
    and path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
]

print("=== DATASET VALIDATION ===")
print(f"Test images found: {len(test_images)}")
print(f"V2 errors: {len(v2)}")
print(f"V3 errors: {len(v3)}")
print(f"Images wrong in both: {len(both_wrong)}")
print(f"V2 wrong, V3 correct: {len(v2_only_wrong)}")
print(f"V2 correct, V3 wrong: {len(v3_only_wrong)}")

print("\n=== COMPARISON OUTPUT ===")
print(OUT)

print(
    "\nNote: These reports contain only misclassified images. "
    "An image absent from a model's error report is treated as "
    "correct, assuming both evaluators used the same test set."
)
