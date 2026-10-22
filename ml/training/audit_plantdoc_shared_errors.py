import pandas as pd
from pathlib import Path

BASE = Path("ml/training/experiments")
COMPARISON = BASE / "plantdoc_v2_v3_comparison"
OUTPUT = COMPARISON / "shared_error_audit.csv"

source = COMPARISON / "wrong_in_both.csv"

if not source.exists():
    raise FileNotFoundError(
        f"Comparison file not found: {source}"
    )

df = pd.read_csv(source)

required = {
    "source_path",
    "actual_label",
    "predicted_label_v2",
    "confidence_v2",
    "predicted_label_v3",
    "confidence_v3",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

def crop_group(label):
    label = str(label).lower()

    crop_prefixes = [
        ("tomato", "Tomato"),
        ("corn", "Corn"),
        ("apple", "Apple"),
        ("bell_pepper", "Bell pepper"),
        ("blueberry", "Blueberry"),
        ("cherry", "Cherry"),
        ("peach", "Peach"),
        ("potato", "Potato"),
        ("grape", "Grape"),
        ("raspberry", "Raspberry"),
        ("soyabean", "Soybean"),
        ("strawberry", "Strawberry"),
        ("squash", "Squash"),
    ]

    for prefix, crop in crop_prefixes:
        if label.startswith(prefix):
            return crop

    return "Other / review"

df["actual_crop"] = df["actual_label"].map(crop_group)
df["v2_predicted_crop"] = df["predicted_label_v2"].map(crop_group)
df["v3_predicted_crop"] = df["predicted_label_v3"].map(crop_group)

df["v2_cross_crop"] = (
    df["actual_crop"] != df["v2_predicted_crop"]
)

df["v3_cross_crop"] = (
    df["actual_crop"] != df["v3_predicted_crop"]
)

def classify_error(row):
    if row["actual_crop"] == "Other / review":
        return "Review crop mapping"

    if (
        row["v2_predicted_crop"] != row["v3_predicted_crop"]
    ):
        return "Models disagree on predicted crop"

    if row["v2_cross_crop"] and row["v3_cross_crop"]:
        return "Cross-crop confusion in both models"

    if not row["v2_cross_crop"] and not row["v3_cross_crop"]:
        return "Same-crop disease confusion"

    return "Cross-crop confusion in one model"

df["error_pattern"] = df.apply(classify_error, axis=1)

df["manual_review_priority"] = df.apply(
    lambda row: (
        "High"
        if row["v2_cross_crop"] and row["v3_cross_crop"]
        else "Medium"
        if row["v2_cross_crop"] or row["v3_cross_crop"]
        else "Review disease label"
    ),
    axis=1,
)

df["image_exists"] = df["source_path"].map(
    lambda path: Path(path).is_file()
)

df["folder_label"] = df["source_path"].map(
    lambda path: Path(path).parent.name
)

df["folder_label_matches"] = (
    df["folder_label"] == df["actual_label"]
)

columns = [
    "source_path",
    "image_exists",
    "actual_label",
    "folder_label",
    "folder_label_matches",
    "actual_crop",
    "predicted_label_v2",
    "confidence_v2",
    "v2_predicted_crop",
    "v2_cross_crop",
    "predicted_label_v3",
    "confidence_v3",
    "v3_predicted_crop",
    "v3_cross_crop",
    "error_pattern",
    "manual_review_priority",
]

df[columns].to_csv(OUTPUT, index=False)

print("=== SHARED ERROR AUDIT ===")
print("Images audited:", len(df))
print("Images found:", int(df["image_exists"].sum()))
print(
    "Folder/actual-label mismatches:",
    int((~df["folder_label_matches"]).sum()),
)

print("\nError patterns:")
print(df["error_pattern"].value_counts().to_string())

print("\nManual review priority:")
print(df["manual_review_priority"].value_counts().to_string())

print("\nCross-crop predictions:")
print("V2:", int(df["v2_cross_crop"].sum()))
print("V3:", int(df["v3_cross_crop"].sum()))

print("\nMost frequent actual classes:")
print(df["actual_label"].value_counts().head(10).to_string())

print("\nSaved audit:")
print(OUTPUT.resolve())
