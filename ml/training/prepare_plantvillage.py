"""Prepare a small PlantVillage dataset for Crop Disease Advisor experiments.

Creates a separate dataset under ml/data/plantvillage_subset.
Original images are not copied or modified.
"""

from __future__ import annotations

import random
from pathlib import Path

SOURCE = Path(
    "ml/data/raw/plantvillage/plantvillage dataset/color"
)
OUTPUT = Path("ml/data/plantvillage_subset")

SEED = 42
TRAIN_RATIO = 0.8
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Map original PlantVillage folders to Crop Disease Advisor labels.
CLASS_MAP = {
    "maize_healthy": [
        "Corn_(maize)___healthy",
    ],
    "maize_leaf_blight": [
        "Corn_(maize)___Northern_Leaf_Blight",
    ],
    "tomato_healthy": [
        "Tomato___healthy",
    ],
    "tomato_blight": [
        "Tomato___Early_blight",
        "Tomato___Late_blight",
    ],
}


def get_images(folder: Path) -> list[Path]:
    """Return supported image files in a source folder."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(
            f"PlantVillage color dataset not found:\n{SOURCE}"
        )

    random.seed(SEED)
    print(f"Source: {SOURCE.resolve()}")
    print(f"Output: {OUTPUT.resolve()}")
    print("Preparing dataset...\n")

    # Validate every source folder before creating output.
    images_by_class: dict[str, list[Path]] = {}

    for target_class, source_folders in CLASS_MAP.items():
        images: list[Path] = []

        for source_folder in source_folders:
            images.extend(get_images(SOURCE / source_folder))

        if len(images) < 2:
            raise SystemExit(
                f"Not enough images for {target_class}: {len(images)}"
            )

        images_by_class[target_class] = sorted(images)

    # Refuse to overwrite an existing preparation.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(
            f"\nOutput already exists and is not empty:\n{OUTPUT}\n"
            "Nothing was changed. Rename or remove this experimental "
            "output folder before running the script again."
        )

    # Create the separate dataset using links to original images.
    for target_class, images in images_by_class.items():
        shuffled = images.copy()
        random.shuffle(shuffled)

        split_index = int(len(shuffled) * TRAIN_RATIO)
        train_images = shuffled[:split_index]
        test_images = shuffled[split_index:]

        for split_name, split_images in (
            ("train", train_images),
            ("test", test_images),
        ):
            destination = OUTPUT / split_name / target_class
            destination.mkdir(parents=True, exist_ok=True)

            for index, source_image in enumerate(split_images):
                # Prefix with the source folder to avoid filename collisions.
                safe_source = source_image.parent.name.replace("/", "_")
                filename = (
                    f"{safe_source}__{index:05d}"
                    f"{source_image.suffix.lower()}"
                )
                destination_image = destination / filename
                destination_image.symlink_to(source_image.resolve())

        print(
            f"{target_class}: "
            f"{len(train_images)} train, {len(test_images)} test"
        )

    print("\nDataset preparation complete.")
    print(f"Dataset location: {OUTPUT.resolve()}")
    print("\nThis is a four-class experiment, not the final model dataset.")


if __name__ == "__main__":
    main()