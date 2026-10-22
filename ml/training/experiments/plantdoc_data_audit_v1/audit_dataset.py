from pathlib import Path
from collections import Counter
import json

ROOT = Path.cwd()

DATA = (
    ROOT
    / "ml"
    / "training"
    / "experiments"
    / "plantdoc_deduplicated_v1"
    / "data"
)

OUTPUT = (
    ROOT
    / "ml"
    / "training"
    / "experiments"
    / "plantdoc_data_audit_v1"
    / "dataset_audit_report.json"
)


def inspect_split(split_path):
    results = {}
    total_images = 0
    extensions = Counter()
    unreadable_images = []

    try:
        from PIL import Image
    except ImportError:
        Image = None

    for class_dir in sorted(split_path.iterdir()):
        if not class_dir.is_dir():
            continue

        files = [
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        ]

        results[class_dir.name] = len(files)
        total_images += len(files)

        for path in files:
            extensions[path.suffix.lower()] += 1

            if Image is not None:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception:
                    unreadable_images.append(str(path))

    return {
        "total_images": total_images,
        "class_count": len(results),
        "images_per_class": results,
        "file_extensions": dict(extensions),
        "unreadable_images": unreadable_images,
    }


def main():
    if not DATA.is_dir():
        raise SystemExit(f"Dataset directory not found: {DATA}")

    train_path = DATA / "train"
    test_path = DATA / "test"

    if not train_path.is_dir() or not test_path.is_dir():
        raise SystemExit("Expected train/ and test/ directories.")

    print("Inspecting experimental PlantDoc dataset...")
    print("Dataset:", DATA)

    train = inspect_split(train_path)
    test = inspect_split(test_path)

    train_classes = set(train["images_per_class"])
    test_classes = set(test["images_per_class"])

    report = {
        "dataset_path": str(DATA),
        "train": train,
        "test": test,
        "classes_only_in_train": sorted(train_classes - test_classes),
        "classes_only_in_test": sorted(test_classes - train_classes),
        "total_images": train["total_images"] + test["total_images"],
        "warning": (
            "This report checks file structure and readability, "
            "not whether disease labels are scientifically correct."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    print("\nDATASET AUDIT SUMMARY")
    print("-" * 35)
    print(f"Training images: {train['total_images']}")
    print(f"Training classes: {train['class_count']}")
    print(f"Test images: {test['total_images']}")
    print(f"Test classes: {test['class_count']}")
    print(f"Total images: {report['total_images']}")

    print("\nClasses only in training:")
    print(report["classes_only_in_train"] or "None")

    print("\nClasses only in test:")
    print(report["classes_only_in_test"] or "None")

    print("\nUnreadable training images:", len(train["unreadable_images"]))
    print("Unreadable test images:", len(test["unreadable_images"]))

    print("\nReport saved to:")
    print(OUTPUT)

    print("\nNo model or production files were modified.")


if __name__ == "__main__":
    main()