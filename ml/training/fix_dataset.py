from pathlib import Path
import csv
import shutil
import hashlib

# Automatically locate the dataset.
SEARCH_ROOTS = [
    Path.cwd(),
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
]

DATASET_NAMES = {"dataset", "plantvillage", "plant_village"}

CORRECTIONS = [
    {
        "old": "Corn leaf blight",
        "file": "Corn-gray-leaf-spot-8-14-15-F1.jpg",
        "new": "Corn Gray leaf spot",
        "action": "RELABEL",
    },
    {
        "old": "Corn leaf blight",
        "file": "corn_gray_leaf_spot.jpg",
        "new": "Corn Gray leaf spot",
        "action": "RELABEL",
    },
    {
        "old": "Corn leaf blight",
        "file": "gray-leaf-spot-lesions-F4.jpg",
        "new": "Corn Gray leaf spot",
        "action": "RELABEL",
    },
    {
        "old": "Corn leaf blight",
        "file": "gray-leaf-spot-severe--.jpg",
        "new": "Corn Gray leaf spot",
        "action": "RELABEL",
    },
    {
        "old": "Potato leaf early blight",
        "file": "1573852-Early-Blight-e1515624612559.jpg",
        "new": "Tomato Early blight leaf",
        "action": "RELABEL",
    },
    {
        "old": "Potato leaf late blight",
        "file": "early_blight.jpg",
        "new": "Potato leaf early blight",
        "action": "RELABEL",
    },
    {
        "old": "Potato leaf late blight",
        "file": "potato_ocw_botrytis-lateblight.jpg",
        "new": None,
        "action": "REMOVE",
    },
    {
        "old": "Potato leaf late blight",
        "file": "potato_ocw_lateblight-earlyblight.jpg",
        "new": None,
        "action": "REMOVE",
    },
    {
        "old": "Tomato mold leaf",
        "file": "tylc.jpg",
        "new": "Tomato leaf yellow virus",
        "action": "RELABEL",
    },
]


def find_dataset():
    candidates = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        try:
            for folder in root.rglob("*"):
                if not folder.is_dir():
                    continue

                if folder.name.lower() in DATASET_NAMES:
                    if any(
                        (folder / split).is_dir()
                        for split in ("train", "valid", "val", "test")
                    ):
                        candidates.append(folder)

        except (PermissionError, OSError):
            continue

    candidates = list(dict.fromkeys(candidates))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise RuntimeError(
            "Dataset not found. Place this script in your project "
            "directory and set DATASET_ROOT manually."
        )

    print("Multiple possible datasets found:")
    for index, path in enumerate(candidates, 1):
        print(f"{index}. {path}")

    raise RuntimeError(
        "Multiple datasets found. Set DATASET_ROOT to the correct path."
    )


# Set this to an explicit path if automatic discovery is ambiguous.
DATASET_ROOT = None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    dataset = Path(DATASET_ROOT).expanduser() if DATASET_ROOT else find_dataset()

    print(f"\nDataset: {dataset}")

    # Find the train split without assuming capitalization.
    train_candidates = [
        path for path in dataset.iterdir()
        if path.is_dir() and path.name.lower() == "train"
    ]

    if len(train_candidates) != 1:
        raise RuntimeError("Could not uniquely identify the training folder.")

    train_dir = train_candidates[0]

    # Require exact class-folder names from the approved review.
    class_dirs = {path.name: path for path in train_dir.iterdir() if path.is_dir()}

    backup_dir = dataset / "_dataset_correction_backup"

    if backup_dir.exists():
        raise RuntimeError(
            f"Backup directory already exists: {backup_dir}\n"
            "Inspect it before running the script again."
        )

    backup_dir.mkdir()

    report = []
    errors = []

    # Preflight: verify every file and destination before making changes.
    planned = []

    for correction in CORRECTIONS:
        source_class = correction["old"]

        if source_class not in class_dirs:
            errors.append(f"Missing source class: {source_class}")
            continue

        matches = list(
            (class_dirs[source_class]).rglob(correction["file"])
        )

        if len(matches) != 1:
            errors.append(
                f"{correction['file']}: expected exactly one source file, "
                f"found {len(matches)}"
            )
            continue

        source = matches[0]

        if correction["action"] == "RELABEL":
            target_class = correction["new"]

            if target_class not in class_dirs:
                errors.append(f"Missing destination class: {target_class}")
                continue

            target = class_dirs[target_class] / source.name

            if target.exists():
                errors.append(f"Destination already exists: {target}")
                continue
        else:
            target = None

        planned.append((correction, source, target))

    if errors:
        backup_dir.rmdir()
        print("\nNo corrections applied. Resolve these issues first:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    # Back up affected files before moving or removing them.
    for correction, source, target in planned:
        backup_path = backup_dir / correction["file"]
        shutil.copy2(source, backup_path)

        report.append({
            "file": correction["file"],
            "old_class": correction["old"],
            "new_class": correction["new"] or "",
            "action": correction["action"],
            "sha256": sha256(source),
            "status": "PENDING",
        })

    # Apply changes.
    for item, (_, source, target) in zip(report, planned):
        try:
            if item["action"] == "RELABEL":
                shutil.move(str(source), str(target))
                item["status"] = "RELABELLED"
                print(f"RELABELLED: {source.name} -> {target.parent.name}")
            else:
                # Quarantine instead of permanently deleting.
                quarantine = backup_dir / f"REMOVED_{source.name}"
                shutil.move(str(source), str(quarantine))
                item["status"] = "QUARANTINED"
                print(f"QUARANTINED: {source.name}")

        except Exception as error:
            item["status"] = f"ERROR: {error}"
            print(f"ERROR: {source}: {error}")

    report_file = dataset / "dataset_correction_report.csv"

    with report_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    # Report current class counts.
    print("\nTRAINING CLASS COUNTS")
    for class_dir in sorted(
        path for path in train_dir.iterdir() if path.is_dir()
    ):
        count = sum(1 for path in class_dir.rglob("*") if path.is_file())
        print(f"{class_dir.name}: {count}")

    print(f"\nReport: {report_file}")
    print(f"Backup/quarantine: {backup_dir}")
    print("\nNext: inspect the report and class counts before retraining.")


if __name__ == "__main__":
    main()