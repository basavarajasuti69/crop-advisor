"""Evaluate PlantDoc V4 and save experiment-only reports."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = ROOT / "ml/training/experiments/plantdoc_27class_v4"
DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v2/data"
CHECKPOINT = EXPERIMENT / "best_model.pt"
CLASSES_FILE = EXPERIMENT / "classes.json"

PER_CLASS_REPORT = EXPERIMENT / "per_class_report.csv"
ERROR_REPORT = EXPERIMENT / "misclassified_images.csv"
CONFUSION_REPORT = EXPERIMENT / "confusion_matrix.csv"
SUMMARY_REPORT = EXPERIMENT / "evaluation_summary.json"


def main():
    # Safety checks: read only the experiment model and dataset.
    for path in [CHECKPOINT, CLASSES_FILE, DATA / "test"]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    classes = json.loads(CLASSES_FILE.read_text())

    if isinstance(classes, dict):
        classes = classes.get("classes", classes)

    if not isinstance(classes, list) or len(classes) != 27:
        raise ValueError(
            f"Expected 27 model classes, found {len(classes)}."
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(DATA / "test", transform=transform)

    if set(dataset.classes) != set(classes):
        raise ValueError(
            "Dataset classes do not match the V4 model classes.\n"
            f"Model classes: {classes}\n"
            f"Dataset classes: {dataset.classes}"
        )

    # Explicitly map ImageFolder labels to the saved model class order.
    model_index = {name: i for i, name in enumerate(classes)}

    dataset.samples = [
        (path, model_index[dataset.classes[label]])
        for path, label in dataset.samples
    ]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    dataset.classes = classes
    dataset.class_to_idx = model_index

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    class_total = Counter()
    class_correct = Counter()
    confusion = Counter()
    mistakes = []
    total = 0
    correct = 0
    confidence_sum = 0.0

    print(f"Device: {device}")
    print(f"Model: {CHECKPOINT}")
    print(f"Test dataset: {DATA / 'test'}")
    print(f"Test images: {len(dataset)}")
    print("\nEvaluating V4...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            probabilities = torch.softmax(model(images), dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for actual_id, predicted_id, confidence in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
                confidences.cpu().tolist(),
            ):
                actual = classes[actual_id]
                predicted = classes[predicted_id]
                confidence = float(confidence)

                class_total[actual] += 1
                confusion[(actual, predicted)] += 1
                total += 1
                confidence_sum += confidence

                if actual_id == predicted_id:
                    correct += 1
                    class_correct[actual] += 1
                else:
                    image_path, _ = dataset.samples[
                        sum(class_total.values()) - 1
                    ]
                    mistakes.append({
                        "source_path": str(Path(image_path).resolve()),
                        "actual_label": actual,
                        "predicted_label": predicted,
                        "confidence": round(confidence, 6),
                        "confidence_percent": round(confidence * 100, 2),
                    })

    # Per-class performance report.
    with PER_CLASS_REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "correct",
                "total",
                "accuracy",
            ],
        )
        writer.writeheader()

        for name in classes:
            count = class_total[name]
            hits = class_correct[name]
            writer.writerow({
                "class_name": name,
                "correct": hits,
                "total": count,
                "accuracy": round(hits / count, 6) if count else "",
            })

    # Misclassified images, highest-confidence errors first.
    mistakes.sort(key=lambda row: row["confidence"], reverse=True)

    with ERROR_REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "actual_label",
                "predicted_label",
                "confidence",
                "confidence_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(mistakes)

    # Confusion matrix: rows are actual labels, columns are predictions.
    with CONFUSION_REPORT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual_label"] + classes)

        for actual in classes:
            writer.writerow([
                actual,
                *[confusion[(actual, predicted)] for predicted in classes],
            ])

    accuracy = correct / total if total else 0.0

    summary = {
        "experiment": "PlantDoc V4",
        "model": "efficientnet_b0",
        "dataset": str(DATA / "test"),
        "device": str(device),
        "classes": len(classes),
        "test_images": total,
        "correct_predictions": correct,
        "incorrect_predictions": len(mistakes),
        "accuracy": round(accuracy, 6),
        "accuracy_percent": round(accuracy * 100, 2),
        "mean_max_probability_percent": round(
            confidence_sum / total * 100, 2
        ) if total else 0,
        "warning": (
            "Exploratory benchmark only. PlantDoc folder labels may be "
            "noisy, and the test set has been reused across experiments. "
            "This is not proof of real-world diagnostic accuracy."
        ),
    }

    SUMMARY_REPORT.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 60)
    print("PLANTDOC V4 EVALUATION")
    print("=" * 60)
    print(f"Correct: {correct}/{total}")
    print(f"Incorrect: {len(mistakes)}/{total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        "Mean maximum predicted probability: "
        f"{summary['mean_max_probability_percent']:.2f}%"
    )

    print("\nLowest-accuracy classes:")
    class_results = []

    for name in classes:
        count = class_total[name]
        hits = class_correct[name]
        if count:
            class_results.append((hits / count, name, hits, count))

    for score, name, hits, count in sorted(class_results)[:10]:
        print(f"  {name}: {hits}/{count} ({score * 100:.1f}%)")

    print("\nHighest-confidence errors:")
    for row in mistakes[:10]:
        print(
            f"  Actual: {row['actual_label']} | "
            f"Predicted: {row['predicted_label']} | "
            f"Confidence: {row['confidence_percent']:.2f}%"
        )

    print("\nReports saved:")
    for path in [
        PER_CLASS_REPORT,
        ERROR_REPORT,
        CONFUSION_REPORT,
        SUMMARY_REPORT,
    ]:
        print(f"  {path}")

    print("\nProduction artifacts were not modified.")


if __name__ == "__main__":
    main()
