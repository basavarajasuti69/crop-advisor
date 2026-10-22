from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


ROOT = Path("ml/training/experiments")
DATA = ROOT / "plantdoc_deduplicated_v1" / "data"
V2_DIR = ROOT / "plantdoc_27class_v2"
EXPERIMENT_DIR = ROOT / "plantdoc_hierarchical_v1"
MODEL_DIR = EXPERIMENT_DIR / "model_output"

DISEASE_MODEL_PATH = V2_DIR / "best_model.pt"
DISEASE_CLASSES_PATH = V2_DIR / "classes.json"
CROP_MODEL_PATH = MODEL_DIR / "crop_best_model.pt"
CROP_CLASSES_PATH = MODEL_DIR / "crop_classes.json"
MAPPING_PATH = EXPERIMENT_DIR / "crop_mapping.json"

METRICS_PATH = MODEL_DIR / "top3_evaluation.json"
PREDICTIONS_PATH = MODEL_DIR / "top3_predictions.csv"


def load_model(model_path, number_of_classes, device):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        number_of_classes,
    )
    state = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    required = [
        DISEASE_MODEL_PATH,
        DISEASE_CLASSES_PATH,
        CROP_MODEL_PATH,
        CROP_CLASSES_PATH,
        MAPPING_PATH,
        DATA / "test",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Required files are missing:\n" + "\n".join(missing)
        )

    disease_classes = json.loads(DISEASE_CLASSES_PATH.read_text())
    crop_classes = json.loads(CROP_CLASSES_PATH.read_text())
    mapping = json.loads(MAPPING_PATH.read_text())

    if set(disease_classes) != set(mapping):
        raise SystemExit("Disease-to-crop mapping mismatch.")

    if set(mapping.values()) != set(crop_classes):
        raise SystemExit("Crop classes and mapping mismatch.")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    dataset = ImageFolder(DATA / "test", transform=transform)

    if set(dataset.classes) != set(disease_classes):
        raise SystemExit("Test classes do not match disease model classes.")

    if len(dataset) != 225:
        raise SystemExit(
            f"Expected 225 test images; found {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    disease_model = load_model(
        DISEASE_MODEL_PATH, len(disease_classes), device
    )
    crop_model = load_model(
        CROP_MODEL_PATH, len(crop_classes), device
    )

    disease_to_id = {
        name: index for index, name in enumerate(disease_classes)
    }

    crop_disease_indices = {
        crop: [
            disease_to_id[disease]
            for disease in disease_classes
            if mapping[disease] == crop
        ]
        for crop in crop_classes
    }

    if any(not ids for ids in crop_disease_indices.values()):
        raise SystemExit("At least one crop has no mapped disease classes.")

    rows = []
    baseline_correct = 0
    top1_correct = 0
    top3_correct = 0
    top3_crop_contains_true = 0
    total = 0

    print("Device:", device)
    print("Test images:", len(dataset))
    print("Disease classes:", len(disease_classes))
    print("Crop classes:", len(crop_classes))
    print("Running Top-1 versus Top-3 evaluation...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            disease_logits = disease_model(images)
            crop_logits = crop_model(images)

            baseline_ids = disease_logits.argmax(dim=1).cpu().tolist()
            crop_top_ids = crop_logits.topk(
                k=min(3, len(crop_classes)), dim=1
            ).indices.cpu().tolist()

            disease_logits = disease_logits.cpu()
            labels = labels.tolist()

            for true_id, baseline_id, top_crop_ids, logits in zip(
                labels, baseline_ids, crop_top_ids, disease_logits
            ):
                true_disease = dataset.classes[true_id]
                true_crop = mapping[true_disease]
                baseline_disease = disease_classes[baseline_id]

                predicted_crops = [
                    crop_classes[crop_id] for crop_id in top_crop_ids
                ]

                top1_crop = predicted_crops[0]
                top3_contains_true_crop = true_crop in predicted_crops

                # Top-1: allow diseases from the most likely crop only.
                top1_ids = crop_disease_indices[top1_crop]
                top1_id = top1_ids[
                    int(logits[top1_ids].argmax().item())
                ]
                top1_disease = disease_classes[top1_id]

                # Top-3: allow diseases from any of the top three crops.
                allowed_ids = sorted({
                    disease_id
                    for crop in predicted_crops
                    for disease_id in crop_disease_indices[crop]
                })

                top3_id = allowed_ids[
                    int(logits[allowed_ids].argmax().item())
                ]
                top3_disease = disease_classes[top3_id]

                is_baseline_correct = baseline_disease == true_disease
                is_top1_correct = top1_disease == true_disease
                is_top3_correct = top3_disease == true_disease

                baseline_correct += int(is_baseline_correct)
                top1_correct += int(is_top1_correct)
                top3_correct += int(is_top3_correct)
                top3_crop_contains_true += int(top3_contains_true_crop)
                total += 1

                rows.append({
                    "image_path": str(dataset.samples[total - 1][0]),
                    "true_disease": true_disease,
                    "true_crop": true_crop,
                    "baseline_prediction": baseline_disease,
                    "top1_crop": top1_crop,
                    "top1_prediction": top1_disease,
                    "top3_crops": "; ".join(predicted_crops),
                    "top3_contains_true_crop": top3_contains_true_crop,
                    "top3_prediction": top3_disease,
                    "baseline_correct": is_baseline_correct,
                    "top1_correct": is_top1_correct,
                    "top3_correct": is_top3_correct,
                })

    if total != len(dataset):
        raise SystemExit(f"Evaluated {total} of {len(dataset)} images.")

    baseline_accuracy = baseline_correct / total
    top1_accuracy = top1_correct / total
    top3_accuracy = top3_correct / total
    top3_crop_recall = top3_crop_contains_true / total

    metrics = {
        "experiment": "PlantDoc top-3 crop-constrained evaluation",
        "device": str(device),
        "test_images": total,
        "baseline_disease_accuracy": round(baseline_accuracy, 4),
        "top1_disease_accuracy": round(top1_accuracy, 4),
        "top3_disease_accuracy": round(top3_accuracy, 4),
        "top3_true_crop_coverage": round(top3_crop_recall, 4),
        "top1_change_vs_baseline_percentage_points": round(
            (top1_accuracy - baseline_accuracy) * 100, 2
        ),
        "top3_change_vs_baseline_percentage_points": round(
            (top3_accuracy - baseline_accuracy) * 100, 2
        ),
        "warning": (
            "Exploratory PlantDoc evaluation. The test set was used in "
            "previous experiments; results do not establish field reliability."
        ),
    }

    with PREDICTIONS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\nRESULTS")
    print("-" * 45)
    print(f"Baseline disease accuracy: {baseline_accuracy:.2%}")
    print(f"Top-1 disease accuracy:    {top1_accuracy:.2%}")
    print(f"Top-3 disease accuracy:    {top3_accuracy:.2%}")
    print(f"Top-3 true-crop coverage:  {top3_crop_recall:.2%}")
    print(
        "Top-3 change vs baseline: "
        f"{metrics['top3_change_vs_baseline_percentage_points']:+.2f} "
        "percentage points"
    )
    print("\nMetrics saved to:", METRICS_PATH.resolve())
    print("Predictions saved to:", PREDICTIONS_PATH.resolve())
    print("Production files were not modified.")


if __name__ == "__main__":
    main()
