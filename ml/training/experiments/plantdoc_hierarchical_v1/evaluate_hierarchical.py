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

METRICS_PATH = MODEL_DIR / "hierarchical_evaluation.json"
PREDICTIONS_PATH = MODEL_DIR / "hierarchical_predictions.csv"


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
            "Required experimental files are missing:\n"
            + "\n".join(missing)
        )

    disease_classes = json.loads(DISEASE_CLASSES_PATH.read_text())
    crop_classes = json.loads(CROP_CLASSES_PATH.read_text())
    mapping = json.loads(MAPPING_PATH.read_text())

    if set(disease_classes) != set(mapping):
        raise SystemExit(
            "Disease classes and crop mapping do not match. Stopping."
        )

    if set(mapping.values()) != set(crop_classes):
        raise SystemExit(
            "Crop classes and mapping do not match. Stopping."
        )

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

    test_dataset = ImageFolder(DATA / "test", transform=transform)

    if set(test_dataset.classes) != set(disease_classes):
        raise SystemExit(
            "Test dataset classes do not match the V2 disease classes."
        )

    if len(test_dataset) != 225:
        raise SystemExit(
            f"Expected 225 test images, found {len(test_dataset)}. "
            "Check the dataset before evaluating."
        )

    loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    disease_model = load_model(
        DISEASE_MODEL_PATH,
        len(disease_classes),
        device,
    )
    crop_model = load_model(
        CROP_MODEL_PATH,
        len(crop_classes),
        device,
    )

    disease_to_id = {
        name: index for index, name in enumerate(disease_classes)
    }

    # For each crop, retain the V2 disease-model output indices
    # corresponding to diseases mapped to that crop.
    crop_disease_indices = {}
    for crop in crop_classes:
        crop_disease_indices[crop] = [
            disease_to_id[disease]
            for disease in disease_classes
            if mapping[disease] == crop
        ]

        if not crop_disease_indices[crop]:
            raise SystemExit(f"No disease classes mapped to {crop}.")

    rows = []
    baseline_correct = 0
    crop_correct = 0
    hierarchical_correct = 0
    crop_correct_hierarchical_correct = 0
    total = 0

    print("Device:", device)
    print("Test images:", len(test_dataset))
    print("Disease classes:", len(disease_classes))
    print("Crop classes:", len(crop_classes))
    print("Running experimental evaluation...")

    with torch.no_grad():
        for images, labels in loader:
            disease_logits = disease_model(images.to(device))
            crop_logits = crop_model(images.to(device))

            # Ordinary 27-class disease prediction.
            baseline_ids = disease_logits.argmax(dim=1).cpu().tolist()

            # Crop prediction.
            crop_ids = crop_logits.argmax(dim=1).cpu().tolist()

            labels = labels.tolist()

            for true_id, baseline_id, crop_id, logits in zip(
                labels,
                baseline_ids,
                crop_ids,
                disease_logits.cpu(),
            ):
                true_disease = test_dataset.classes[true_id]
                baseline_disease = disease_classes[baseline_id]
                predicted_crop = crop_classes[crop_id]
                true_crop = mapping[true_disease]

                # Restrict disease candidates to the predicted crop.
                allowed_ids = crop_disease_indices[predicted_crop]
                allowed_logits = logits[allowed_ids]
                chosen_position = int(allowed_logits.argmax().item())
                hierarchical_id = allowed_ids[chosen_position]
                hierarchical_disease = disease_classes[hierarchical_id]

                is_baseline_correct = baseline_disease == true_disease
                is_crop_correct = predicted_crop == true_crop
                is_hierarchical_correct = (
                    hierarchical_disease == true_disease
                )

                baseline_correct += int(is_baseline_correct)
                crop_correct += int(is_crop_correct)
                hierarchical_correct += int(is_hierarchical_correct)
                crop_correct_hierarchical_correct += int(
                    is_crop_correct and is_hierarchical_correct
                )
                total += 1

                rows.append({
                    "image_path": str(
                        Path(test_dataset.samples[len(rows)][0])
                    ),
                    "true_disease": true_disease,
                    "true_crop": true_crop,
                    "baseline_predicted_disease": baseline_disease,
                    "predicted_crop": predicted_crop,
                    "hierarchical_predicted_disease": hierarchical_disease,
                    "baseline_correct": is_baseline_correct,
                    "crop_correct": is_crop_correct,
                    "hierarchical_correct": is_hierarchical_correct,
                })

    if total != len(test_dataset):
        raise SystemExit(
            f"Evaluated {total} images but expected {len(test_dataset)}."
        )

    baseline_accuracy = baseline_correct / total
    crop_accuracy = crop_correct / total
    hierarchical_accuracy = hierarchical_correct / total

    metrics = {
        "experiment": "PlantDoc crop-first hierarchical evaluation",
        "device": str(device),
        "test_images": total,
        "disease_classes": len(disease_classes),
        "crop_classes": len(crop_classes),
        "baseline_disease_accuracy": round(baseline_accuracy, 4),
        "crop_accuracy": round(crop_accuracy, 4),
        "hierarchical_disease_accuracy": round(
            hierarchical_accuracy, 4
        ),
        "hierarchical_change_vs_baseline_percentage_points": round(
            (hierarchical_accuracy - baseline_accuracy) * 100, 2
        ),
        "hierarchical_correct_when_crop_correct": (
            round(
                crop_correct_hierarchical_correct / crop_correct, 4
            )
            if crop_correct
            else None
        ),
        "warning": (
            "Exploratory evaluation on PlantDoc. Results do not establish "
            "real-world field reliability. This test set has been used "
            "in previous experiments, so treat comparisons as exploratory."
        ),
    }

    with PREDICTIONS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\nRESULTS")
    print("-" * 40)
    print(f"Baseline disease accuracy: {baseline_accuracy:.2%}")
    print(f"Crop classification accuracy: {crop_accuracy:.2%}")
    print(f"Hierarchical disease accuracy: {hierarchical_accuracy:.2%}")
    print(
        "Change vs baseline: "
        f"{metrics['hierarchical_change_vs_baseline_percentage_points']:+.2f} "
        "percentage points"
    )
    print("\nSaved metrics:", METRICS_PATH.resolve())
    print("Saved predictions:", PREDICTIONS_PATH.resolve())
    print("Production files were not modified.")


if __name__ == "__main__":
    main()
