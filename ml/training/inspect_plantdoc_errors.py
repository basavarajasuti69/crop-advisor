"""Save misclassified PlantDoc test images and prediction details."""

import csv
import shutil
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.training.train_plantdoc import MappedSubset, EXCLUDED_CLASS


ROOT = Path("ml/data/field_evaluation/PlantDoc-Dataset")
EXPERIMENT = Path("ml/training/experiments/plantdoc_27class_v1")
CHECKPOINT = EXPERIMENT / "best_model.pt"
OUTPUT = EXPERIMENT / "error_images"
REPORT = EXPERIMENT / "misclassified_images.csv"

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Missing model checkpoint: {CHECKPOINT}")

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

OUTPUT.mkdir(parents=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

train_original = ImageFolder(ROOT / "train")
test_original = ImageFolder(ROOT / "test", transform=transform)

classes = sorted(
    (set(train_original.classes) & set(test_original.classes))
    - {EXCLUDED_CLASS}
)
class_to_label = {name: i for i, name in enumerate(classes)}

test_label_map = {
    index: class_to_label[name]
    for index, name in enumerate(test_original.classes)
    if name in class_to_label
}

test_indices = [
    i for i, (_, label) in enumerate(test_original.samples)
    if test_original.classes[label] in class_to_label
]

test_dataset = MappedSubset(
    test_original,
    test_indices,
    test_label_map,
)

loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes),
)
model.load_state_dict(
    torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
)
model.to(device)
model.eval()

rows: list[dict[str, object]] = []
correct = 0
total = 0
sample_position = 0

with torch.no_grad():
    for images, labels in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1)
        confidences, predictions = probabilities.max(dim=1)

        for actual, predicted, confidence in zip(
            labels.tolist(),
            predictions.cpu().tolist(),
            confidences.cpu().tolist(),
        ):
            original_index = test_indices[sample_position]
            source_path = Path(test_original.samples[original_index][0])
            sample_position += 1
            total += 1

            if actual == predicted:
                correct += 1
                continue

            filename = (
                f"{len(rows) + 1:03d}_"
                f"actual-{actual}_pred-{predicted}_"
                f"{source_path.name}"
            )
            destination = OUTPUT / filename
            shutil.copy2(source_path, destination)

            rows.append({
                "image": filename,
                "actual_label": classes[actual],
                "predicted_label": classes[predicted],
                "confidence": round(confidence, 4),
                "source_path": str(source_path),
            })

with REPORT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "actual_label",
            "predicted_label",
            "confidence",
            "source_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Test accuracy: {correct / total:.2%} ({correct}/{total})")
print(f"Misclassified images: {len(rows)}")
print(f"CSV report: {REPORT}")
print(f"Copied images: {OUTPUT}")

print("\nFirst 20 mistakes:")
for row in rows[:20]:
    print(
        f'{row["actual_label"]} -> {row["predicted_label"]} '
        f'(confidence {row["confidence"]:.1%})'
    )
