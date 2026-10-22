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
sys.path.insert(0, str(ROOT))

EXPERIMENT = ROOT / "ml/training/experiments/plantdoc_27class_v3"
DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v1/data"
REPORT = EXPERIMENT / "per_class_report.csv"
ERRORS = EXPERIMENT / "misclassified_images.csv"
ERROR_IMAGES = EXPERIMENT / "error_images"

with (EXPERIMENT / "classes.json").open() as f:
    classes = json.load(f)

if isinstance(classes, dict):
    classes = classes.get("classes", classes)

if not isinstance(classes, list):
    raise ValueError("Could not read class names from classes.json")

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
        "Dataset classes don't match model classes.\n"
        f"Model: {classes}\nDataset: {dataset.classes}"
    )

# ImageFolder's alphabetical order should match the saved model order,
# but explicitly map labels to the model's saved class order.
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
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))

checkpoint = torch.load(
    EXPERIMENT / "best_model.pt",
    map_location=device,
    weights_only=True,
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

confusion = [[0 for _ in classes] for _ in classes]
mistakes = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        probabilities = torch.softmax(model(images), dim=1)
        confidences, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):
            actual = labels[i].item()
            predicted = predictions[i].item()
            confidence = confidences[i].item()

            confusion[actual][predicted] += 1
            total += 1
            correct += int(actual == predicted)

            if actual != predicted:
                source = Path(dataset.samples[total - 1][0])
                mistakes.append({
                    "source_path": str(source),
                    "actual_label": classes[actual],
                    "predicted_label": classes[predicted],
                    "confidence": confidence,
                })

print(f"Test accuracy: {correct / total:.2%} ({correct}/{total})")

with REPORT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class", "test_images", "correct",
            "accuracy", "most_common_wrong_prediction",
        ],
    )
    writer.writeheader()

    rows = []
    for i, name in enumerate(classes):
        support = sum(confusion[i])
        hits = confusion[i][i]
        wrong = [
            (confusion[i][j], classes[j])
            for j in range(len(classes))
            if j != i and confusion[i][j] > 0
        ]
        count, wrong_name = max(wrong, default=(0, ""))
        accuracy = hits / support if support else 0

        rows.append({
            "class": name,
            "test_images": support,
            "correct": hits,
            "accuracy": f"{accuracy:.4f}",
            "most_common_wrong_prediction": (
                f"{wrong_name} ({count})" if count else ""
            ),
        })
        writer.writerow(rows[-1])

with ERRORS.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_path", "actual_label",
            "predicted_label", "confidence",
        ],
    )
    writer.writeheader()
    writer.writerows(mistakes)

print(f"Per-class report: {REPORT}")
print(f"Misclassification report: {ERRORS}")
print(f"Misclassified images: {len(mistakes)}")

print("\nLowest-performing classes:")
for row in sorted(rows, key=lambda r: float(r["accuracy"]))[:10]:
    print(
        f'{float(row["accuracy"]):6.1%} '
        f'({row["correct"]}/{row["test_images"]}) '
        f'{row["class"]} | '
        f'Common confusion: {row["most_common_wrong_prediction"]}'
    )

print("\nMost common incorrect predictions:")
for name, count in Counter(
    row["predicted_label"] for row in mistakes
).most_common(10):
    print(f"{count:2} errors predicted as: {name}")

print("\nThis is experimental evaluation only; production was not modified.")
