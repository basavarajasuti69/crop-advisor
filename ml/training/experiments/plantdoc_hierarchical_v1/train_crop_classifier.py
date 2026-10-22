from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


SEED = 42
ROOT = Path("ml/training/experiments")
DATA = Path("ml/training/experiments/plantdoc_deduplicated_v1/data")
OUT = ROOT / "plantdoc_hierarchical_v1" / "model_output"
MAPPING_PATH = ROOT / "plantdoc_hierarchical_v1" / "crop_mapping.json"


class RemappedDataset(Dataset):
    def __init__(self, base_dataset, indices, label_map):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, original_label = self.base_dataset[self.indices[index]]
        return image, self.label_map[original_label]


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Evaluation dataset is empty.")

    return correct / total


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not DATA.joinpath("train").is_dir():
        raise SystemExit(f"Training directory not found: {DATA / 'train'}")

    if not DATA.joinpath("test").is_dir():
        raise SystemExit(f"Test directory not found: {DATA / 'test'}")

    output_model = OUT / "crop_best_model.pt"
    if output_model.exists():
        raise SystemExit(
            f"Refusing to overwrite existing model: {output_model}"
        )

    mapping = json.loads(MAPPING_PATH.read_text())

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_augmented = ImageFolder(DATA / "train", transform=train_transform)
    train_evaluation = ImageFolder(DATA / "train", transform=eval_transform)
    test_dataset = ImageFolder(DATA / "test", transform=eval_transform)

    excluded_class = "Tomato two spotted spider mites leaf"

    train_classes = set(train_augmented.classes) - {excluded_class}
    test_classes = set(test_dataset.classes)

    if train_classes != test_classes:
        print("Classes in training but not test:", sorted(train_classes - test_classes))
        print("Classes in test but not training:", sorted(test_classes - train_classes))
        raise SystemExit("Shared-class validation failed. Stopping safely.")

    shared_classes = sorted(train_classes)

    if len(shared_classes) != 27:
        raise SystemExit(
            f"Expected 27 shared classes, found {len(shared_classes)}. "
            "Stopping safely."
        )

    if set(shared_classes) != set(mapping):
        print("Classes missing from mapping:", sorted(set(shared_classes) - set(mapping)))
        print("Unexpected mapping classes:", sorted(set(mapping) - set(shared_classes)))
        raise SystemExit(
            "Crop mapping does not match the 27 shared classes. Stopping."
        )

    crop_names = sorted(set(mapping.values()))
    crop_to_id = {name: index for index, name in enumerate(crop_names)}
    class_to_crop_id = {
        train_augmented.class_to_idx[name]: crop_to_id[mapping[name]]
        for name in shared_classes
    }

    train_indices = []
    validation_indices = []

    for class_name in shared_classes:
        class_id = train_augmented.class_to_idx[class_name]
        indices = [
            index
            for index, (_, label) in enumerate(train_augmented.samples)
            if label == class_id
        ]
        random.Random(SEED + class_id).shuffle(indices)
        validation_count = max(1, round(len(indices) * 0.15))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    train_label_map = dict(class_to_crop_id)

    test_label_map = {
        test_dataset.class_to_idx[name]: crop_to_id[mapping[name]]
        for name in test_dataset.classes
    }

    train_set = RemappedDataset(
        train_augmented, train_indices, train_label_map
    )
    validation_set = RemappedDataset(
        train_evaluation, validation_indices, train_label_map
    )

    test_indices = list(range(len(test_dataset)))
    test_set = RemappedDataset(
        test_dataset, test_indices, test_label_map
    )

    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_set, batch_size=16, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=16, shuffle=False, num_workers=0
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Crop categories:", len(crop_names), crop_names)
    print("Training images:", len(train_set))
    print("Validation images:", len(validation_set))
    print("Held-out test images:", len(test_set))
    print("Output:", output_model.resolve())

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(crop_names)
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    best_validation_accuracy = -1.0
    best_epoch = 0
    history = []

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / seen
        validation_accuracy = evaluate(
            model, validation_loader, device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "validation_accuracy": round(validation_accuracy, 4),
        })

        print(
            f"Epoch {epoch + 1}/5 | "
            f"train_loss={train_loss:.4f} | "
            f"val_accuracy={validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            torch.save(model.cpu().state_dict(), output_model)
            model.to(device)

    model.load_state_dict(torch.load(
        output_model, map_location=device, weights_only=True
    ))

    test_accuracy = evaluate(model, test_loader, device)

    metrics = {
        "experiment": "PlantDoc hierarchical crop classifier",
        "crop_names": crop_names,
        "seed": SEED,
        "device": str(device),
        "epochs": 5,
        "best_epoch": best_epoch,
        "training_images": len(train_set),
        "validation_images": len(validation_set),
        "test_images": len(test_set),
        "best_validation_accuracy": round(best_validation_accuracy, 4),
        "test_crop_accuracy": round(test_accuracy, 4),
        "history": history,
        "warning": (
            "Experimental only. Test images are held out from training. "
            "PlantDoc performance does not prove field reliability."
        ),
    }

    (OUT / "crop_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    (OUT / "crop_classes.json").write_text(
        json.dumps(crop_names, indent=2) + "\n"
    )

    print("\nCrop classifier training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_validation_accuracy:.2%}")
    print(f"Held-out test crop accuracy: {test_accuracy:.2%}")
    print("Saved metrics:", (OUT / "crop_metrics.json").resolve())
    print("Production files were not modified.")


if __name__ == "__main__":
    main()
