"""Train an experimental EfficientNet-B0 model on PlantDoc."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


SEED = 42
EXCLUDED_CLASS = "Tomato two spotted spider mites leaf"


class MappedSubset(torch.utils.data.Dataset):
    """Subset a dataset and safely remap its labels."""

    def __init__(self, dataset, indices, label_map):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, original_label = self.dataset[
            self.indices[position]
        ]
        return image, self.label_map[original_label]


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Evaluation dataset is empty.")

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="ml/data/field_evaluation/PlantDoc-Dataset",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--out",
        default="ml/training/experiments/plantdoc_27class_v1",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    output_dir = Path(args.out)

    if not train_dir.is_dir() or not test_dir.is_dir():
        parser.error(f"Expected both {train_dir} and {test_dir}.")

    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --out directory to avoid overwriting results."
        )

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

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

    augmented_dataset = ImageFolder(
        train_dir, transform=train_transform
    )
    evaluation_dataset = ImageFolder(
        train_dir, transform=eval_transform
    )
    full_test_dataset = ImageFolder(
        test_dir, transform=eval_transform
    )

    train_class_map = {
        name: index
        for index, name in enumerate(augmented_dataset.classes)
    }
    test_class_map = {
        name: index
        for index, name in enumerate(full_test_dataset.classes)
    }

    classes = sorted(
        (set(train_class_map) & set(test_class_map))
        - {EXCLUDED_CLASS}
    )

    if len(classes) < 2:
        parser.error("Not enough shared classes were found.")

    # Map each dataset's labels to the same shared-class indices.
    shared_label = {name: index for index, name in enumerate(classes)}

    train_indices = []
    validation_indices = []
    test_indices = []

    for class_name in classes:
        source_label = train_class_map[class_name]
        indices = [
            i for i, (_, label) in enumerate(augmented_dataset.samples)
            if label == source_label
        ]
        random.Random(SEED + source_label).shuffle(indices)

        validation_count = max(1, round(len(indices) * 0.15))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

        test_source_label = test_class_map[class_name]
        test_indices.extend(
            i for i, (_, label) in enumerate(full_test_dataset.samples)
            if label == test_source_label
        )

    # Map labels without modifying the original datasets.
    class_to_new = {
        train_class_map[name]: shared_label[name]
        for name in classes
    }

    test_class_to_new = {
        test_class_map[name]: shared_label[name]
        for name in classes
    }

    train_dataset = MappedSubset(
        augmented_dataset,
        train_indices,
        class_to_new,
    )

    validation_dataset = MappedSubset(
        evaluation_dataset,
        validation_indices,
        class_to_new,
    )

    test_dataset = MappedSubset(
        full_test_dataset,
        test_indices,
        test_class_to_new,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Device: {device}")
    print(f"Shared classes: {len(classes)}")
    for name in classes:
        print(f"  {name}")
    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(validation_indices)}")
    print(f"Held-out test images: {len(test_indices)}")
    print(f"Output directory: {output_dir.resolve()}")

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(classes)
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_accuracy = -1.0
    best_epoch = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / seen
        val_loss, val_accuracy = evaluate(
            model, validation_loader, loss_fn, device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(val_loss, 4),
            "validation_accuracy": round(val_accuracy, 4),
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_accuracy:.2%}"
        )

        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(
                model.cpu().state_dict(),
                output_dir / "best_model.pt",
            )
            model.to(device)

    model.load_state_dict(torch.load(
        output_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    ))

    test_loss, test_accuracy = evaluate(
        model, test_loader, loss_fn, device
    )

    metrics = {
        "model": "efficientnet_b0",
        "dataset": "PlantDoc",
        "classes": classes,
        "excluded_class": EXCLUDED_CLASS,
        "seed": SEED,
        "device": str(device),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "training_images": len(train_indices),
        "validation_images": len(validation_indices),
        "test_images": len(test_indices),
        "best_validation_accuracy": round(best_validation_accuracy, 4),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
        "history": history,
        "warning": (
            "PlantDoc test accuracy is not proof of field reliability. "
            "Check for duplicate or related images and evaluate on a "
            "separate, independently labelled field-photo collection."
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    (output_dir / "classes.json").write_text(
        json.dumps(classes, indent=2) + "\n"
    )

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_validation_accuracy:.2%}")
    print(f"Held-out test accuracy: {test_accuracy:.2%}")
    print(f"Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
