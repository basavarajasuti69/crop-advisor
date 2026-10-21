"""Train an experimental crop classifier with augmentation and validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


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
    parser.add_argument("--data", default="ml/data/plantvillage_subset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        default="ml/training/experiments/plantvillage_4class_v2",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    output_dir = Path(args.out)

    if not train_dir.is_dir() or not test_dir.is_dir():
        parser.error(f"Expected {train_dir} and {test_dir}.")

    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --out directory to avoid overwriting results."
        )

    random.seed(42)
    torch.manual_seed(42)

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

    evaluation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Separate dataset instances allow augmentation only for training.
    augmented_dataset = ImageFolder(train_dir, transform=train_transform)
    evaluation_dataset = ImageFolder(
        train_dir,
        transform=evaluation_transform,
    )
    test_dataset = ImageFolder(test_dir, transform=evaluation_transform)

    classes = augmented_dataset.classes

    if classes != test_dataset.classes:
        parser.error("Training and test class names do not match.")

    if len(classes) < 2:
        parser.error("At least two classes are required.")

    # Deterministic, stratified 85/15 split from the training directory.
    class_indices = {i: [] for i in range(len(classes))}
    for index, (_, label) in enumerate(augmented_dataset.samples):
        class_indices[label].append(index)

    train_indices = []
    validation_indices = []

    for label, indices in class_indices.items():
        random.Random(42 + label).shuffle(indices)
        validation_count = max(1, round(len(indices) * 0.15))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    train_loader = DataLoader(
        Subset(augmented_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(evaluation_dataset, validation_indices),
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
    print(f"Classes: {classes}")
    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(validation_indices)}")
    print(f"Test images (held out): {len(test_dataset)}")
    print(f"Output: {output_dir.resolve()}")

    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_validation_accuracy = -1.0
    best_epoch = 0

    output_dir.mkdir(parents=True, exist_ok=True)

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
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_fn, device
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(validation_loss, 4),
            "validation_accuracy": round(validation_accuracy, 4),
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"validation_loss={validation_loss:.4f} | "
            f"validation_accuracy={validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            torch.save(
                model.cpu().state_dict(),
                output_dir / "best_model.pt",
            )
            model.to(device)

    # The test set is evaluated only once, after checkpoint selection.
    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    test_loss, test_accuracy = evaluate(
        model, test_loader, loss_fn, device
    )

    metrics = {
        "model": "efficientnet_b0",
        "classes": classes,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "training_images": len(train_indices),
        "validation_images": len(validation_indices),
        "test_images": len(test_dataset),
        "best_validation_accuracy": round(best_validation_accuracy, 4),
        "final_test_loss": round(test_loss, 4),
        "final_test_accuracy": round(test_accuracy, 4),
        "history": history,
        "warning": (
            "PlantVillage results do not establish reliability on field "
            "photographs. Evaluate on independently labelled field images."
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_validation_accuracy:.2%}")
    print(f"Held-out test accuracy: {test_accuracy:.2%}")
    print(f"Saved model: {output_dir / 'best_model.pt'}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
