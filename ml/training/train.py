"""Train and evaluate a crop-disease image classifier."""

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


def build_model(num_classes: int) -> nn.Module:
    """Create an ImageNet-pretrained EfficientNet-B0."""
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return average loss and accuracy for a dataset."""
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
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("The evaluation dataset contains no images.")

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="ml/data/plantvillage_subset",
        help="Dataset containing train/ and test/ class folders.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        default="ml/training/experiments/plantvillage_4class",
        help="Separate output directory for this experiment.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive.")

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    output_dir = Path(args.out)

    if not train_dir.is_dir() or not test_dir.is_dir():
        parser.error(
            f"Expected both {train_dir} and {test_dir} to exist."
        )

    # Refuse to overwrite any existing experiment.
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new --out path to preserve existing experiment files."
        )

    random.seed(42)
    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Dataset: {data_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = ImageFolder(
        train_dir,
        transform=train_transform,
    )
    test_dataset = ImageFolder(
        test_dir,
        transform=test_transform,
    )

    classes = train_dataset.classes

    if classes != test_dataset.classes:
        parser.error(
            "Training and test folders must contain exactly the same classes."
        )

    if len(classes) < 2:
        parser.error("At least two classes are required.")

    print(f"Classes: {classes}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

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
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
        )

        result = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_accuracy, 4),
        }
        history.append(result)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_accuracy={test_accuracy:.2%}"
        )

    # Save only to the experiment directory, never the production artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"

    torch.save(model.cpu().state_dict(), model_path)

    metrics = {
        "model": "efficientnet_b0",
        "classes": classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "history": history,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "warning": (
            "Experimental PlantVillage results are not evidence of "
            "reliable performance on real farm photographs."
        ),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\nExperiment complete.")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print(
        f"Final test accuracy: "
        f"{history[-1]['test_accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()