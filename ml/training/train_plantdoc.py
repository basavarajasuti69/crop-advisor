"""Train an experimental EfficientNet-B0 model on PlantDoc.

The split, the class order and the evaluation transform are the three
things that have to stay identical between a training run and any later
evaluation of its checkpoint. The split and the class order are
therefore defined once, here, in ``resolve_classes`` and ``build_split``,
and ``compare_plantdoc_v4_v6.py`` imports them rather than
re-implementing them.

The label space is the sorted intersection of the train and test folder
names, minus ``EXCLUDED_CLASS``. That is the behaviour of every run
from V1 onwards and it is kept, so the checkpoints stay comparable.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


SEED = 42
EXCLUDED_CLASS = "Tomato two spotted spider mites leaf"
VALIDATION_FRACTION = 0.15


def resolve_classes(
    train_classes: list[str],
    test_classes: list[str],
    excluded_class: str = EXCLUDED_CLASS,
) -> list[str]:
    """The label space: the sorted train/test intersection, minus exclusions.

    Kept as the intersection because every run from V1 onwards used it
    and changing it would make the checkpoints incomparable.
    """
    return sorted(
        (set(train_classes) & set(test_classes)) - {excluded_class}
    )


def build_split(
    train_samples: list[tuple[str, int]],
    train_classes: list[str],
    test_samples: list[tuple[str, int]],
    test_classes: list[str],
    classes: list[str],
    seed: int = SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict:
    """Deterministic per-class split of the train folder, plus the test set.

    The validation set is carved out of the **train** folder, 15% per
    class, so the test folder is never used for training or for model
    selection. The per-class shuffle is seeded with
    ``seed + train_label_index``, so the split is identical on every
    machine and every run with the same seed - which is what lets a
    later evaluation reproduce the exact validation set.

    Returns the index lists and the label maps that align the two
    ImageFolder label spaces with the shared class order.
    """
    train_class_map = {name: index for index, name in enumerate(train_classes)}
    test_class_map = {name: index for index, name in enumerate(test_classes)}
    shared_label = {name: index for index, name in enumerate(classes)}

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for class_name in classes:
        source_label = train_class_map[class_name]
        indices = [
            i for i, (_, label) in enumerate(train_samples)
            if label == source_label
        ]
        random.Random(seed + source_label).shuffle(indices)

        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

        test_source_label = test_class_map[class_name]
        test_indices.extend(
            i for i, (_, label) in enumerate(test_samples)
            if label == test_source_label
        )

    return {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "class_to_new": {
            train_class_map[name]: shared_label[name] for name in classes
        },
        "test_class_to_new": {
            test_class_map[name]: shared_label[name] for name in classes
        },
        "shared_label": shared_label,
    }


def _macro_f1(per_class: dict[str, dict[str, float]], classes: list[str]) -> float:
    """Mean F1 over the classes that have at least one example."""
    supported = [per_class[name]["f1"] for name in classes if per_class[name]["support"]]
    if not supported:
        return 0.0
    return sum(supported) / len(supported)


def _git_commit() -> str | None:
    """The commit the run was started from, when git can answer."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def per_class_metrics(
    confusion: Counter,
    classes: list[str],
) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support per class, from a confusion counter.

    Computed from the confusion matrix rather than a library so the
    numbers can be checked against the matrix that is written out.
    """
    per_class = {}
    for name in classes:
        true_positive = confusion[(name, name)]
        support = sum(confusion[(name, p)] for p in classes)
        predicted = sum(confusion[(a, name)] for a in classes)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return per_class


class MappedSubset(torch.utils.data.Dataset):
    """Subset a dataset and safely remap its labels."""

    def __init__(self, dataset, indices, label_map):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map
        # (path, remapped label) for every member, in order. Lets an
        # evaluation report which file produced which prediction.
        self.samples = [
            (
                dataset.samples[i][0],
                label_map[dataset.samples[i][1]],
            )
            for i in indices
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, original_label = self.dataset[
            self.indices[position]
        ]
        return image, self.label_map[original_label]


def evaluate(model, loader, loss_fn, device, class_names=None):
    """Average loss, accuracy and the confusion counter over a loader.

    ``class_names`` is only needed for the confusion counter: it maps
    the label indices back to the class names.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion: Counter = Counter()

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

            if class_names is not None:
                for actual, predicted in zip(
                    labels.cpu().tolist(), predictions.cpu().tolist()
                ):
                    confusion[
                        (class_names[actual], class_names[predicted])
                    ] += 1

    if total == 0:
        raise ValueError("Evaluation dataset is empty.")

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "confusion": confusion,
        "correct": correct,
        "total": total,
    }


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
    "--label-smoothing",
    type=float,
    default=0.0,
    help="Label smoothing factor for cross-entropy loss (0.0 to 1.0).",
    )
    parser.add_argument(
        "--class-weighting",
        action="store_true",
        help="Weight training loss inversely to class frequency.",
    )
    parser.add_argument(
        "--out",
        default="ml/training/experiments/plantdoc_27class_v1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed for the split and for the initialisation.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=VALIDATION_FRACTION,
        help="Share of each class held out of the train folder for validation.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Learning-rate schedule. 'none' reproduces the V1-V6 runs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement. 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=0,
        help="Never stop before this many epochs have run.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1.")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative.")

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

    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    classes = resolve_classes(
        augmented_dataset.classes,
        full_test_dataset.classes,
    )

    if len(classes) < 2:
        parser.error("Not enough shared classes were found.")

    # The split lives in build_split so that an evaluation of this
    # checkpoint can reproduce the exact same validation set.
    split = build_split(
        augmented_dataset.samples,
        augmented_dataset.classes,
        full_test_dataset.samples,
        full_test_dataset.classes,
        classes,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )

    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]
    test_indices = split["test_indices"]

    # Map labels without modifying the original datasets.
    class_to_new = split["class_to_new"]
    test_class_to_new = split["test_class_to_new"]

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

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        print("LR scheduler: cosine annealing")
    else:
        print("LR scheduler: none")

    if args.class_weighting:
        class_counts = torch.bincount(
            torch.tensor(
                [label for _, label in train_dataset],
                dtype=torch.long,
            ),
            minlength=len(classes),
        ).float()

        if torch.any(class_counts == 0):
            raise ValueError(
                "At least one class has no training examples."
            )

        class_weights = (
            class_counts.sum()
            / (len(classes) * class_counts)
        ).to(device)

        loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
        )

        print("Class weighting: enabled")
        print(
            "Class weights:",
            {
                name: round(class_weights[index].item(), 3)
                for index, name in enumerate(classes)
            },
        )
    else:
        loss_fn = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
        )
        print("Class weighting: disabled")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_accuracy = -1.0
    best_epoch = 0
    history = []
    epochs_without_improvement = 0
    stopped_early = False
    started_at = time.time()

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

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / seen
        validation = evaluate(
            model, validation_loader, loss_fn, device
        )
        val_loss = validation["loss"]
        val_accuracy = validation["accuracy"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(val_loss, 4),
            "validation_accuracy": round(val_accuracy, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
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
            epochs_without_improvement = 0
            torch.save(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        patience = args.early_stopping_patience
        if (
            patience
            and epoch + 1 >= args.early_stopping_min_epochs
            and epochs_without_improvement >= patience
        ):
            stopped_early = True
            print(
                f"Early stopping: no validation improvement for "
                f"{patience} epochs (best epoch {best_epoch})."
            )
            break

    elapsed_seconds = round(time.time() - started_at, 1)

    model.load_state_dict(torch.load(
        output_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    ))

    test_result = evaluate(
        model, test_loader, loss_fn, device, class_names=classes
    )
    validation_result = evaluate(
        model, validation_loader, loss_fn, device, class_names=classes
    )

    train_class_counts = Counter(
        label for _, label in train_dataset
    )
    validation_class_counts = Counter(
        label for _, label in validation_dataset
    )
    test_class_counts = Counter(
        label for _, label in test_dataset
    )

    def counts_for(counter):
        return {
            name: counter.get(index, 0) for index, name in enumerate(classes)
        }

    metrics = {
        "label_smoothing": args.label_smoothing,
        "model": "efficientnet_b0",
        "pretrained_weights": "IMAGENET1K_V1",
        "dataset": "PlantDoc",
        "classes": classes,
        "excluded_class": EXCLUDED_CLASS,
        "seed": args.seed,
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_epochs": args.early_stopping_min_epochs,
        "class_weighting": args.class_weighting,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "optimizer": "Adam",
        "weight_decay": 0,
        "batch_size": args.batch_size,
        "data": str(data_dir),
        "validation_fraction": args.validation_fraction,
        "augmentation": {
            "resize": 256,
            "random_resized_crop": {"size": 224, "scale": [0.75, 1.0]},
            "random_horizontal_flip": True,
            "random_rotation_degrees": 15,
            "color_jitter": {
                "brightness": 0.15,
                "contrast": 0.15,
                "saturation": 0.10,
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "eval_transform": "Resize(256) + CenterCrop(224)",
        },
        "best_epoch": best_epoch,
        "training_images": len(train_indices),
        "validation_images": len(validation_indices),
        "test_images": len(test_indices),
        "train_class_counts": counts_for(train_class_counts),
        "validation_class_counts": counts_for(validation_class_counts),
        "test_class_counts": counts_for(test_class_counts),
        "best_validation_accuracy": round(best_validation_accuracy, 4),
        "best_validation_macro_f1": round(
            _macro_f1(per_class_metrics(
                validation_result["confusion"], classes
            ), classes), 6
        ),
        "best_validation_per_class": per_class_metrics(
            validation_result["confusion"], classes
        ),
        "test_loss": round(test_result["loss"], 4),
        "test_accuracy": round(test_result["accuracy"], 4),
        "test_macro_f1": round(
            _macro_f1(per_class_metrics(
                test_result["confusion"], classes
            ), classes), 6
        ),
        "test_per_class": per_class_metrics(
            test_result["confusion"], classes
        ),
        "elapsed_seconds": elapsed_seconds,
        "git_commit": _git_commit(),
        "history": history,
        "warning": (
            "PlantDoc test accuracy is not proof of field reliability. "
            "Check for duplicate or related images and evaluate on a "
            "separate, independently labelled field-photo collection. "
            "The test set has been scored by every experiment in this "
            "series, so it is an exploratory comparison set and not an "
            "independent benchmark."
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
    print(f"Best validation macro F1: {metrics['best_validation_macro_f1']:.4f}")
    print(f"Held-out test accuracy: {test_result['accuracy']:.2%}")
    print(f"Held-out test macro F1: {metrics['test_macro_f1']:.4f}")
    print(f"Elapsed: {elapsed_seconds}s")
    print(f"Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
