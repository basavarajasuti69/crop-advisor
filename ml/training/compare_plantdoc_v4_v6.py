"""Compare the PlantDoc experiments on a common test set.

Read-only with respect to every experiment directory: checkpoints,
metrics and reports are opened, never written. All new artifacts go to
a separate comparison directory.

The evaluation deliberately reuses the exact training-time recipe:

* the same class order, read from each experiment's ``classes.json`` and
  checked to be identical across every model;
* the same preprocessing as ``train_plantdoc.py`` -
  ``Resize(256) + CenterCrop(224) + Normalize(ImageNet)``;
* the same architecture - EfficientNet-B0 with a 27-way head;
* the same split - ``build_split`` is imported from
  ``train_plantdoc.py`` rather than re-implemented, so the validation
  set scored here is byte-for-byte the one the checkpoints were selected
  on.

Two splits are scored for every model:

* **validation** - the 347 images held out of the train folder. This
  is the split the checkpoints were selected on, so it is the honest
  signal for comparing models.
* **test** - the 225 held-out PlantDoc images. The same test set has
  been scored by every experiment in this series, so it is an
  exploratory comparison and not an independent benchmark.

Example
-------
    python ml/training/compare_plantdoc_v4_v6.py

    python ml/training/compare_plantdoc_v4_v6.py \\
        --experiments plantdoc_27class_v4 plantdoc_27class_v5a \\
                     plantdoc_27class_v5b plantdoc_27class_v6 \\
        --out plantdoc_v4_v6_comparison
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "ml/training"))

from train_plantdoc import (  # noqa: E402
    MappedSubset,
    build_split,
    resolve_classes,
)

EXPERIMENTS_DIR = ROOT / "ml/training/experiments"
DATA_DIR = EXPERIMENTS_DIR / "plantdoc_deduplicated_v2/data"
PROVENANCE_PATH = EXPERIMENTS_DIR / "plantdoc_audit_v1/provenance.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The preprocessing used by train_plantdoc.py for evaluation.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Metrics recorded per experiment, in a stable order.
METRIC_KEYS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "correct",
    "incorrect",
    "total",
    "mean_max_probability",
    "mean_max_probability_correct",
    "mean_max_probability_incorrect",
]


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model


def load_classes(experiment: Path) -> list[str]:
    classes = json.loads((experiment / "classes.json").read_text())
    if isinstance(classes, dict):
        classes = classes.get("classes", classes)
    if not isinstance(classes, list):
        raise ValueError(f"{experiment}: classes.json is not a list")
    return classes


def load_checkpoint(experiment: Path, num_classes: int):
    state = torch.load(
        experiment / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model = build_model(num_classes)
    # Fail loudly if the head shape does not match the class list.
    head = state.get("classifier.1.weight")
    if head is None:
        raise ValueError(f"{experiment}: checkpoint has no classifier head")
    if head.shape[0] != num_classes:
        raise ValueError(
            f"{experiment}: checkpoint head has {head.shape[0]} outputs "
            f"but classes.json lists {num_classes}"
        )
    model.load_state_dict(state)
    return model


def evaluate(model, dataset, device, class_names):
    """Score a model on a dataset that exposes ``.samples`` as (path, label).

    ``class_names`` maps the label index back to the class name, so the
    same function serves the test set and the validation subset.
    """
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0
    )
    model.eval()
    confusion: Counter = Counter()
    mistakes = []
    correct_rows = []
    total = 0
    sample_index = 0
    correct = 0
    confidence_sum = 0.0
    confidence_correct = 0.0
    confidence_wrong = 0.0

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
                image_path = dataset.samples[sample_index][0]
                sample_index += 1
                actual = class_names[actual_id]
                predicted = class_names[predicted_id]
                confidence = float(confidence)

                confusion[(actual, predicted)] += 1
                total += 1
                confidence_sum += confidence

                row = {
                    "source_path": str(Path(image_path).resolve()),
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "confidence": round(confidence, 6),
                    "confidence_percent": round(confidence * 100, 2),
                }
                if actual_id == predicted_id:
                    correct += 1
                    confidence_correct += confidence
                    correct_rows.append(row)
                else:
                    confidence_wrong += confidence
                    mistakes.append(row)

    return {
        "confusion": confusion,
        "mistakes": mistakes,
        "correct_rows": correct_rows,
        "total": total,
        "correct": correct,
        "confidence_sum": confidence_sum,
        "confidence_correct": confidence_correct,
        "confidence_wrong": confidence_wrong,
    }


def prf(confusion: Counter, classes: list[str]):
    """Per-class precision, recall, F1 and support from the confusion matrix.

    Computed here rather than taken from a library so the numbers can be
    audited against the confusion matrix that is written out.
    """
    per_class = {}
    for name in classes:
        tp = confusion[(name, name)]
        support = sum(confusion[(name, p)] for p in classes)
        predicted = sum(confusion[(a, name)] for a in classes)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
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
            "tp": tp,
            "fp": predicted - tp,
            "fn": support - tp,
        }
    return per_class


def aggregate(per_class: dict) -> dict:
    """Macro and weighted averages over the classes that have support."""
    supported = [m for m in per_class.values() if m["support"]]
    if not supported:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
    total_support = sum(m["support"] for m in supported)
    return {
        "macro_precision": round(
            sum(m["precision"] for m in supported) / len(supported), 6
        ),
        "macro_recall": round(
            sum(m["recall"] for m in supported) / len(supported), 6
        ),
        "macro_f1": round(
            sum(m["f1"] for m in supported) / len(supported), 6
        ),
        "weighted_f1": round(
            sum(m["f1"] * m["support"] for m in supported) / total_support, 6
        ),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "plantdoc_27class_v4",
            "plantdoc_27class_v5a",
            "plantdoc_27class_v5b",
            "plantdoc_27class_v6",
        ],
        help="Experiment directory names under ml/training/experiments.",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_DIR),
        help="Dataset root containing train/ and test/.",
    )
    parser.add_argument(
        "--out",
        default="plantdoc_v4_v6_comparison",
        help="Comparison output directory under ml/training/experiments/.",
    )
    parser.add_argument(
        "--provenance",
        default=str(PROVENANCE_PATH),
        help="JSON record of hyper-parameters the older runs did not record.",
    )
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir = EXPERIMENTS_DIR / args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {out_dir}. "
            "Choose a new --out directory."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # One dataset, one class order, shared by every model.
    test_dataset = datasets.ImageFolder(
        data_root / "test", transform=EVAL_TRANSFORM
    )
    train_eval_dataset = datasets.ImageFolder(
        data_root / "train", transform=EVAL_TRANSFORM
    )
    dataset_classes = test_dataset.classes

    # The class order each model was trained with.
    per_experiment_classes = {}
    for name in args.experiments:
        experiment = EXPERIMENTS_DIR / name
        if not experiment.is_dir():
            raise SystemExit(f"Missing experiment directory: {experiment}")
        per_experiment_classes[name] = load_classes(experiment)

    reference_name = args.experiments[0]
    reference = per_experiment_classes[reference_name]
    for name, classes in per_experiment_classes.items():
        if classes != reference:
            raise SystemExit(
                f"{name} has a different class order from {reference_name}. "
                "The checkpoints would not be comparable."
            )
    if set(reference) != set(dataset_classes):
        raise SystemExit(
            "The model classes and the test folder names do not match."
        )

    # Rebuild the training split with the very function the checkpoints
    # were trained with, so the validation set scored here is the same
    # one the checkpoints were selected on.
    split_classes = resolve_classes(
        train_eval_dataset.classes, test_dataset.classes
    )
    if split_classes != reference:
        raise SystemExit(
            "The rebuilt class order does not match the models' classes.json."
        )
    split = build_split(
        train_eval_dataset.samples,
        train_eval_dataset.classes,
        test_dataset.samples,
        test_dataset.classes,
        split_classes,
    )
    validation_dataset = MappedSubset(
        train_eval_dataset,
        split["validation_indices"],
        split["class_to_new"],
    )
    test_subset = MappedSubset(
        test_dataset,
        split["test_indices"],
        split["test_class_to_new"],
    )

    # Hyper-parameters the older runs never wrote down.
    provenance = {}
    if Path(args.provenance).is_file():
        provenance = json.loads(
            Path(args.provenance).read_text()
        ).get("experiments", {})

    summary = {
        "dataset": str(data_root / "test"),
        "device": str(device),
        "classes": reference,
        "test_images": len(test_subset),
        "validation_images": len(validation_dataset),
        "preprocessing": "Resize(256) + CenterCrop(224) + ImageNet normalize",
        "split": (
            "build_split() from train_plantdoc.py, seed 42, "
            "15% of each class held out of the train folder"
        ),
        "experiments": {},
        "warning": (
            "The test set has been reused across every experiment, so these "
            "numbers are an exploratory comparison, not an independent "
            "benchmark. Use the validation numbers to compare models. "
            "PlantDoc folder labels are noisy and this is not "
            "evidence of reliability on real farm photos. The probabilities "
            "are uncalibrated softmax outputs, not disease probabilities."
        ),
    }

    per_class_rows = []
    comparison_rows = []
    validation_rows = []

    for name in args.experiments:
        experiment = EXPERIMENTS_DIR / name
        model = load_checkpoint(experiment, len(reference))
        model.to(device)

        result = evaluate(model, test_subset, device, reference)
        per_class = prf(result["confusion"], reference)
        aggregate_metrics = aggregate(per_class)

        validation_result = evaluate(
            model, validation_dataset, device, reference
        )
        validation_per_class = prf(
            validation_result["confusion"], reference
        )
        validation_aggregate = aggregate(validation_per_class)

        total = result["total"]
        accuracy = result["correct"] / total if total else 0.0
        metrics = {
            "accuracy": round(accuracy, 6),
            "macro_precision": aggregate_metrics["macro_precision"],
            "macro_recall": aggregate_metrics["macro_recall"],
            "macro_f1": aggregate_metrics["macro_f1"],
            "weighted_f1": aggregate_metrics["weighted_f1"],
            "correct": result["correct"],
            "incorrect": len(result["mistakes"]),
            "total": total,
            "mean_max_probability": round(
                result["confidence_sum"] / total, 6
            ) if total else 0.0,
            "mean_max_probability_correct": round(
                result["confidence_correct"] / result["correct"], 6
            ) if result["correct"] else 0.0,
            "mean_max_probability_incorrect": round(
                result["confidence_wrong"] / len(result["mistakes"]), 6
            ) if result["mistakes"] else 0.0,
        }

        # Hyper-parameters and training facts from the experiment's own file,
        # with the provenance record as a fallback for the runs that were
        # trained before the fields existed.
        training = {}
        metrics_path = experiment / "metrics.json"
        if metrics_path.exists():
            recorded = json.loads(metrics_path.read_text())
            training = {
                key: recorded.get(key)
                for key in (
                    "epochs_requested",
                    "epochs_completed",
                    "best_epoch",
                    "best_validation_accuracy",
                    "best_validation_macro_f1",
                    "label_smoothing",
                    "class_weighting",
                    "seed",
                    "device",
                    "training_images",
                    "validation_images",
                    "test_images",
                    "lr",
                    "learning_rate",
                    "batch_size",
                    "optimizer",
                    "weight_decay",
                    "lr_scheduler",
                    "data",
                    "validation_fraction",
                    "stopped_early",
                    "elapsed_seconds",
                )
            }
            training["lr_recorded"] = (
                "lr" in recorded or "learning_rate" in recorded
            )
            training["batch_size_recorded"] = "batch_size" in recorded

        fallback = provenance.get(name, {})
        for field in ("lr", "batch_size", "data"):
            if training.get(field) is None and field in fallback:
                training[field] = fallback[field]["value"]
                training[f"{field}_source"] = fallback[field]["source"]
            elif field in fallback:
                training[f"{field}_source"] = "recorded"

        validation_metrics = {
            "accuracy": round(
                validation_result["correct"] / validation_result["total"], 6
            ) if validation_result["total"] else 0.0,
            "macro_precision": validation_aggregate["macro_precision"],
            "macro_recall": validation_aggregate["macro_recall"],
            "macro_f1": validation_aggregate["macro_f1"],
            "weighted_f1": validation_aggregate["weighted_f1"],
            "correct": validation_result["correct"],
            "incorrect": validation_result["total"] - validation_result["correct"],
            "total": validation_result["total"],
        }

        summary["experiments"][name] = {
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "training": training,
        }

        comparison_rows.append({"experiment": name, **metrics})
        validation_rows.append({"experiment": name, **validation_metrics})

        for class_name in reference:
            row = {
                "experiment": name,
                "class": class_name,
                **per_class[class_name],
            }
            per_class_rows.append(row)

        # Per-model artifacts.
        model_dir = out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            model_dir / "per_class_metrics.csv",
            [
                "class", "precision", "recall", "f1", "support",
                "tp", "fp", "fn",
            ],
            [
                {
                    "class": class_name,
                    **per_class[class_name],
                }
                for class_name in reference
            ],
        )

        write_csv(
            model_dir / "validation_per_class_metrics.csv",
            [
                "class", "precision", "recall", "f1", "support",
                "tp", "fp", "fn",
            ],
            [
                {
                    "class": class_name,
                    **validation_per_class[class_name],
                }
                for class_name in reference
            ],
        )

        with (model_dir / "confusion_matrix.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual_label"] + reference)
            for actual in reference:
                writer.writerow([
                    actual,
                    *[
                        result["confusion"][(actual, predicted)]
                        for predicted in reference
                    ],
                ])

        mistakes = sorted(
            result["mistakes"],
            key=lambda row: row["confidence"],
            reverse=True,
        )
        write_csv(
            model_dir / "misclassified_images.csv",
            [
                "source_path", "actual_label", "predicted_label",
                "confidence", "confidence_percent",
            ],
            mistakes,
        )
        write_csv(
            model_dir / "correct_images.csv",
            [
                "source_path", "actual_label", "predicted_label",
                "confidence", "confidence_percent",
            ],
            result["correct_rows"],
        )

        (model_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment": name,
                    "metrics": metrics,
                    "validation_metrics": validation_metrics,
                    "training": training,
                    "note": (
                        "Probabilities are uncalibrated softmax outputs, "
                        "not disease probabilities. The test set is reused "
                        "across every experiment in this series; use the "
                        "validation metrics to compare models."
                    ),
                },
                indent=2,
            )
            + "\n"
        )

        print(
            f"{name}: test_acc={metrics['accuracy']:.4f} "
            f"test_macro_f1={metrics['macro_f1']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} "
            f"({metrics['correct']}/{metrics['total']})"
        )

    write_csv(
        out_dir / "comparison.csv",
        ["experiment", *METRIC_KEYS],
        [
            {key: row.get(key) for key in ["experiment", *METRIC_KEYS]}
            for row in comparison_rows
        ],
    )
    write_csv(
        out_dir / "validation_comparison.csv",
        [
            "experiment", "accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "correct", "incorrect", "total",
        ],
        validation_rows,
    )
    write_csv(
        out_dir / "per_class_all_models.csv",
        [
            "experiment", "class", "precision", "recall", "f1",
            "support", "tp", "fp", "fn",
        ],
        per_class_rows,
    )
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"\nComparison written to {out_dir}")


if __name__ == "__main__":
    main()
