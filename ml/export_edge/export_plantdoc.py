"""Export a PlantDoc experiment checkpoint to ONNX for the serving path.

The export goes to its own directory, so the model that currently ships
in ``ml/export_edge/artifacts`` is not touched. Point the service at the
new export with the ``CA_MODEL_DIR`` environment variable:

    CA_MODEL_DIR=ml/export_edge/artifacts_plantdoc_v7b \\
        uvicorn main:app --reload

The manifest records what the model was trained on and how it scored,
so the API describes the model that is actually being served rather
than a hardcoded string.

Examples
--------
    # list the experiments that can be exported
    python ml/export_edge/export_plantdoc.py --list

    # export the best checkpoint of one experiment
    python ml/export_edge/export_plantdoc.py \\
        --experiment plantdoc_27class_v7b \\
        --output ml/export_edge/artifacts_plantdoc_v7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.export_edge.export_onnx import export_onnx  # noqa: E402

EXPERIMENTS_DIR = ROOT / "ml/training/experiments"
DEFAULT_OUTPUT = ROOT / "ml/export_edge/artifacts_plantdoc_v7b"


def available() -> list[dict]:
    """Every experiment that has a checkpoint and a metrics file."""
    found = []
    for directory in sorted(EXPERIMENTS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        checkpoint = directory / "best_model.pt"
        metrics = directory / "metrics.json"
        if not (checkpoint.exists() and metrics.exists()):
            continue
        payload = json.loads(metrics.read_text())
        found.append({
            "name": directory.name,
            "best_epoch": payload.get("best_epoch"),
            "best_validation_accuracy": payload.get("best_validation_accuracy"),
            "best_validation_macro_f1": payload.get("best_validation_macro_f1"),
            "test_accuracy": payload.get("test_accuracy"),
            "test_macro_f1": payload.get("test_macro_f1"),
            "classes": len(payload.get("classes", [])),
        })
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        help="Experiment directory name under ml/training/experiments/.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Export directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the experiments that can be exported and exit.",
    )
    args = parser.parse_args()

    if args.list:
        rows = available()
        if not rows:
            print("No experiments with a checkpoint were found.")
            return
        header = (
            f"{'experiment':28s} {'best_ep':>7s} {'val_acc':>8s} "
            f"{'val_f1':>7s} {'test_acc':>8s} {'test_f1':>8s} {'n_cls':>5s}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['name']:28s} "
                f"{str(row['best_epoch']):>7s} "
                f"{str(row['best_validation_accuracy']):>8s} "
                f"{str(row['best_validation_macro_f1']):>7s} "
                f"{str(row['test_accuracy']):>8s} "
                f"{str(row['test_macro_f1']):>8s} "
                f"{row['classes']:>5d}"
            )
        return

    if not args.experiment:
        parser.error("pass --experiment, or --list to see the options")

    experiment = EXPERIMENTS_DIR / args.experiment
    if not experiment.is_dir():
        raise SystemExit(f"Experiment directory not found: {experiment}")
    checkpoint = experiment / "best_model.pt"
    metrics = experiment / "metrics.json"
    if not (checkpoint.exists() and metrics.exists()):
        raise SystemExit(
            f"{experiment} needs both best_model.pt and metrics.json"
        )

    payload = json.loads(metrics.read_text())
    classes = payload.get("classes") or []
    if not classes:
        raise SystemExit(f"{metrics} does not list the class names")

    notes = (
        "Trained on the PlantDoc field-photo dataset. The scores are "
        "measured on a small held-out PlantDoc split and are not "
        "evidence of reliability on real farm photos. The softmax "
        "confidence is uncalibrated."
    )

    export_onnx(
        checkpoint,
        metrics,
        Path(args.output),
        dataset="PlantDoc",
        notes=notes,
    )
    print(f"  experiment: {args.experiment}")
    print(f"  classes:   {len(classes)}")
    print("  dataset:   PlantDoc")
    print(
        f"  validation accuracy: {payload.get('best_validation_accuracy')} "
        f"(best epoch {payload.get('best_epoch')})"
    )
    print(f"  test accuracy:       {payload.get('test_accuracy')}")


if __name__ == "__main__":
    main()
