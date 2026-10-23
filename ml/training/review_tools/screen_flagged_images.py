from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn


ROOT = Path.cwd()

DATA = ROOT / "ml/training/experiments/plantdoc_deduplicated_v2/data"
MANIFEST = ROOT / "ml/training/experiments/plantdoc_v9_audit/v9_cleaning_manifest.csv"
OUT = ROOT / "ml/training/experiments/plantdoc_v9_audit/model_screening.csv"

MODEL_DIRS = {
    "v8": ROOT / "ml/training/experiments/plantdoc_27class_v8",
    "v9": ROOT / "ml/training/experiments/plantdoc_27class_v9",
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(version: str, folder: Path):
    classes_path = folder / "classes.json"
    checkpoint_path = folder / "best_model.pt"

    if not classes_path.exists():
        raise FileNotFoundError(f"Missing class metadata: {classes_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes),
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded {version}: {len(classes)} classes, device={DEVICE}")
    return model, classes


def predict(model, classes, image_path: Path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    top_values, top_indices = probabilities.topk(min(3, len(classes)))

    top = [
        (classes[index.item()], round(value.item(), 4))
        for value, index in zip(top_values, top_indices)
    ]

    return {
        "label": top[0][0],
        "confidence": top[0][1],
        "top3": top,
    }


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    models_loaded = {}
    class_lists = {}

    for version, folder in MODEL_DIRS.items():
        models_loaded[version], class_lists[version] = load_model(version, folder)

    if class_lists["v8"] != class_lists["v9"]:
        raise ValueError(
            "V8 and V9 class order differs. Refusing to compare predictions."
        )

    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    output_rows = []

    for number, row in enumerate(rows, start=1):
        split = row["split"].strip()
        current_label = row["current_label"].strip()
        relative_path = Path(row["file"].strip())
        image_path = DATA / relative_path

        result = {
            "split": split,
            "current_label": current_label,
            "file": row["file"],
            "reason": row.get("reason", ""),
            "v8_prediction": "",
            "v8_confidence": "",
            "v8_top3": "",
            "v9_prediction": "",
            "v9_confidence": "",
            "v9_top3": "",
            "models_agree": "",
            "both_disagree_with_current": "",
            "screening_priority": "",
            "reviewer_decision": row.get("reviewer_decision", "PENDING"),
            "reviewer_new_label": row.get("reviewer_new_label", ""),
            "reviewer_notes": row.get("reviewer_notes", ""),
        }

        if not image_path.is_file():
            result["screening_priority"] = "FILE_NOT_FOUND"
            output_rows.append(result)
            print(f"[{number}/{len(rows)}] MISSING: {relative_path}")
            continue

        try:
            predictions = {
                version: predict(model, class_lists[version], image_path)
                for version, model in models_loaded.items()
            }
        except Exception as exc:
            result["screening_priority"] = "PREDICTION_ERROR"
            result["reviewer_notes"] = (
                str(result["reviewer_notes"]) + f" Prediction error: {exc}"
            ).strip()
            output_rows.append(result)
            print(f"[{number}/{len(rows)}] ERROR: {relative_path}: {exc}")
            continue

        p8 = predictions["v8"]
        p9 = predictions["v9"]

        result.update({
            "v8_prediction": p8["label"],
            "v8_confidence": p8["confidence"],
            "v8_top3": json.dumps(p8["top3"], ensure_ascii=False),
            "v9_prediction": p9["label"],
            "v9_confidence": p9["confidence"],
            "v9_top3": json.dumps(p9["top3"], ensure_ascii=False),
            "models_agree": p8["label"] == p9["label"],
            "both_disagree_with_current": (
                p8["label"] != current_label
                and p9["label"] != current_label
            ),
        })

        if p8["label"] == p9["label"] == current_label:
            priority = "LOWER_PRIORITY_MODELS_AGREE_WITH_LABEL"
        elif (
            p8["label"] == p9["label"]
            and p8["label"] != current_label
        ):
            priority = "HIGHER_PRIORITY_BOTH_MODELS_DISAGREE"
        elif p8["label"] != p9["label"]:
            priority = "HIGHER_PRIORITY_MODELS_DISAGREE"
        else:
            priority = "INVESTIGATE"

        result["screening_priority"] = priority
        output_rows.append(result)

        print(
            f"[{number}/{len(rows)}] {split} | "
            f"{current_label} | V8={p8['label']} ({p8['confidence']:.2f}) | "
            f"V9={p9['label']} ({p9['confidence']:.2f})"
        )

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nFinished. Results saved to:\n{OUT}")
    print(f"Images in manifest: {len(rows)}")
    print(f"Rows exported: {len(output_rows)}")
    print("No dataset images were changed or deleted.")


if __name__ == "__main__":
    main()
