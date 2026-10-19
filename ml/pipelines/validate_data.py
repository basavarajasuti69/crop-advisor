"""Data validation step for SageMaker Pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    input_dir = Path("/opt/ml/processing/input/train")
    output_dir = Path("/opt/ml/processing/output/validation_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in input_dir.iterdir() if d.is_dir()]
    total_images = 0
    for cls in classes:
        cls_dir = input_dir / cls
        total_images += sum(1 for _ in cls_dir.glob("*") if _.is_file())

    report = {
        "classes": classes,
        "num_classes": len(classes),
        "total_images": total_images,
        "validation_passed": len(classes) >= 2 and total_images > 0,
    }

    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
