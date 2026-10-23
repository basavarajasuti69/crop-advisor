# V9 training and evaluation plan

## Status

V9 has been trained (`plantdoc_27class_v9/`, 40 epochs requested, stopped
early at epoch 32, best epoch 24). It is evaluated but did NOT beat V8
on the held-out test set. See `V9_COMPARISON.md`.

## Reproducible procedure

### 1. Dataset (do not touch the original)

Source: `ml/training/experiments/plantdoc_deduplicated_v2/data`
Cleaned: `ml/training/experiments/plantdoc_deduplicated_v3/data`
  - removed the byte-identical duplicate pair
    (early blight 18028_1.jpg = late blight 24064_1.jpg)
  - test split byte-identical to v2

Cleaning is reproducible via `ml/training/build_v9_dataset.py` from a
reviewed manifest (`v9_cleaning_manifest.csv`). To remove more images,
edit the manifest (set reviewer_decision) and rebuild.

### 2. Training

```bash
python ml/training/train_plantdoc.py \
    --data ml/training/experiments/plantdoc_deduplicated_v3/data \
    --epochs 40 \
    --lr 0.00003 \
    --label-smoothing 0.1 \
    --lr-scheduler cosine \
    --early-stopping-patience 8 \
    --early-stopping-min-epochs 10 \
    --out ml/training/experiments/plantdoc_27class_v9
```

Model selection: best validation accuracy (checkpoint saved at best
epoch). Test set is NOT used for selection.

### 3. Evaluation (validation first, test once)

```bash
python ml/training/compare_plantdoc_v4_v6.py \
    --experiments plantdoc_27class_v8 plantdoc_27class_v9 \
    --out plantdoc_v8_v9_comparison
```

### 4. Calibration

```bash
python ml/training/assess_calibration.py \
    --experiment plantdoc_27class_v9 \
    --data ml/training/experiments/plantdoc_deduplicated_v3/data \
    --out ml/training/experiments/plantdoc_v9_audit/calibration_v9.json
```

### 5. Export (do not overwrite V8 artifacts)

```bash
python ml/export_edge/export_plantdoc.py \
    --experiment plantdoc_27class_v9 \
    --output ml/export_edge/artifacts_plantdoc_v9
```

### 6. Tests

```bash
python -m pytest -q
```

## Next experiment (V10) - data curation

The hard classes (Tomato mold leaf, Tomato leaf bacterial spot, Tomato
leaf mosaic virus, Cherry leaf, Corn leaf blight, Corn Gray leaf spot)
regressed or stayed flat. The candidate mislabeled images are in
`label_review_report.csv`. After human review and approval, rebuild the
dataset with `build_v9_dataset.py` and retrain. Do not auto-remove.
