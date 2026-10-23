# V10 Checkpoint Audit Report

Read-only audit. No datasets, checkpoints, manifests, or source code were
modified. No training was run. No evaluation was performed on V10.

## 1. V10 checkpoint inspection

### 1.1 Files present
| File | Size | Type |
|------|------|------|
| `ml/training/experiments/plantdoc_27class_v10/best_model.pt` | 16,447,353 bytes | raw state_dict |
| `ml/training/experiments/plantdoc_27class_v10_reviewed/best_model.pt` | 16,447,353 bytes | raw state_dict |

No `classes.json`, `metrics.json`, training log, or configuration file exists
in either directory.

### 1.2 Architecture
- Type: `dict` (raw `state_dict`, no wrapper)
- Keys: 360
- Backbone: EfficientNet-B0 (`features.*` + `classifier.*`)
- Classifier head: `Linear(1280, 27)` — shape `(27, 1280)` + bias `(27,)`
- Weights: finite, no NaN/Inf, mean -0.000279, std 0.016090

### 1.3 Metadata in checkpoint
**None.** The checkpoint is a bare `state_dict`. It does not contain:
- Class names or class ordering
- Dataset version or path
- Random seed
- Training hyperparameters (lr, epochs, batch size, label smoothing, etc.)
- Optimizer state
- Metrics (accuracy, F1, loss)
- Git commit
- Training/elapsed time

### 1.4 Provenance from shell history (not from artifacts)
The shell history (`~/.zsh_history`) records two training commands:

```
python ml/training/train_plantdoc.py \\
  --data ml/training/experiments/plantdoc_deduplicated_v4/data \\
  --epochs 40 --lr 0.00003 --label-smoothing 0.1 \\
  --lr-scheduler cosine --early-stopping-patience 8 \\
  --early-stopping-min-epochs 10 \\
  --out ml/training/experiments/plantdoc_27class_v10

python ml/training/train_plantdoc.py \\
  --data ml/training/experiments/plantdoc_deduplicated_v4/data \\
  --epochs 40 --lr 0.00003 --label-smoothing 0.1 \\
  --lr-scheduler cosine --early-stopping-patience 8 \\
  --early-stopping-min-epochs 10 \\
  --out ml/training/experiments/plantdoc_27class_v10_reviewed
```

This indicates V10 was trained on `plantdoc_deduplicated_v4/data` (which
contains the byte-identical duplicate pair with conflicting labels and
unreviewed candidate images). However, shell history is not a reproducible
artifact and cannot be treated as provenance.

### 1.5 Checkpoint comparison
| Pair | Max param diff |
|------|---------------|
| V9 vs V10 | 2.85e+03 |
| V9 vs V10_reviewed | 2.48e+03 |
| V10 vs V10_reviewed | 3.72e+02 |
| V8 vs V10 | 2.23e+03 |

The two V10 checkpoints are distinct (not identical), confirming two
separate training runs. Neither matches V8 or V9.

## 2. V9 experiment artifacts (for comparison)

| Field | Value |
|-------|-------|
| Directory | `ml/training/experiments/plantdoc_27class_v9/` |
| Checkpoint | `best_model.pt` (16,447,353 bytes) |
| classes.json | present (27 classes, sorted) |
| metrics.json | present |
| Best epoch | 24 |
| Best val accuracy | 0.6888 |
| Best val macro F1 | 0.649 |
| Test accuracy | 0.6667 |
| Test macro F1 | 0.648335 |
| Training images | 1,970 |
| Validation images | 347 |
| Test images | 225 |
| LR | 3e-5 |
| Label smoothing | 0.1 |
| LR scheduler | cosine |
| Early stopping | patience 8, min 10 epochs |
| Class weighting | False |
| Seed | 42 |
| Batch size | 16 |
| Dataset | `plantdoc_deduplicated_v3/data` |
| Device | MPS |
| Git commit | dae0cff81b805dce2279135e060e6df26a0f32ea |

## 3. Training script analysis (`ml/training/train_plantdoc.py`)

### 3.1 Class ordering
- `resolve_classes(train_classes, test_classes)` computes the sorted
  intersection of train and test folder names, minus `EXCLUDED_CLASS`
  ("Tomato two spotted spider mites leaf").
- Result: 27 classes in alphabetical order.
- This is deterministic given the same folder names.

### 3.2 Evaluation class selection
- The 27 classes are the intersection of train and test folder names.
- The spider-mites class has 2 training images and is excluded.
- `MappedSubset` aligns ImageFolder labels to the shared class order via
  `class_to_new` and `test_class_to_new` mappings.

### 3.3 Checkpoint save/load
- Save: `torch.save(state_dict, output_dir / "best_model.pt")` — raw
  state_dict only, no metadata wrapper.
- Load: `torch.load(path, map_location=device, weights_only=True)` then
  `model.load_state_dict(...)`.
- The checkpoint cannot establish its own class ordering, dataset, or
  training config. It relies on an external `classes.json`.

### 3.4 Safe evaluation without retraining
- A checkpoint can be loaded and evaluated IF a verified `classes.json`
  with the correct class ordering is available.
- The V10 checkpoints have NO `classes.json`. Using V8/V9's classes.json
  would be an assumption, not verification.

## 4. V10 evaluability assessment

**FAIL — V10 cannot be evaluated reliably.**

Reasons:
1. No `classes.json` in the V10 directories. Class ordering is unverified.
2. No `metrics.json`. No recorded evaluation results exist.
3. No training log or configuration file.
4. Provenance is inferred only from shell history (not an artifact).
5. V10 was trained on v4 data, which contains the conflicting duplicate
   pair and unreviewed images — a known data-quality issue.
6. Matching tensor dimensions (27, 1280) does NOT prove matching class
   ordering. The classifier head shape only confirms 27 output classes.

Even if V8/V9's class ordering were assumed (sorted intersection minus
excluded), this would be an unverified assumption, not a measurement.

## 5. Recommended V5-based experiment configuration

To make a V5-based model comparable to V9, match all V9 settings and
change only the dataset:

| Parameter | Value (match V9) | Rationale |
|-----------|-----------------|-----------|
| Architecture | EfficientNet-B0, ImageNet1K_V1 | Same model |
| Epochs requested | 40 | Same budget |
| Learning rate | 3e-5 | Same |
| Label smoothing | 0.1 | Same |
| LR scheduler | cosine | Same |
| Early stopping | patience 8, min 10 epochs | Same |
| Class weighting | False | Same |
| Optimizer | Adam, weight decay 0 | Same |
| Batch size | 16 | Same |
| Seed | 42 | Same |
| Validation fraction | 0.15 | Same |
| Device | MPS | Same |
| **Dataset** | **V5 (changed)** | Only variable |
| **Output dir** | **new (e.g. plantdoc_27class_v11)** | Never overwrite |

Key principle: change only the dataset. All other settings must match V9
exactly so that any performance difference is attributable to the data
cleaning, not to hyperparameter changes.

## Summary

### A. What is known about V10
- Two checkpoints exist (v10, v10_reviewed), both raw state_dicts.
- Architecture: EfficientNet-B0, 27-way classifier head.
- Weights are finite and loadable.
- Shell history suggests training on v4 data with V9-equivalent hyperparameters.
- V10 is distinct from V8 and V9 (parameter weights differ).
- V4 (the training data) contains the conflicting duplicate pair and
  unreviewed candidate images.

### B. What remains unknown
- Class ordering (no classes.json).
- Dataset version used (inferred from shell history, not artifact).
- Training configuration (inferred from shell history).
- Training/validation/test metrics (no metrics.json).
- Whether training completed or was interrupted.
- Whether early stopping triggered and at which epoch.
- Whether the two V10 runs differ in any way other than random seed.

### C. Whether V10 can be evaluated safely
**No.** The class mapping cannot be established from the checkpoint or
its directory. Evaluating with an assumed class ordering would produce
unreliable per-class metrics. The test set has not been scored by V10.

### D. Recommended next action
1. Do not evaluate or deploy V10. Its provenance is unsupported.
2. Train a new model on the validated V5 dataset using V9-identical
   settings (see table above) in a new output directory.
3. Ensure the training script saves `classes.json` and `metrics.json`
   alongside the checkpoint (it already does — V10's are simply missing,
   likely because the run was interrupted or manually cleaned).
4. Verify the new checkpoint's class ordering matches V5's folder
   ordering before any evaluation.
5. Evaluate on the same 225-image test split and the 347-image
   validation split for a fair V9-vs-new comparison.
6. Do not claim improvement unless the evidence (validation metrics,
   per-class F1, calibration) supports it.

No training is authorized by this audit.
