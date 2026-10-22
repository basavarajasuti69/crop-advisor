# Experiment provenance - PlantDoc 27-class runs

These are the hyperparameters of the runs that predate the
`lr` / `batch_size` / `data` fields in `train_plantdoc.py`.

**Why this file exists:** V4, V5-A, V5-B and V6 were trained before
commit `0793b82` added `lr`, `batch_size` and `data` to the metrics
file, so their own `metrics.json` does not contain them. Without this
record the four headline runs cannot be reproduced from the repository.

`source` says where each value comes from:

* `recorded` - written by the training script into `metrics.json`.
* `reconstructed` - the value is the script default at the time of the
  run (`--lr` default `0.0001`, `--batch-size` default `16`) and is
  consistent with the recorded loss curve, but is not in the artifact.
  Treat it as a documented assumption, not a measurement.

Everything else in the table was read directly from each run's
`metrics.json`.

## The runs

| Field | V4 | V5-A | V5-B | V6 |
|---|---|---|---|---|
| directory | `plantdoc_27class_v4` | `plantdoc_27class_v5a` | `plantdoc_27class_v5b` | `plantdoc_27class_v6` |
| architecture | efficientnet_b0 | efficientnet_b0 | efficientnet_b0 | efficientnet_b0 |
| pretrained weights | ImageNet1K_V1 | ImageNet1K_V1 | ImageNet1K_V1 | ImageNet1K_V1 |
| head | 27-way | 27-way | 27-way | 27-way |
| epochs requested | 5 (recorded) | 15 (recorded) | 15 (recorded) | 15 (recorded) |
| best epoch | 5 (recorded) | 12 (recorded) | 8 (recorded) | 15 (recorded) |
| learning rate | 0.0001 (reconstructed) | 0.0001 (reconstructed) | 0.0001 (reconstructed) | 0.00003 (recorded) |
| batch size | 16 (reconstructed) | 16 (reconstructed) | 16 (reconstructed) | 16 (reconstructed) |
| optimizer | Adam (reconstructed) | Adam (reconstructed) | Adam (reconstructed) | Adam (reconstructed) |
| weight decay | 0 (reconstructed) | 0 (reconstructed) | 0 (reconstructed) | 0 (reconstructed) |
| lr schedule | none | none | none | none |
| label smoothing | 0.0 (reconstructed) | 0.0 (reconstructed) | 0.0 (reconstructed) | 0.1 (recorded) |
| class weighting | False (recorded) | False (recorded) | True (recorded) | False (recorded) |
| seed | 42 (recorded) | 42 (recorded) | 42 (recorded) | 42 (recorded) |
| device | mps (recorded) | mps (recorded) | mps (recorded) | mps (recorded) |
| dataset | `plantdoc_deduplicated_v2/data` (reconstructed) | same (reconstructed) | same (reconstructed) | same (reconstructed) |
| train images | 1972 (recorded) | 1972 (recorded) | 1972 (recorded) | 1972 (recorded) |
| validation images | 347 (recorded) | 347 (recorded) | 347 (recorded) | 347 (recorded) |
| test images | 225 (recorded) | 225 (recorded) | 225 (recorded) | 225 (recorded) |
| best validation accuracy | 0.6455 (recorded) | 0.6916 (recorded) | 0.6744 (recorded) | 0.6628 (recorded) |
| test accuracy | 0.6400 (recorded) | 0.6533 (recorded) | 0.6400 (recorded) | 0.6578 (recorded) |

## Why the dataset is a reconstruction

`plantdoc_deduplicated_v2/data` is the only root whose train folder
yields 2,321 images, which is the count implied by the recorded
`training_images` (1,972) + `validation_images` (347) + the 2 excluded
`Tomato two spotted spider mites leaf` images. The other two roots in
the repository do not produce those numbers:

| Root | train images | implied by the recorded metrics |
|---|---|---|
| `ml/data/field_evaluation/PlantDoc-Dataset` | 2,336 | no |
| `ml/training/experiments/plantdoc_deduplicated_v1/data` | 2,325 | no |
| `ml/training/experiments/plantdoc_deduplicated_v2/data` | **2,321** | **yes** |

The split itself is fully deterministic and is not a reconstruction: it
is `random.Random(42 + train_label_index)` per class over the
alphabetically sorted `ImageFolder` sample list, 15% held out. The
same code reproduces the same 1,972 / 347 / 225 split for all four
runs, and the four runs share the same seed and the same class order.

## The training recipe, identical for all four runs

```
train transform : Resize(256) -> RandomResizedCrop(224, 0.75-1.0)
                  -> RandomHorizontalFlip -> RandomRotation(15)
                  -> ColorJitter(brightness .15, contrast .15, saturation .10)
                  -> ToTensor -> Normalize(ImageNet)
eval transform  : Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
```

## Caveats

* MPS kernels are not bit-deterministic, so a rerun reproduces the
  split and the starting weights but not necessarily the exact metric
  values.
* The test set has been scored by every one of these runs. It is an
  exploratory comparison set, not an independent benchmark.
