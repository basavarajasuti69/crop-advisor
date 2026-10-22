# Model card — PlantDoc 27-class disease classifier

**Architecture:** EfficientNet-B0, ImageNet-pretrained (`IMAGENET1K_V1`), 27-way classification head
**Input:** 224×224 RGB, ImageNet normalisation
**Framework:** PyTorch, trained on Apple Silicon (MPS)
**Framework:** PyTorch, trained on Apple Silicon (MPS)

## What this model is

A single-label image classifier that, given a photograph of a plant leaf,
predicts one of 27 shared PlantDoc classes covering apple, bell pepper,
blueberry, cherry, corn, grape, peach, potato, raspberry, soyabean,
squash, strawberry and tomato — each as a specific disease (or healthy).

It is **not** the model that ships with this repository. The export in
`ml/export_edge/artifacts/` is a 2-class `healthy` / `leaf_blight`
model trained on 40 solid-colour placeholder squares; it cannot identify
a real leaf disease and the service answers "retake the photo" for real
photos.

## What this model is NOT

- Not a guaranteed diagnosis. It is an informational estimate only.
- Not a substitute for an agriculture officer. Confirm before spraying.
- Not evidence of reliability on real farm photos. It was trained and
  evaluated on the PlantDoc field-photo dataset only.
- Not calibrated. The softmax score is an uncalibrated confidence, not
  the probability of a disease. The API reports
  `confidence_is_calibrated: false`.
- Not able to identify rice. There is no rice class in the trained set,
  and there is no labelled rice dataset in this project. Do not add rice
  as a supported crop without one.
- Not complete coverage: the knowledge base covers rice, wheat, cotton,
  maize and tomato. 22 of the 27 classes have no knowledge-base section
  and receive an officer referral rather than treatment advice.

## Data

`ml/training/experiments/plantdoc_deduplicated_v2/data`

| Split | Images | Classes | Notes |
|---|---|---|---|
| Train | 1,972 | 27 | 15% of each class held out for validation |
| Validation | 347 | 27 | held out of train; used for model selection |
| Test | 225 | 27 | reused by every experiment |

The excluded class `Tomato two spotted spider mites leaf` had only two
training images. The dataset is deduplicated against exact and perceptual
duplicates; one exact duplicate pair remains in training under two
different labels (`Potato leaf early blight/18028_1.jpg` =
`Potato leaf late blight/24064_1.jpg`) and was left untouched because
neither label can be proven wrong.

The train and test images were audited at SHA-256 level: there is **no
exact-hash overlap** between the train and test splits.

## Training recipe (V4–V6 baseline)

```
transform: Resize(256) -> RandomResizedCrop(224, 0.75-1.0)
           -> RandomHorizontalFlip -> RandomRotation(15)
           -> ColorJitter(brightness .15, contrast .15, saturation .10)
           -> ToTensor -> Normalize(ImageNet)
eval:    Resize(256) -> CenterCrop(224) -> Normalize(ImageNet)
optimizer: Adam, weight decay 0
lr scheduler: none
early stopping: none (fixed epoch budget)
label smoothing: 0.0 (V4/V5-A/V5-B) / 0.1 (V6)
class weighting: off (V4/V5-A/V6) / on (V5-B)
seed: 42
device: MPS (Apple Silicon)
```

## Measured results (validation is the honest signal; test is exploratory)

The test set has been scored by every experiment, so test numbers are
not an independent benchmark. Validation accuracy is the correct basis
for comparing runs.

| Run | epochs | lr | label smoothing | class weighting | best epoch | val acc | val macro F1 | test acc | test macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| V4 | 5 | 0.0001 | 0.0 | off | 5 | 0.6455 | 0.5930 | 0.6400 | 0.6077 |
| V5-A | 15 | 0.0001 | 0.0 | off | 12 | **0.6916** | **0.6695** | 0.6533 | 0.6191 |
| V5-B | 15 | 0.0001 | 0.0 | on | 8 | 0.6744 | 0.6588 | 0.6400 | 0.6149 |
| V6 | 15 | 0.00003 | 0.1 | off | 15 | 0.6628 | 0.6178 | **0.6578** | **0.6419** |
| V7c | 40 | 0.00003 | 0.1 | off | 30 | 0.6888 | 0.6610 | **0.6711** | **0.6459** |

V4 is literally V5-A truncated at 5 epochs (identical loss curves).
V5-A's validation loss bottoms at epoch 6 then rises (over-fitting).
V6's validation loss falls for all 15 epochs and its best epoch is its
last — it was still improving when the budget ran out (under-trained).

## V7c experiment — extending the epoch budget (controlled)

Hypothesis: V6 was under-trained (val loss falling at epoch 15, best
epoch = last epoch), so extending the epoch budget from 15 to 40 with
every other hyperparameter identical should let it converge.

Setup: `train_plantdoc.py --epochs 40 --lr 0.00003 --label-smoothing
0.1`, everything else identical to V6, output in
`ml/training/experiments/plantdoc_27class_v7c/` (~82 minutes on MPS).

Result: best epoch 30; val acc 0.6628→0.6888, val macro F1
0.6178→0.6610, test acc 0.6578→0.6711, test macro F1 0.6419→0.6459.
V7c beats V6 overall and beats V5-A on test metrics (though V5-A still
leads on validation accuracy 0.6916 vs 0.6888).

**But the improvement is not broad.** Per-class F1 vs V6 on the hard
classes:

| Class | V6 F1 | V7c F1 | delta |
|---|---|---|---|
| Tomato mold leaf | 0.2857 | 0.1667 | -0.119 |
| Tomato leaf bacterial spot | 0.4615 | 0.2667 | -0.195 |
| Tomato leaf mosaic virus | 0.3333 | 0.3333 | 0.000 |
| Cherry leaf | 0.4444 | 0.4000 | -0.044 |
| Corn leaf blight | 0.7059 | 0.5333 | -0.173 |
| Corn Gray leaf spot | 0.3333 | 0.3333 | 0.000 |
| Potato leaf early blight | 0.7692 | 0.7143 | -0.055 |
| Potato leaf late blight | 0.6667 | 0.5455 | -0.121 |

9 of 27 classes improved, 11 worsened. The hard classes — exactly the
ones with label contamination and domain shift — got **worse**, not
better.

**Overfitting evidence:** val_loss bottoms at epoch 19 (1.4589) then
rises to 1.5481 at epoch 40 while train_loss keeps falling
(0.8408 at epoch 40, still descending). The best checkpoint (epoch 30)
is past the val-loss minimum. Longer training helped the majority
classes but overfit the minority/ambiguous ones.

**Conclusion:** extending the budget fixed V6's under-training for the
aggregate metrics, but it did not fix the hard classes. Those are
limited by label contamination (e.g. 9 Corn-leaf-blight training images
that are actually Gray Leaf Spot) and train/test domain shift — more
epochs cannot repair mislabeled data. The next experiment should be
data curation (remove the verified corn blight/gray-spot and
potato/tomato cross-crop misfilings), not more epochs.

## Known failure classes (Phase 3)

Classes where most test images are missed by **all four** models:
- Tomato leaf mosaic virus (7/10 never correct by any model)
- Cherry leaf (6/10)
- Tomato leaf bacterial spot (5/8)
- Corn leaf blight (4/10)
- Tomato mold leaf (3/6)

Small classes are the limiting factor: mosaic virus has 37 training /
7 validation / 10 test images; recall is 0.20. With 3–10 test images
per class, one image is 10–33% of a class accuracy, so per-class test
numbers are not stable.

## Confusion the model makes

- **Corn leaf blight → Corn Gray leaf spot** (7/6/5/4 times across the
  four models). Root cause: 9 of 178 training images in the blight
  folder are actually Gray Leaf Spot images (filenames say "gray leaf
  spot" and show Cercospora lesions). This is label contamination, not
  model weakness.
- **Tomato leaf bacterial spot → Tomato Septoria leaf spot** (4 times
  for V4) and **→ Tomato Early blight leaf** (3 times for V5-B).
- **Tomato leaf mosaic virus → Tomato leaf yellow virus** (3 times for
  every model); precision 1.000 but recall 0.20.
- **Potato early blight vs late blight**: 5 training images in the
  early-blight folder are tomato leaves, and 2 of 7 test images are
  tomato leaves; the models predict "tomato early blight" on them.

## The classes with knowledge-base guidance

Only these four classes map to a knowledge-base section:

| Class | KB section |
|---|---|
| Tomato leaf | Tomato - Healthy |
| Tomato leaf late blight | Tomato - Blight |
| Tomato Early blight leaf | Tomato - Blight |
| Corn leaf blight | Maize - Leaf Blight |

All other 23 classes receive an honest "no guidance" answer with an
agriculture-officer referral, rather than another crop's advice.

## Reproducing

```bash
pip install torch torchvision scikit-learn pandas pillow onnxruntime

python ml/training/train_plantdoc.py \
    --data ml/training/experiments/plantdoc_deduplicated_v2/data \
    --epochs 40 \
    --lr 0.00003 \
    --label-smoothing 0.1 \
    --out ml/training/experiments/plantdoc_27class_v7c

python ml/training/compare_plantdoc_v4_v6.py \
    --experiments plantdoc_27class_v4 plantdoc_27class_v5a \
                plantdoc_27class_v5b plantdoc_27class_v6 \
                plantdoc_27class_v7c \
    --out plantdoc_v4_v6_comparison_v3

python ml/export_edge/export_plantdoc.py \
    --experiment plantdoc_27class_v7c \
    --output ml/export_edge/artifacts_plantdoc_v7c

CA_MODEL_DIR=ml/export_edge/artifacts_plantdoc_v7c \
    uvicorn main:app --reload
```

`best_model.pt` is not committed (gitignored); export the checkpoint you
want to serve.

## Deployment checklist (not yet satisfied)

- [ ] A 27-class checkpoint has converged (V7c in progress)
- [ ] That checkpoint exported to ONNX with an honest manifest
- [ ] The export evaluated on an independent field-photo set (PlantDoc
      is not independent — it has been reused)
- [ ] The knowledge base coverage matches the deployed classes
- [ ] Unit tests for inference and preprocessing pass

## Before the model is called production-ready

Evaluate on a separately collected, independently labelled set of real
farm photos. PlantDoc accuracy does not prove field reliability, and
the confidence score is not calibrated.
