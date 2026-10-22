# Phase 4 - controlled experiment V7

## Hypothesis

**V6 was under-trained, not over-fitted.** Its validation loss was still
falling monotonically at the last epoch and its best epoch was the last
epoch, so the run was stopped before it converged.

Evidence from the recorded history (`metrics.json`):

| Run | lr | best epoch | train_loss (last) | val_loss trend | diagnosis |
|---|---|---|---|---|---|
| V5-A | 1e-4 | 12 / 15 | 0.217 | 0.957 (ep6) -> 1.173 (ep15) | **over-fitting** from ~ep6 |
| V5-B | 1e-4 | 8 / 15 | 0.217 | 1.103 (ep5) -> 1.239 (ep15) | **over-fitting** |
| V6 | 3e-5 | **15 / 15** | 1.331 | 3.090 -> 1.507, still falling | **under-trained** |

V4 is V5-A truncated at 5 epochs (identical loss curve for epochs 1-5).

The lower learning rate plus label smoothing (V6) converges more slowly,
so 15 epochs was not enough for it to reach the same place V5-A reached
by epoch 6.

## The experiment

One variable changes: **epochs 15 -> 40**.

Everything else is identical to V6:

| Setting | Value |
|---|---|
| learning rate | 0.00003 |
| label smoothing | 0.1 |
| class weighting | off |
| batch size | 16 |
| seed | 42 |
| architecture | EfficientNet-B0, ImageNet weights, 27-way head |
| augmentation | Resize(256) + RandomResizedCrop(224, 0.75-1.0) + HFlip + Rotation(15) + ColorJitter |
| split | identical (seed 42, 15% per class held out for validation) |
| device | MPS |

```bash
python ml/training/train_plantdoc.py \
    --data ml/training/experiments/plantdoc_deduplicated_v2/data \
    --epochs 40 \
    --lr 0.00003 \
    --label-smoothing 0.1 \
    --batch-size 16 \
    --out ml/training/experiments/plantdoc_27class_v7
```

## Why it is worth running

The comparison (`plantdoc_v4_v6_comparison/comparison_summary.json`)
shows V6 already has the best test accuracy (0.6578) and the best macro
F1 (0.6419) of the four runs, while also having the worst validation
accuracy (0.6628). If V6 is simply under-trained, giving it the epochs
it needs should raise validation accuracy past V5-A's 0.6916 and, if the
ranking is real, the held-out numbers as well.

If validation accuracy plateaus well before epoch 40, that is itself the
answer: the model is at its ceiling for this data, and the remaining
errors are label noise and visually indistinguishable class pairs
(corn leaf blight vs corn gray leaf spot; tomato mosaic virus vs
tomato yellow virus), which no amount of extra training will fix.

## Cost

Observed wall-clock: V5-A 79.6 min / 15 epochs, V5-B 30.5 min / 15,
V6 38.4 min / 15. That is 2.0-5.3 min per epoch on the MPS device, so
40 epochs is roughly **80-210 minutes**.

The model selection is unchanged: the best checkpoint by validation
accuracy is kept, and the held-out test set is only read once at the
end.
