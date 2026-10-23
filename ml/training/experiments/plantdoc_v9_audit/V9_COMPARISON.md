# V9 vs V8 comparison

Both evaluated on the same 225-image PlantDoc test split and the same
347-image validation split (rebuilt from the v2 data by `build_split`).

| Run | best epoch | val acc | val macro F1 | test acc | test macro F1 | ECE (val) |
|---|---|---|---|---|---|---|
| V8 | 19 | 0.6686 | 0.6315 | **0.6933** | **0.6755** | 0.1055 |
| V9 | 24 | **0.6888** | **0.6514** | 0.6667 | 0.6483 | 0.1152 |

## What changed in V9

- Training data: `plantdoc_deduplicated_v3` (removed the byte-identical
  duplicate pair `Potato leaf early blight/18028_1.jpg` =
  `Potato leaf late blight/24064_1.jpg`).
- Learning rate: cosine annealing (was none).
- Early stopping: patience 8, min 10 epochs (was none).
- Label smoothing 0.1, lr 3e-5, no class weighting (unchanged from V6).

## Honest reading

- V9 improved **validation** accuracy (+0.020) and validation macro F1
  (+0.0199) over V8.
- V9 **regressed** on the test set: accuracy -0.0266, macro F1 -0.0272.
- V9's ECE is slightly worse (0.1152 vs 0.1055).
- Per class: 5 improved, 13 worsened, 9 unchanged. The only clear
  winners are the potato classes (early +0.0308, late +0.0606) and
  Tomato Early blight leaf (+0.0867) - consistent with removing the
  potato duplicate and the cosine schedule. The losses are spread across
  apple, blueberry, cherry, peach, soybean, tomato septoria/bacterial.

## Conclusion

V9 is **not** a clear improvement over V8 on the held-out test set. The
validation gain did not transfer. Model selection was done on validation
only (correct), but the test regression shows the validation set is
small (347 images) and noisy. Neither model is production-ready.

## Classes needing more data (support < 10 in test)

Corn Gray leaf spot (3), Potato leaf late blight (5), Tomato leaf yellow
virus (5), Squash Powdery mildew leaf (6), Tomato mold leaf (6),
Potato leaf early blight (7), Raspberry leaf (7), Bell_pepper leaf (8),
Soyabean leaf (8), Tomato leaf (8), Tomato leaf bacterial spot (8).

Classes with worst F1 in both runs: Tomato leaf mosaic virus (0.3333),
Corn Gray leaf spot (0.3333), Cherry leaf (0.2857-0.3529), Tomato mold
leaf (0.3077-0.3333), Tomato leaf bacterial spot (0.3636-0.4286).
