# PlantDoc model audit - phase 1

Date: 2026-10-22
Scope: `ml/training/train_plantdoc.py`, experiments V1-V6, the
`plantdoc_deduplicated_v2` dataset, the comparison tooling and the
serving path in `agents/` + `apps/api/`.

Nothing was modified, deleted or retrained to produce this document.
All checks are read-only.

---

## 1. What was inspected

| Area | What |
|---|---|
| Repository | `git status`, `git log`, README, `docs/architecture.md`, `ml/training/DATASETS.md` |
| Training script | `ml/training/train_plantdoc.py` (all 400 lines) |
| Experiments | `plantdoc_27class_v1` .. `_v6`, `plantdoc_deduplicated_v1/v2`, `plantdoc_data_audit_v1`, `plantdoc_perceptual_audit_v1`, `plantdoc_v4_v6_comparison` |
| Tooling | `compare_plantdoc_v4_v6.py`, `ml/pipelines/evaluate_model.py`, `inspect_plantdoc_v*.py`, `analyse_plantdoc_*.py`, `audit_plantdoc_shared_errors.py` |
| Dataset | 2,321 train images / 28 folders, 225 test images / 27 folders, SHA-256 for every file |
| Serving path | `agents/graph.py`, `agents/classifier.py`, `apps/api/main.py`, `ml/export_edge/`, `rag/`, `mcp_servers/crop_knowledge/` |
| Tests | `python -m pytest tests/ -q` -> **150 passed, 4 skipped** |

---

## 2. Verified facts

### 2.1 The training script is correct on the things that matter most

* The validation split is carved out of the **train** folder only
  (`train_plantdoc.py:171-181`), 15% per class, seeded per class with
  `random.Random(SEED + source_label)`. The test folder is never used
  for training or model selection.
* The evaluation transform (`Resize(256) + CenterCrop(224) + ImageNet
  normalise`) is identical in training, in `compare_plantdoc_v4_v6.py`
  and in `agents/classifier.py`.
* Label smoothing is implemented correctly: it is passed straight to
  `nn.CrossEntropyLoss(label_smoothing=...)` on both the weighted and
  the unweighted branch (`train_plantdoc.py:277-293`).
* Class ordering is derived from the sorted train/test intersection and
  is **identical across all seven experiments** (verified by diffing
  `classes.json`).
* The checkpoint head shape is checked against the class list before
  loading, so a mismatched checkpoint fails loudly instead of silently
  permuting labels.
* `MappedSubset` indexes the train and validation transforms over the
  same `ImageFolder` sample ordering, so the split indices line up.

### 2.2 No train/test leakage

| Check | Result |
|---|---|
| SHA-256 identical file in train and test | **0** pairs |
| SHA-256 identical file inside train | **1** pair (the known one) |
| SHA-256 identical file inside test | **0** pairs |
| Perceptual (pHash) near-duplicate across the split | 4 candidates, all already removed from train in `deduplicated_v2` |
| Known duplicate pair in the train/val split | **both land in train**, not in validation |

The 4 removed near-duplicates are still present in
`plantdoc_deduplicated_v1/data`, so nothing was destroyed - `deduplicated_v2`
is a derived copy, and the source is intact.

### 2.3 The known duplicate

`Potato leaf early blight/18028_1.jpg` and
`Potato leaf late blight/24064_1.jpg` are byte-identical.

* Both are in the **training** portion, so there is no validation leak.
* They are shown to the network as two different labels, which is an
  irreducible contradiction. It is a plausible cause of the
  early-blight / late-bright confusion.
* **Not changed.** The instructions are to leave labels and files alone
  without evidence of a labelling error, and an image appearing under
  two folders is a dataset curation problem, not proof that either
  label is wrong. It is recorded here and left for a human decision.

---

## 3. Problems found

### P1 - The four headline runs are not reproducible from their own artifacts (high)

`metrics.json` for V4, V5-A, V5-B and V6 is **missing `lr`,
`batch_size` and `data`**. Those fields were added to the script in
commit `0793b82` *after* those runs, so the artifacts were written
before the fields existed. `compare_plantdoc_v4_v6.py` already reports
this (`lr_recorded: false`).

Consequence: the learning rate of V4/V5-A/V5-B can only be recovered
from the shell history, not from the repository. The values are known
(0.0001 for V4/V5-A/V5-B, 0.00003 for V6) and are written down in
`PROVENANCE.md` next to this file, marked as reconstructed.

### P2 - No early stopping and no learning-rate schedule (high)

`train_plantdoc.py` runs a fixed epoch budget and keeps the best
validation-accuracy checkpoint. There is no scheduler and no patience
based stop. The recorded histories show the two failure modes this
causes:

| Run | best epoch | behaviour |
|---|---|---|
| V4 | 5 / 5 | truncated mid-rise |
| V5-A | 12 / 15 | over-fits after ~epoch 6 |
| V5-B | 8 / 15 | over-fits |
| V6 | 15 / 15 | validation loss still falling - **under-trained** |

V6 is the clearest case: its best epoch is the last epoch and its
validation loss was still decreasing, so the budget stopped it early.

### P3 - Model selection uses accuracy on 347 samples (medium)

One validation sample is 0.29% of the validation set, and accuracy
ignores the fact that 6 classes have fewer than 10 test images. A
selector that breaks ties on macro-F1 or on validation loss would be
more stable, but changing the selector now would invalidate the
comparison with V4-V6, so it is not changed in the same experiment.

### P4 - Validation loss is not comparable between weighted and unweighted runs (low)

`evaluate()` reuses the training `loss_fn`
(`train_plantdoc.py:320-322`), so in V5-B the validation loss is
class-weighted. V5-B's `validation_loss` is therefore not on the same
scale as V4/V5-A/V6's and must not be read as such.

### P5 - The class list is derived from the test folder names (medium, latent)

```python
classes = sorted((set(train_class_map) & set(test_class_map)) - {EXCLUDED_CLASS})
```

The training label space is defined by the intersection with the test
folder. Today the intersection is exactly 27 classes and everything is
consistent, but a class that is absent from the test split would
silently vanish from training. The label space should be defined by
the training folder alone.

### P6 - The deployed model is not the model that was studied (high)

The artifacts that actually ship are a **2-class** model:

```
ml/export_edge/artifacts/classes.json -> ["healthy", "leaf_blight"]
ml/export_edge/artifacts/manifest.json -> num_classes: 2
ml/training/artifacts/metrics.json     -> {"classes": [...], "epochs": 3}
```

`ml/data/train/` holds 40 **solid-colour squares** (verified: 1 unique
colour each). The 27-class PlantDoc work has never been exported or
deployed.

Consequences:

* `apps/api/main.py` `/model` hardcodes `"trained_on": "PlantDoc"` and
  the `DISCLAIMER` says "a model trained on the PlantDoc dataset". Both
  are **false for the artifacts currently in the repository**.
* `README.md` and `ml/training/DATASETS.md` say the same thing. The
  README's "Known limitations" section is the accurate one.

### P7 - The PlantDoc taxonomy cannot drive the current retrieval (high, deployment blocker)

The knowledge base (`rag/source_docs/crop_guide_en.md`) has exactly ten
sections: Rice, Wheat, Cotton, Maize and Tomato, each with `Healthy`
plus one disease. The PlantDoc taxonomy has 13 crops and 27 classes
covering apple, bell pepper, blueberry, cherry, corn, grape, peach,
potato, raspberry, soyabean, squash, strawberry and tomato.

`crop_from_label()` in `agents/classifier.py:66-73` parses
`<crop>_<disease>` and returns `None` when the label has no
underscore. A PlantDoc label such as `Tomato mold leaf` therefore
yields no crop, and `agents/graph.py` falls back to
`state.crop or "rice"` in four places. Deploying the 27-class model
unchanged would retrieve **rice** guidance for a potato leaf.

`SUPPORTED_CROPS` already contains `"rice"`, and the graph defaults to
rice, so the product surface advertises rice support that no trained
model can provide.

### P8 - An aborted V7 run left a partial artifact (medium)

`ml/training/experiments/plantdoc_27class_v7/best_model.pt` exists
(27-way head, written 09:07) with **no `metrics.json`** and no
training process running. The run was interrupted. That directory must
not be reused; a new run needs a new directory.

### P9 - The PlantDoc test set is not an independent benchmark (high, methodological)

The same 225 test images have now been scored by at least four
experiments plus the comparison tooling. Selecting a configuration on
test accuracy would fit the test set. Every number in this project has
to be read as an **exploratory comparison**, and the honest statement
of the model's quality is the validation number.

---

## 4. What is *not* a problem

* Label smoothing: implemented correctly.
* Train/val/test leakage: none found.
* Class-order drift between checkpoints: none found.
* The `looks_like_leaf` gate: rejects **0 of 2,546** PlantDoc images,
  so it does not turn away real leaves from this dataset.
* The comparison tooling already validates class order, head shape and
  preprocessing, and writes to a separate directory.

---

## 5. Consequences for the plan

1. Run the controlled experiment in a **new** directory
   (`plantdoc_27class_v7b`), not `plantdoc_27class_v7`.
2. Choose the checkpoint on **validation** only.
3. Re-evaluate the test set once, for exploration, and report the gap
   between validation and test honestly.
4. Do not touch the deployed artifacts until a 27-class model exists
   with a label mapping into the knowledge base.
