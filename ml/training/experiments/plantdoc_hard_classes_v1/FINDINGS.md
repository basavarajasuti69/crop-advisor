# Phase 3 - failure analysis of the PlantDoc 27-class models

Date: 2026-10-22
Inputs: `plantdoc_v4_v6_comparison_v2` (the four checkpoints scored on
both the validation and the test split) and the
`plantdoc_deduplicated_v2` dataset.

Everything in this document is read-only analysis. **No label was
changed, no image was moved or deleted, and no dataset file was
modified.** The candidates listed here are for a human to decide on.

Artifacts:

| File | What |
|---|---|
| `class_support.csv` | train / validation / test counts for all 27 classes |
| `hard_class_report.json` | per-class metrics, confusions, cross-model agreement |
| `confusion_pairs.csv` | the confusion counts between related disease pairs |
| `resolution_shift.csv` | median pixel count, train vs test, per class |
| `label_review_candidates.csv` | filename-based cross-class candidates |
| `contact_sheets/` | per-class train vs test contact sheets |
| `evidence/` | close-ups of the individual suspect images |

---

## 1. The headline: the failures are systematic, not checkpoint-specific

For each test image of a hard class, how many of the four
independently trained models (V4, V5-A, V5-B, V6 - different learning
rates, label smoothing, class weighting and epoch budgets) classify it
correctly:

| Class | test images | 0 of 4 correct | never correct by any model |
|---|---|---|---|
| Tomato leaf mosaic virus | 10 | 7 | **7** |
| Cherry leaf | 10 | 6 | **6** |
| Tomato leaf bacterial spot | 8 | 5 | **5** |
| Corn leaf blight | 10 | 4 | **4** |
| Tomato mold leaf | 6 | 3 | **3** |
| Potato leaf late blight | 5 | 2 | **2** |
| Potato leaf early blight | 7 | 1 | 1 |
| Corn Gray leaf spot | 3 | 1 | 1 |

For the first five classes most of the test images are missed by
*every* model. A better-tuned checkpoint would not fix these. The
information needed to separate the classes is either not in the images
or is contradicted by the labels the models were trained on.

By contrast `Potato leaf early blight` is mostly solvable: 3 of its 7
test images are right for all four models and only 1 is right for none.
The class is hard because its *labels* are inconsistent, not because
the images are ambiguous - see section 3.2.

## 2. Class support

| Class | train used | validation | test | train:test |
|---|---|---|---|---|
| Tomato leaf mosaic virus | 37 | 7 | 10 | 4.4 |
| Cherry leaf | 40 | 7 | 10 | 4.7 |
| grape leaf | 48 | 8 | 12 | 4.7 |
| Tomato leaf yellow virus | 59 | 10 | 5 | 13.8 |
| Tomato mold leaf | 72 | 13 | 6 | 14.2 |
| Corn leaf blight | 151 | 27 | 10 | 17.8 |
| Potato leaf late blight | 82 | 14 | 5 | 19.2 |
| Corn Gray leaf spot | 53 | 9 | 3 | 20.7 |
| Squash Powdery mildew leaf | 105 | 19 | 6 | 20.7 |
| **total** | **1972** | **347** | **225** | |

Two things stand out.

* **Tomato leaf mosaic virus has 37 training images and 7 validation
  images** - the smallest training set of the 27 classes. Its recall
  is 0.20 on the test set and 0.30 on validation for every model.
* **The train:test ratio is very uneven.** Several classes have 15-20
  training images for every test image, while others have 4.7. With 3
  to 10 test images per class, one image is 10-33% of a class's
  accuracy, so the per-class test numbers for 12 of the 27 classes are
  not stable enough to guide a decision.

## 3. Label contamination, with evidence

The following are **candidates for human review**, identified by
filename evidence and confirmed by looking at the images. The images
are in `evidence/`.

### 3.1 Corn leaf blight contains 9 Corn Gray Leaf Spot images (5.1%)

`train/Corn leaf blight/` contains:

```
2-3 Gray leaf spot BRUCE.jpg      Corn-Gray-Leaf-Spot-7-31-F2.jpg
Corn-gray-leaf-spot-8-14-15-F1.jpg  Corn-gray-leaf-spot-8-21.jpg
Corn-gray-leaf-spot-F2-8-7-15.jpg   corn-gray-leaf-spot-f5.jpg
corn_gray_leaf_spot.jpg             gray-leaf-spot-lesions-F4.jpg
gray-leaf-spot-severe--.jpg
```

All nine show **small, discrete, rectangular tan lesions running
parallel to the leaf veins** - the Cercospora Gray Leaf Spot symptom -
and not the long, cigar-shaped grey-green Northern Corn Leaf Blight
lesions. See `evidence/corn_blight_vs_gray_candidates.jpg`.

This is the direct cause of the confusion the mission asks about. The
confusion is **one-directional**:

| Model | Corn leaf blight -> Corn Gray leaf spot | Corn Gray leaf spot -> Corn leaf blight |
|---|---|---|
| V4 | 7 | 1 |
| V5-A | 6 | 1 |
| V5-B | 5 | 1 |
| V6 | 4 | 1 |

With 5% of the blight training images really being gray spot, the
model is being taught to call gray spot "blight". Removing the
contamination should collapse this confusion, and it should also lift
`Corn Gray leaf spot` precision, which is 0.20-0.22 for every model.

### 3.2 Potato leaf early blight contains tomato images (4.8% train, 28.6% test)

`train/Potato leaf early blight/` contains five images whose filenames
name tomato:

```
500px-Tomato_Early_Blight_Leaf.jpg
Early_Blight_of_Tomato177.jpg
Early_blight_on_tomato_leaves_7871930010-Fairfax.jpg
early-blight-or-target-spot-alternaria-solani-lesions-on-a-tomato-by9j8r.jpg
early-blight-or-targfet-spot-alternaria-solani-lesions-on-a-tomato-ACG8W6.jpg
```

and the **test** folder contains two more:

```
test/Potato leaf early blight/earlyblightpotato.jpg
test/Potato leaf early blight/early-blight-or-target-spot-alternaria-solani-lesions-on-a-tomato-AXK6AY.jpg
```

All of them are **tomato leaves** - serrated, hairy, compound tomato
leaflets, visibly different from the smooth ovate leaflets of a potato.
See `evidence/potato_early_blight_evidence.jpg`.

The `earlyblightpotato.jpg` case is the sharpest one. The **same
photograph** (identical "S. Rideau" watermark, identical leaf and soil)
exists twice in the dataset:

| Copy | Location | Label |
|---|---|---|
| A | `test/Potato leaf early blight/earlyblightpotato.jpg` | Potato leaf early blight |
| B | `train/Tomato Early blight leaf/earlyblightpotato.jpg` (deduplicated_v1) | Tomato Early blight leaf |

The perceptual audit in `plantdoc_deduplicated_v2` found the pair and
removed **copy B**, the copy that was in the *training* set. Copy A, the
copy in the *test* set, was kept. The leaf is visually a tomato leaf,
so the surviving copy carries the label that is more likely to be
wrong. For a class with only 7 test images, that is 14% of the class.

This also explains why the models predict `Tomato Early blight leaf`
for potato early blight images (V5-B: 2, V6: 1): the model is
responding to the visual evidence, and the label is what disagrees.

### 3.3 Tomato Septoria leaf spot contains other diseases (3.6%)

```
Tomato%20bacterial%20spot%20closeup-050S.JPG.jpg   <- bacterial spot
confused-early-blight-septoria-zoom.jpg             <- curator already flagged it
early+blight+on+leaf+yel+halo+IMG_2313.JPG.jpg      <- early blight
tomato-early-blight-e1437503226856.jpg              <- early blight
septoria+on+rudbeckia.jpg                           <- rudbeckia, not tomato
```

The bacterial spot file is the important one: it is the direct cause of
`Tomato leaf bacterial spot -> Tomato Septoria leaf spot` (4 times for
V4) and `-> Tomato Early blight leaf` (3 times for V5-B).

### 3.4 Tomato mold leaf contains other conditions (7-9%)

See `evidence/mold_train_suspects.jpg`:

| File | What the filename says | What the image shows |
|---|---|---|
| `tylc.jpg` | tomato yellow leaf curl virus | 4-panel plant with upward curling, yellowed leaflets - TYLC |
| `tomato_Leaf-roll-physiological-7059-SLoewen_thm.jpg` | physiological leaf roll | cupped, rolled green leaflet |
| `Tomato%20greenhouse%20leaf%20edema%20and%20distortion2F.JPG.jpg` | leaf edema and distortion | a single dark green leaf, no lesions |
| `gray-leaf-spot-300x225.jpg` | gray leaf spot | necrotic blotches |
| `tomato-gray-leaf-spot.jpg` | gray leaf spot | dark spots with yellow haloes |
| `Botrytis-Tomato-Seedlings-2-...jpg` | Botrytis | grey mould on a seedling |
| `grey-mould-botrytis-cinerea-...jpg` | Botrytis cinerea | dense grey fuzz on the underside |
| `latest?cb=20100723105221.jpg` | meaningless | yellow, distorted, curled leaflets |

`leaf_mold_nightshade1x500.jpg` and `leaf_mold_nightshade2x500.jpg`
were checked and **are** consistent with tomato leaf mold (the same
pathogen, *Passalora fulva*, infects nightshade), so they are not
flagged. Botrytis is a genuinely different disease from tomato leaf
mold, but the class name says "mold" and the dataset has no
Botrytis class, so these may be the curator's intended bucket - that is
a judgement call for a human, not an error to assert.

### 3.5 Other cross-crop candidates

* `train/Bell_pepper leaf spot/` contains `Septoria-leaf-spot.jpg` and
  `gray-leaf-spot-disease-on-260nw-559799659.jpg`.
* `train/Squash Powdery mildew leaf/` contains a `downy-mildew` file -
  downy mildew and powdery mildew are different diseases.
* `train/Tomato leaf yellow virus/` contains
  `8.--Virus-Damaged-foliage-at-top-versus-healthy,-non-infected-bottom-foliage.jpg`,
  a two-panel comparison image.

## 4. Why Tomato mold leaf is hard, if the labels are right

The test labels for this class do look reasonable - the six images are
genuinely about tomato leaf mold. The problem is elsewhere.

* **The training set is 85 field photographs of whole tomato plants**
  and leaf clusters, mostly with diffuse yellow and olive blotches -
  symptoms that are hard to separate from Septoria leaf spot, early
  blight, TYLC and plain nutrient deficiency.
* **The test set mixes domains**: two lab photographs on kraft paper
  (`49 Leafmold Bottom`, `50 Leafmold Top`), three field photographs,
  and **one image that is a University of Minnesota extension slide
  with text** (`BIGSD.jpg`). A slide screenshot is not a leaf photo;
  all four models predict something else for it.
* The class also has 13 validation images against 72 training images,
  and 6 test images, so its accuracy moves in 17% steps.

So for this class the honest conclusion is: *the training signal is
diluted by visually similar diseases and a handful of misfiled images,
and the test set contains a non-photograph.* Retuning will not fix it.

## 5. Resolution shift between train and test

`resolution_shift.csv` has the full table. The extremes:

| Class | train median px | test median px | ratio |
|---|---|---|---|
| Tomato leaf mosaic virus | 114,600 | 578,118 | **5.04x** |
| Potato leaf early blight | 347,310 | 786,432 | 2.26x |
| Apple leaf | 726,843 | 1,526,200 | 2.10x |
| Tomato leaf | 1,248,000 | 1,920,000 | 1.54x |
| ... | | | |
| Potato leaf late blight | 424,076 | 189,200 | 0.45x |
| Tomato leaf late blight | 673,920 | 240,590 | 0.36x |
| Corn rust leaf | 566,052 | 128,736 | **0.23x** |

The mosaic virus class - the one with 37 training images and 0.20
recall - is also the one with the largest train/test resolution
mismatch, 5x. A model trained on 120k-pixel images and tested on
580k-pixel images has never seen that scale.

## 6. Are the four models actually different?

No. A paired McNemar test on the 225 test images gives, for every
pair of models, p >= 0.608 - the differences in test accuracy between
V4, V5-A, V5-B and V6 are **not statistically significant**. The
95% Wilson interval on V6's 0.6578 is [0.5936, 0.7167] and on V5-A's
0.6533 it is [0.5890, 0.7125]; the intervals overlap almost
completely.

The four runs are four draws from the same distribution. The one place
where the models *do* separate cleanly is validation macro F1, which
the checkpoints were selected on:

| Model | val accuracy | val macro F1 |
|---|---|---|
| V5-A | 0.6916 | **0.6695** |
| V5-B | 0.6744 | 0.6588 |
| V6 | 0.6628 | 0.6178 |
| V4 | 0.6455 | 0.5930 |

## 7. What this means for the next experiment

Ranked by how much each one is worth, and by how much evidence there
is for it:

1. **Let V6 converge.** V6's validation loss fell monotonically for
   all 15 epochs (3.09 -> 1.51) and its best epoch was its *last*
   epoch, so it was still improving when the budget ran out. Until it
   converges, "V6 vs V5-A" is not a fair comparison and every other
   configuration is being measured against an under-trained baseline.
   This is a pure budget change, needs no dataset decision, and is the
   prerequisite for everything else.
2. **Fix the Corn leaf blight contamination** (9 images, 5.1%). It has
   the clearest evidence and the cleanest expected effect: the
   blight -> gray spot confusion should collapse. It needs a human
   decision on whether to move the 9 images to the Corn Gray leaf spot
   folder or drop them, and it should be a separate experiment so its
   effect can be attributed.
3. **Fix the potato/tomato contamination** (5 train, 2 test). It needs
   the same human decision, and it touches the test set, so it should
   be decided before any further test-set scoring.
4. **Re-label or drop the `Tomato mold leaf` outliers** and reconsider
   the class. It has the lowest information content of the 27.
5. **Do not add a crop class** until the label quality of the existing
   ones is fixed.

## 8. What was deliberately not done

* No ground-truth label was changed, on the strength of a model
  disagreeing or on the strength of a filename alone.
* No image was deleted from the dataset.
* The `plantdoc_deduplicated_v1` copy is intact, so every file removed
  in `deduplicated_v2` is still recoverable.
* The single known byte-identical duplicate
  (`Potato leaf early blight/18028_1.jpg` = `Potato leaf late blight/24064_1.jpg`)
  was left alone. It is a genuine contradiction - the same image
  presented under two labels - and it lands in the training half of
  the split, so it caps the potato pair rather than leaking into
  validation. Removing it is a curation decision, not an automatic fix.
