# Making the classifier accurate

## The problem today

`ml/data/train/` holds 40 images that are **solid-colour squares** - one
colour, one unique value per image, 224x224. They are placeholders, not
leaves. A model trained on them has never seen a leaf, so a real photo
scores around 0.55 and the API correctly answers "retake the photo".

```
$ python -c "from PIL import Image; import numpy as np; \
    a=np.asarray(Image.open('ml/data/train/healthy/healthy_0.jpg').convert('RGB')); \
    print(len(np.unique(a.reshape(-1,3), axis=0)), 'unique colours')"
1 unique colours
```

Two classes (`healthy`, `leaf_blight`) are also too few: the knowledge
base has ten sections, so the model cannot tell wheat rust from cotton
leaf curl.

## The target

Ten classes, one per knowledge-base section, named `<crop>_<disease>`:

| Class | Knowledge base section |
|---|---|
| `rice_healthy` | Rice - Healthy |
| `rice_leaf_blight` | Rice - Leaf Blight |
| `wheat_healthy` | Wheat - Healthy |
| `wheat_rust` | Wheat - Rust |
| `cotton_healthy` | Cotton - Healthy |
| `cotton_leaf_curl_virus` | Cotton - Leaf Curl Virus |
| `maize_healthy` | Maize - Healthy |
| `maize_leaf_blight` | Maize - Leaf Blight |
| `tomato_healthy` | Tomato - Healthy |
| `tomato_blight` | Tomato - Blight |

The name matters: the graph asks the knowledge base for
`<disease> <crop>` and the matcher splits it on `_`, so
`rice_leaf_blight` becomes the query "rice leaf blight rice" and finds
the "Rice - Leaf Blight" section. Change the folder names and the
retrieval breaks, so keep them exactly as above.

## The data

Free, publicly available datasets. **Check each licence yourself before
publishing anything derived from it.**

| Crop | Dataset | Notes |
|---|---|---|
| maize, tomato | PlantVillage (Kaggle) | 54k images, 38 classes. Covers maize (Northern Leaf Blight, Common Rust, Gray Leaf Spot) and tomato (Early/Late blight, Septoria, ...) well. Class folders are named like `Tomato___Early_blight` - map them onto the target names. |
| rice | a rice-leaf-disease set (Kaggle) | Several exist with different class counts. Pick one, print the folder names, and map them. |
| wheat | wheat disease images (Kaggle / Plant Pathology 2020 FGVC7) | Rust, leaf blight, healthy. |
| cotton | cotton leaf curl virus images (Kaggle) | Mostly leaf curl virus plus healthy. |
| field photos | PlantDoc | ~2.5k real field photos across 13 species. Use a slice of it as a **separate test set**, not for training, so you can measure how the model does on real photographs rather than lab images. |

PlantVillage images are single leaves on a uniform background, so a model
trained only on them can be over-confident on field photos. Mixing in
some field photos, or at least testing on them, keeps the 0.70
threshold honest.

## The steps

### 1. Prepare the folders

```
ml/data/train/rice_healthy/...
ml/data/train/rice_leaf_blight/...
...  (ten folders)
ml/data/test/<same ten names>/...
```

`ml/training/train.py` derives the class list from the folder names and
saves it to `ml/training/artifacts/metrics.json`, so the names above
flow straight into the model.

### 2. Train

```bash
python ml/training/train.py --data ml/data --epochs 10 --out ml/training/artifacts
```

### 3. Export to ONNX

```bash
python ml/export_edge/export_onnx.py \
    --weights ml/training/artifacts/model.pt \
    --metrics ml/training/artifacts/metrics.json \
    --output ml/export_edge/artifacts
```

The ONNX preprocessing must match the training transform exactly
(`Resize(256)` + `CenterCrop(224)` + ImageNet normalisation).
`agents/graph.py` implements that in numpy and
`tests/test_graph.py::test_onnx_preprocess_matches_torch_transform`
asserts the two agree to within 1e-6, so the export and the API cannot
drift apart.

### 4. Check the confidence on real photos

```bash
python ml/pipelines/evaluate_model.py \
    --onnx ml/export_edge/artifacts/classifier.onnx \
    --data ml/data/test
```

The report gives overall and per-class accuracy, a confusion matrix, and
`would_diagnose_rate` - the share of images the API would actually
answer for instead of asking for a retake. That last number is the one
to watch: on the model that is in the repository today it is **0.0**,
because the model was trained on solid-colour squares.

A quick look at the same thing without the tool:

```bash
python - <<'PY'
from agents.graph import _predict_image
for photo in ("tests/rice.jpg", "tests/maize.webp"):
    print(photo, _predict_image(open(photo, "rb").read()))
PY
```

Real photos should land **above 0.70** for a confident diagnosis. If
they do not, the model is not ready - do not lower the threshold to
hide it.

### 5. Commit the model

The model is a build artifact and is not in git history yet:

```bash
brew install git-lfs && git lfs install
git lfs track "*.onnx"
git add .gitattributes ml/export_edge/artifacts/classifier.onnx \
        ml/export_edge/artifacts/classes.json \
        ml/export_edge/artifacts/manifest.json
git commit -m "feat: retrain the classifier on real leaf data"
git push origin main
```

Render and GitHub then pull the LFS object.

## What the API does with the result

`agents/graph.py` looks for the ONNX export first and falls back to the
PyTorch checkpoint, so both work. With `CA_SAGEMAKER_ENDPOINT` set it
uses SageMaker instead and the bundled model is ignored.

The confidence threshold is `CONFIDENCE_MIN = 0.70` in `agents/graph.py`
and `HEALTHY_MIN = 0.85` for the extra "a healthy result below 0.85 is
not sure" rule. Both are deliberate: below them the API says "retake
the photo" rather than guessing, and it says it in the farmer's
language with a helpline.
