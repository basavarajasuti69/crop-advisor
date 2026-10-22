# Machine learning - models, data and experiments

This folder holds the training code, the datasets and every experiment
of the Crop Disease Advisor classifier. The serving path lives in
`agents/classifier.py` and is deliberately free of training code.

## The short version

| | |
|---|---|
| What is being trained | EfficientNet-B0, ImageNet-pretrained, a 27-way head |
| Data | PlantDoc field photographs, deduplicated (`plantdoc_deduplicated_v2`) |
| Classes | 27, shared by the train and the test folders |
| Selection | **validation accuracy on 347 images held out of the train folder** |
| Test set | 225 images, **reused by every experiment** - an exploratory comparison, not an independent benchmark |
| Deployed today | a 2-class placeholder model, **not** the 27-class model |

The 27-class model is **not** the model that ships. The export in
`ml/export_edge/artifacts/` is a 2-class `healthy` / `leaf_blight`
model trained on 40 solid-colour placeholder squares. The 27-class
checkpoints live in `ml/training/experiments/` and have not been
exported or deployed. See [MODEL_CARD.md](MODEL_CARD.md).

## Layout

```
ml/
├── data/
│   └── field_evaluation/PlantDoc-Dataset/   the raw dataset
├── training/
│   ├── train_plantdoc.py                     the training script
│   ├── compare_plantdoc_v4_v6.py             the comparison tool
│   ├── analyse_plantdoc_hard_classes.py      the failure analysis
│   ├── export (in ../export_edge/)
│   ├── artifacts/                            the 2-class placeholder model
│   └── experiments/                          every run, its metrics and its
│                                            comparison artifacts
└── export_edge/
    ├── export_onnx.py                        PyTorch -> ONNX
    ├── export_plantdoc.py                    export a PlantDoc experiment
    └── artifacts/                            the deployed export
```

## Reproducible setup

```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the training and evaluation code:

```bash
pip install torch torchvision scikit-learn pandas pillow
# optional, for the ONNX path:
pip install onnxruntime
```

The environment this project was developed in is Python 3.13,
torch 2.14.1, torchvision 0.29.1, scikit-learn 1.7.2, onnxruntime
1.31.0, on Apple Silicon with MPS. The training code uses MPS when it
is available and falls back to CPU.

## Running a training experiment

```bash
python ml/training/train_plantdoc.py \
    --data ml/training/experiments/plantdoc_deduplicated_v2/data \
    --epochs 40 \
    --lr 0.00003 \
    --label-smoothing 0.1 \
    --out ml/training/experiments/plantdoc_27class_v7b
```

The script refuses to write into a directory that already has files in
it, so an experiment can never overwrite an earlier one.

Useful options, all off by default so the V1-V6 runs stay reproducible:

| Option | Effect |
|---|---|
| `--seed` | seed for the split and the initialisation (default 42) |
| `--validation-fraction` | share of each class held out for validation (default 0.15) |
| `--lr-scheduler {none,cosine}` | learning-rate schedule (default none) |
| `--early-stopping-patience N` | stop after N epochs without validation improvement (0 = off) |
| `--early-stopping-min-epochs N` | never stop before epoch N |
| `--class-weighting` | weight the loss inversely to class frequency |

## The split

The validation set is carved out of the **train** folder, 15% per
class, and the test folder is never used for training or for model
selection. The split is deterministic: `build_split()` in
`train_plantdoc.py` seeds a per-class shuffle with `seed + label`, so
the same seed always produces the same 1,972 / 347 / 225 split.

`compare_plantdoc_v4_v6.py` imports that same function rather than
re-implementing it, so an evaluation cannot drift away from the split
the checkpoints were selected on. `tests/test_plantdoc_split.py` pins
the behaviour.

## Comparing experiments

```bash
python ml/training/compare_plantdoc_v4_v6.py \
    --out plantdoc_v4_v6_comparison_v2
```

It writes to a **new** directory and never writes into an experiment
directory. Every experiment is scored on both the validation and the
test split, with per-class precision, recall, F1 and support, a
confusion matrix, the correct and incorrect predictions, and the
uncalibrated confidence values.

## Analysing failures

```bash
python ml/training/analyse_plantdoc_hard_classes.py \
    --out plantdoc_hard_classes_v1
```

Writes the class support table, the per-class confusions, the
cross-model agreement, the train/test resolution shift and contact
sheets for the hard classes, into a new directory.

## Exporting a checkpoint

```bash
python ml/export_edge/export_plantdoc.py --list
python ml/export_edge/export_plantdoc.py \
    --experiment plantdoc_27class_v7b \
    --output ml/export_edge/artifacts_plantdoc_v7b
```

To serve the new export without overwriting the one that ships:

```bash
CA_MODEL_DIR=ml/export_edge/artifacts_plantdoc_v7b \
    uvicorn main:app --reload
```

## The documents in this folder

| File | What it is |
|---|---|
| [MODEL_CARD.md](MODEL_CARD.md) | what the 27-class model is, what it scores, what it cannot do |
| [DATASETS.md](DATASETS.md) | the original data plan for the 10-class model |
| [experiments/plantdoc_audit_v1/AUDIT.md](experiments/plantdoc_audit_v1/AUDIT.md) | the phase 1 code and data audit |
| [experiments/plantdoc_audit_v1/PROVENANCE.md](experiments/plantdoc_audit_v1/PROVENANCE.md) | the hyper-parameters of the runs that did not record them |
| [experiments/plantdoc_hard_classes_v1/FINDINGS.md](experiments/plantdoc_hard_classes_v1/FINDINGS.md) | the phase 3 failure analysis |
| [experiments/V7_PLAN.md](experiments/V7_PLAN.md) | the plan for the V7 run |
