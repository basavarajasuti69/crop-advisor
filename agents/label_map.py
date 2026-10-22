"""The label vocabulary of the classifiers this service can serve.

Two naming schemes are in use in this repository.

**The 10-class crop-specific scheme** - ``rice_healthy``,
``tomato_blight`` - used by the small placeholder export in
``ml/export_edge/artifacts``. The crop is the part before the first
underscore.

**The 27-class PlantDoc scheme** - ``Tomato mold leaf``,
``Corn leaf blight`` - used by the experiments in
``ml/training/experiments``. The crop is not in the label at all, so it
has to be looked up.

The knowledge base in ``rag/source_docs/crop_guide_en.md`` has exactly
ten sections: rice, wheat, cotton, maize and tomato, each with
``Healthy`` plus one disease. **Most PlantDoc classes have no guidance
behind them.** That is recorded here rather than guessed at, so the
service can say "no guidance for this crop" instead of quietly
retrieving guidance for a different crop.

The knowledge base is not a plant-disease reference: it is a short
guidance sheet per section. A class with no section gets no treatment
advice, only a pointer to an agriculture officer.
"""

from __future__ import annotations


# The ten knowledge-base sections, exactly as they are headed in
# rag/source_docs/crop_guide_en.md.
KB_SECTIONS: tuple[str, ...] = (
    "Rice - Healthy",
    "Rice - Leaf Blight",
    "Wheat - Healthy",
    "Wheat - Rust",
    "Cotton - Healthy",
    "Cotton - Leaf Curl Virus",
    "Maize - Healthy",
    "Maize - Leaf Blight",
    "Tomato - Healthy",
    "Tomato - Blight",
)

# The crops the knowledge base covers. This is the set of crops the
# service is allowed to give crop-specific guidance for. It is NOT the
# set of crops the model can identify - those are different things and
# confusing them is how a potato photo ends up with rice advice.
KB_CROPS: tuple[str, ...] = (
    "rice",
    "wheat",
    "cotton",
    "maize",
    "tomato",
)


# The 27 PlantDoc classes, in the exact order the checkpoints use.
# crop      - the plant the photo shows
# disease   - the condition, phrased for a farmer
# healthy   - True when the class means "no disease"
# kb        - the knowledge-base section for this class, or None when
#             the knowledge base has nothing for it
PLANTDOC_CLASSES: dict[str, dict[str, object]] = {
    "Apple Scab Leaf": {
        "crop": "apple", "disease": "scab", "healthy": False, "kb": None,
    },
    "Apple leaf": {
        "crop": "apple", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Apple rust leaf": {
        "crop": "apple", "disease": "rust", "healthy": False, "kb": None,
    },
    "Bell_pepper leaf": {
        "crop": "bell_pepper", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Bell_pepper leaf spot": {
        "crop": "bell_pepper", "disease": "leaf spot", "healthy": False, "kb": None,
    },
    "Blueberry leaf": {
        "crop": "blueberry", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Cherry leaf": {
        "crop": "cherry", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Corn Gray leaf spot": {
        "crop": "corn", "disease": "gray leaf spot", "healthy": False, "kb": None,
    },
    "Corn leaf blight": {
        "crop": "corn", "disease": "leaf blight", "healthy": False,
        "kb": "Maize - Leaf Blight",
    },
    "Corn rust leaf": {
        "crop": "corn", "disease": "rust", "healthy": False, "kb": None,
    },
    "Peach leaf": {
        "crop": "peach", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Potato leaf early blight": {
        "crop": "potato", "disease": "early blight", "healthy": False, "kb": None,
    },
    "Potato leaf late blight": {
        "crop": "potato", "disease": "late blight", "healthy": False, "kb": None,
    },
    "Raspberry leaf": {
        "crop": "raspberry", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Soyabean leaf": {
        "crop": "soyabean", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Squash Powdery mildew leaf": {
        "crop": "squash", "disease": "powdery mildew", "healthy": False, "kb": None,
    },
    "Strawberry leaf": {
        "crop": "strawberry", "disease": "healthy", "healthy": True, "kb": None,
    },
    "Tomato Early blight leaf": {
        "crop": "tomato", "disease": "early blight", "healthy": False,
        "kb": "Tomato - Blight",
    },
    "Tomato Septoria leaf spot": {
        "crop": "tomato", "disease": "Septoria leaf spot", "healthy": False, "kb": None,
    },
    "Tomato leaf": {
        "crop": "tomato", "disease": "healthy", "healthy": True,
        "kb": "Tomato - Healthy",
    },
    "Tomato leaf bacterial spot": {
        "crop": "tomato", "disease": "bacterial spot", "healthy": False, "kb": None,
    },
    "Tomato leaf late blight": {
        "crop": "tomato", "disease": "late blight", "healthy": False,
        "kb": "Tomato - Blight",
    },
    "Tomato leaf mosaic virus": {
        "crop": "tomato", "disease": "mosaic virus", "healthy": False, "kb": None,
    },
    "Tomato leaf yellow virus": {
        "crop": "tomato", "disease": "yellow virus", "healthy": False, "kb": None,
    },
    "Tomato mold leaf": {
        "crop": "tomato", "disease": "leaf mold", "healthy": False, "kb": None,
    },
    "grape leaf": {
        "crop": "grape", "disease": "healthy", "healthy": True, "kb": None,
    },
    "grape leaf black rot": {
        "crop": "grape", "disease": "black rot", "healthy": False, "kb": None,
    },
}


# The crops the PlantDoc model can identify, in the model's own naming.
PLANTDOC_CROPS: tuple[str, ...] = tuple(sorted({
    str(entry["crop"]) for entry in PLANTDOC_CLASSES.values()
}))


def is_plantdoc_label(label: str) -> bool:
    """True when the label belongs to the 27-class PlantDoc vocabulary."""
    return label in PLANTDOC_CLASSES


def plantdoc_crop(label: str) -> str | None:
    """The crop of a PlantDoc label, or None when the label is unknown."""
    entry = PLANTDOC_CLASSES.get(label)
    return str(entry["crop"]) if entry else None


def plantdoc_disease(label: str) -> str | None:
    """The farmer-facing condition of a PlantDoc label."""
    entry = PLANTDOC_CLASSES.get(label)
    return str(entry["disease"]) if entry else None


def plantdoc_is_healthy(label: str) -> bool:
    """True when a PlantDoc label means the leaf has no disease."""
    entry = PLANTDOC_CLASSES.get(label)
    return bool(entry["healthy"]) if entry else False


def plantdoc_kb_section(label: str) -> str | None:
    """The knowledge-base section for a PlantDoc label, if there is one.

    None means the knowledge base has nothing for this class, which is
    the case for 23 of the 27 classes. The service must not substitute
    guidance for another crop.
    """
    entry = PLANTDOC_CLASSES.get(label)
    return str(entry["kb"]) if entry and entry["kb"] else None


def kb_crops_with_guidance() -> tuple[str, ...]:
    """The crops that have at least one knowledge-base section."""
    return KB_CROPS
