# Crop/disease name glossary — English -> Kannada (Phase 5)
# Used by the Localizer agent to write farmer-friendly answers in Kannada.

CROP_NAMES = {
    "rice": "ಅಕ್ಕಿ",
    "wheat": "ಗೋಧು",
    "cotton": "ಹತ್ತಿ",
    "maize": None,  # NEEDS NATIVE REVIEW
    "tomato": "ಟೊಮೆಟೊ",
}

DISEASE_NAMES = {
    "healthy": "ಆರೋಗ್ಯವಂತು",
    "leaf_blight": "ಎಲೆ ಕಲುಷ",
    "leaf_rust": None,  # NEEDS NATIVE REVIEW
    "leaf_curl_virus": "ಎಲೆ ವಳಸು ವೈರಸ್",
}

ADVICE_PHRASES = {
    "kn": {
        "intro": "ನಿಮ್ಮ {crop} ಎಲೆಗೆ {disease} ಕಾಣಿಸುತ್ತಿದೆ.",
        "confirm": "ಸ್ಪ್ರೇ ಮಾಡುವ ಮೊದಲು ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "gloves_mask": "ಸ್ಪ್ರೇ ಮಾಡುವ ಮೊದಲು ಕೈಗವಸು ಮತ್ತು ಮಾಸ್ಕ್ ಧರಿಸಿ.",
        "no_rain": "ಮಳೆ ಬರುವ ಮೊದಲು ಸ್ಪ್ರೇ ಮಾಡಬೇಡಿ.",
    },
}

def localize(template_key: str, language: str = "kn", **kwargs) -> str:
    table = ADVICE_PHRASES.get(language, {})
    if not table:
        table = ADVICE_PHRASES.get("kn", {})
    return table.get(template_key, "").format(**kwargs)
