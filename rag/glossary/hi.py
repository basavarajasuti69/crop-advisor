# Crop/disease name glossary — English -> Hindi (Phase 2)
# Used by the Localizer agent to write farmer-friendly answers in Hindi.

CROP_NAMES = {
    "rice": "चावल",
    "wheat": "गेहूं",
    "cotton": "कपास",
    "maize": "मक्का",
    "tomato": "टमाटर",
}

DISEASE_NAMES = {
    "healthy": "स्वस्थ",
    "leaf_blight": "पत्ती झुलसा",
    "leaf_rust": "पत्ती जंग",
    "leaf_curl_virus": None,  # NEEDS NATIVE REVIEW
}

ADVICE_PHRASES = {
    "en": {
        "intro": "Your {crop} leaf shows {disease}.",
        "confirm": "Confirm with your local agriculture officer before spraying.",
    },
    "hi": {
        "intro": "आपके {crop} पत्ते पर {disease} दिखाई दे रहा है।",
        "confirm": "स्प्रे करने से पहले अपने स्थानीय कृषि अधिकारी से सलाह लें।",
    },
}

def localize(template_key: str, language: str = "en", **kwargs) -> str:
    table = ADVICE_PHRASES.get(language, ADVICE_PHRASES["en"])
    return table[template_key].format(**kwargs)
