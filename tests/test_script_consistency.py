from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest  # noqa: E402

# Expected letter ranges per language code.
EXPECTED_SCRIPT = {
    "kn": ([(0x0C80, 0x0CFF)], "Kannada"),
    "hi": ([(0x0900, 0x097F)], "Devanagari"),
    "en": ([(0x0041, 0x005A), (0x0061, 0x007A)], "Latin"),
}

PLACEHOLDER = re.compile(r"\{[^}]*\}")


def _offending_chars(text, expected_ranges):
    """Alphabetic chars in `text` outside the expected script (placeholders excluded)."""
    cleaned = PLACEHOLDER.sub("", text)
    off = []
    for ch in cleaned:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if not any(lo <= cp <= hi for lo, hi in expected_ranges):
            off.append(ch)
    return off


def _iter_glossary_values(module):
    for d in ("CROP_NAMES", "DISEASE_NAMES"):
        for k, v in getattr(module, d, {}).items():
            yield f"{d}.{k}", v
    for lang, sub in getattr(module, "ADVICE_PHRASES", {}).items():
        for k, v in sub.items():
            yield f"ADVICE_PHRASES[{lang!r}].{k}", v


def _glossary_modules():
    mods = []
    for path in sorted(Path("rag/glossary").glob("*.py")):
        if path.name == "__init__.py":
            continue
        mods.append((path.stem, importlib.import_module(f"rag.glossary.{path.stem}")))
    return mods


@pytest.mark.parametrize("lang_key,mod", _glossary_modules())
def test_glossary_values_match_script(lang_key, mod):
    expected_ranges, _name = EXPECTED_SCRIPT[lang_key]
    bad = []
    for loc, val in _iter_glossary_values(mod):
        if val is None:
            continue
        if loc.startswith("ADVICE_PHRASES"):
            m = re.match(r"ADVICE_PHRASES\[['\"]?(\w+)['\"]?\]", loc)
            if m and m.group(1) != lang_key:
                continue  # fallback language sub-dict (e.g. English) is intentional
        off = _offending_chars(val, expected_ranges)
        if off:
            bad.append((loc, val, off))
    assert not bad, bad


def _string_values():
    for path in sorted(Path("strings").glob("*.json")):
        lang = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        for k, v in data.items():
            yield path.name, lang, k, v


@pytest.mark.parametrize("fname,lang,key,val", _string_values())
def test_strings_values_match_script(fname, lang, key, val):
    if val is None:
        pytest.skip(f"{fname}:{key} is null (NEEDS NATIVE REVIEW)")
    expected_ranges, _name = EXPECTED_SCRIPT[lang]
    off = _offending_chars(val, expected_ranges)
    assert not off, (fname, key, val, off)
