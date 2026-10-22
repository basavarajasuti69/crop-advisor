"""Tests for how a prediction reaches the knowledge base.

The bug these guard against: a PlantDoc label such as
``Potato leaf early blight`` has no knowledge-base section, and an
earlier version of the graph fell back to a hardcoded rice crop. A
potato photo then received rice guidance, which is worse than no
guidance at all.

Run with::

    python -m pytest tests/test_plantdoc_routing.py -q
"""

from __future__ import annotations

import asyncio
import io
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.graph import AgentState, retrieve  # noqa: E402
from mcp_servers.crop_knowledge import server as ck_server  # noqa: E402


def _leaf_image_bytes() -> bytes:
    rng = random.Random(0)
    from PIL import Image

    img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    pixels = img.load()
    for y in range(300):
        for x in range(300):
            jitter = rng.randint(-25, 25)
            pixels[x, y] = (
                max(0, min(255, 34 + jitter)),
                max(0, min(255, 139 + jitter)),
                max(0, min(255, 34 + jitter)),
            )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _state(label: str, score: float = 0.9) -> AgentState:
    return AgentState(
        case_id="test",
        image_bytes=_leaf_image_bytes(),
        predictions=[{"label": label, "score": score}],
    )


def _retrieve(state: AgentState) -> AgentState:
    return asyncio.run(retrieve(state))


# ---------- routing ----------

def test_a_tomato_class_with_a_section_gets_that_section():
    state = _retrieve(_state("Tomato leaf late blight"))
    assert state.crop == "tomato"
    assert state.guidance_available is True
    text = " ".join(passage["text"] for passage in state.passages)
    assert "Tomato - Blight" in text


def test_corn_leaf_blight_routes_to_the_maize_section():
    state = _retrieve(_state("Corn leaf blight"))
    assert state.crop == "corn"
    assert state.guidance_available is True
    text = " ".join(passage["text"] for passage in state.passages)
    assert "Maize - Leaf Blight" in text


def test_a_tomato_healthy_class_routes_to_the_healthy_section():
    state = _retrieve(_state("Tomato leaf"))
    assert state.crop == "tomato"
    text = " ".join(passage["text"] for passage in state.passages)
    assert "Tomato - Healthy" in text


def test_a_potato_class_never_receives_rice_guidance():
    state = _retrieve(_state("Potato leaf early blight"))
    assert state.crop == "potato"
    text = " ".join(passage["text"] for passage in state.passages)
    assert "Rice" not in text
    assert "rice" not in text.lower()
    assert state.guidance_available is False


def test_a_cherry_class_never_receives_maize_guidance():
    state = _retrieve(_state("Cherry leaf"))
    assert state.crop == "cherry"
    text = " ".join(passage["text"] for passage in state.passages)
    assert "Maize" not in text
    assert state.guidance_available is False


def test_a_class_with_no_section_reports_no_guidance():
    state = _retrieve(_state("Tomato mold leaf"))
    assert state.crop == "tomato"
    assert state.guidance_available is False
    text = " ".join(passage["text"] for passage in state.passages)
    assert "agriculture officer" in text


# ---------- the knowledge base itself ----------

@pytest.mark.asyncio
async def test_the_knowledge_base_has_no_rice_guidance_for_a_potato_query():
    result = await ck_server._search_knowledge({
        "query": "early blight potato",
        "crop": "potato",
    })
    payload = result[0].text
    assert "Rice" not in payload
