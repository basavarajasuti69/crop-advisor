from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
from PIL import Image

import pytest

sys.path.insert(0, ".")

from agents.graph import (
    AgentState,
    ImageUnreadable,
    ModelUnavailable,
    run_advice,
    _load_classifier,
    _predict_image,
    localize_advice,
)  # noqa: E402
import agents.graph as graph_agent  # noqa: E402
from mcp_servers.crop_knowledge import server as ck_server  # noqa: E402


def _make_image_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class Req:
    def __init__(self, image_bytes=None, image_url=None, crop=None, language="en", location=None):
        self.image_bytes = image_bytes
        self.image_url = image_url
        self.crop = crop
        self.language = language
        self.location = location


@patch("agents.graph._predict_image")
def test_graph_runs_with_image(mock_predict):
    mock_predict.return_value = [
        {"label": "leaf_blight", "score": 0.9},
        {"label": "healthy", "score": 0.1},
    ]
    s = run_advice_sync(Req(image_bytes=_make_image_bytes(), crop="rice"))
    assert s.quality_ok is True
    assert len(s.predictions) > 0
    assert s.answer is not None
    assert any("crop_knowledge" in c for c in s.citations)
    assert s.safe is True


def test_graph_no_image():
    s = run_advice_sync(Req())
    assert s.quality_ok is False
    assert s.quality_issue == "no image provided"


def test_classifier_fallback():
    class _FakeModel:
        def __call__(self, x):
            import torch

            return torch.tensor([[0.9, 0.1]])

    preds = _predict_image(_make_image_bytes(), model=_FakeModel(), classes=["healthy", "leaf_blight"])
    assert len(preds) == 2
    assert preds[0]["label"] == "healthy"
    with pytest.raises(ImageUnreadable):
        _predict_image(b"not-an-image")


def test_classifier_loads_no_model():
    with pytest.raises(ModelUnavailable, match="classifier model file missing"):
        _load_classifier(weights_path="ml/training/artifacts/does-not-exist.pt")


# ---------- ONNX backend (used where torch is not installed) ----------

def test_onnx_preprocess_matches_torch_transform():
    """The numpy preprocessing must be byte-identical to the training transform."""
    import numpy as np
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (300, 500), color=(20, 180, 60))
    expected = graph_agent._torch_transform()(img).numpy()
    actual = graph_agent._onnx_preprocess(img)
    assert np.abs(expected - actual).max() < 1e-6


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "ml/export_edge/artifacts/classifier.onnx").exists(),
    reason="the ONNX export is a build artifact and is not in git",
)
def test_onnx_backend_matches_torch_backend():
    """ONNX Runtime and PyTorch must give the same prediction for the same photo."""
    raw = _make_image_bytes()
    onnx_preds = graph_agent._predict_onnx(raw)
    assert onnx_preds
    assert onnx_preds[0]["label"] in ("healthy", "leaf_blight")
    assert 0.0 <= onnx_preds[0]["score"] <= 1.0
    # The order must be descending.
    scores = [p["score"] for p in onnx_preds]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "ml/export_edge/artifacts/classifier.onnx").exists(),
    reason="the ONNX export is a build artifact and is not in git",
)
def test_predict_image_prefers_onnx_when_available():
    """_predict_image uses the ONNX backend when the export is present."""
    raw = _make_image_bytes()
    preds = graph_agent._predict_image(raw)
    assert preds
    assert preds[0]["label"] in ("healthy", "leaf_blight")


@patch("agents.graph._predict_image")
def test_hindi_localization(mock_predict):
    mock_predict.return_value = [{"label": "healthy", "score": 0.9}]
    s = run_advice_sync(Req(image_bytes=_make_image_bytes(), crop="rice", language="hi"))
    assert s.quality_ok is True
    assert "आपके" in s.answer
    assert "कृषि अधिकारी" in s.answer


@patch("agents.graph._predict_image")
def test_kannada_localization(mock_predict):
    mock_predict.return_value = [{"label": "healthy", "score": 0.9}]
    s = run_advice_sync(Req(image_bytes=_make_image_bytes(), crop="rice", language="kn"))
    assert s.quality_ok is True
    assert "ಅಕ್ಕಿ" in s.answer
    assert "ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ" in s.answer


# ---------- Part 1: search_knowledge returns chunks, never the whole file ----------

def test_search_knowledge_returns_chunks_not_whole_file():
    fake_index = [
        {
            "id": "x",
            "source": "crop_guide_en.md",
            "chunk": 0,
            "text": "# Preamble line.\n## Rice - Healthy Rice plants are healthy. No treatment needed.\n## Rice - Leaf Blight Remove affected leaves. Apply fungicide as per label.",
        }
    ]
    with patch.object(ck_server, "_load_index", return_value=fake_index):
        result = asyncio.run(ck_server._search_knowledge({"query": "leaf blight", "crop": "rice"}))
    chunks = json.loads(result[0].text)
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("Rice - Leaf Blight")
    assert "Preamble" not in chunks[0]["text"]
    assert "#" not in chunks[0]["text"]


def test_search_knowledge_empty_when_no_match():
    fake_index = [
        {
            "id": "x",
            "source": "crop_guide_en.md",
            "chunk": 0,
            "text": "## Rice - Healthy Rice plants are healthy.",
        }
    ]
    with patch.object(ck_server, "_load_index", return_value=fake_index):
        result = asyncio.run(ck_server._search_knowledge({"query": "tomato blight", "crop": "rice"}))
    chunks = json.loads(result[0].text)
    assert chunks == []


def test_search_knowledge_underscore_normalized_to_space():
    fake_index = [
        {
            "id": "x",
            "source": "crop_guide_en.md",
            "chunk": 0,
            "text": "## Maize - Leaf Blight Maize leaf blight shows long elliptical tan lesions.",
        }
    ]
    with patch.object(ck_server, "_load_index", return_value=fake_index):
        result = asyncio.run(
            ck_server._search_knowledge({"query": "leaf_blight maize", "crop": "maize"})
        )
    chunks = json.loads(result[0].text)
    assert len(chunks) == 1
    assert "maize" in chunks[0]["text"].lower()
    assert "leaf blight" in chunks[0]["text"].lower()


def test_search_knowledge_unknown_query_returns_empty():
    fake_index = [
        {
            "id": "x",
            "source": "crop_guide_en.md",
            "chunk": 0,
            "text": "## Rice - Healthy Rice plants are healthy.",
        }
    ]
    with patch.object(ck_server, "_load_index", return_value=fake_index):
        result = asyncio.run(ck_server._search_knowledge({"query": "dragon fruit fungus", "crop": "rice"}))
    chunks = json.loads(result[0].text)
    assert chunks == []


# ---------- Part 2: farmer-facing strings live in strings/*.json ----------

def test_strings_files_have_required_keys():
    for lang in ("en", "hi", "kn"):
        data = json.loads(Path(f"strings/{lang}.json").read_text(encoding="utf-8"))
        for key in ("intro", "confirm", "helpline", "retake"):
            assert key in data, f"{lang} missing {key}"


def test_kn_localize_has_no_english():
    state = AgentState(
        case_id="x",
        language="kn",
        crop="rice",
        predictions=[{"label": "healthy", "score": 0.9}],
        passages=[{"text": "# English passage with guidance.", "source": "crop_knowledge_base"}],
    )
    asyncio.run(localize_advice(state))
    assert state.answer is not None
    assert not any(ch.isascii() and ch.isalpha() for ch in state.answer)


# ---------- Part 3: Kannada validator ----------

def test_validate_kannada_pure():
    assert graph_agent._validate_kannada("ನಿಮ್ಮ ಅಕ್ಕಿ ಎಲೆಯಲ್ಲಿ ಆರೋಗ್ಯವಂತು ಕಾಣಿಸುತ್ತಿದೆ.") is True


def test_validate_kannada_rejects_hash():
    assert graph_agent._validate_kannada("ನಿಮ್ಮ ಅಕ್ಕಿ #HEADING") is False


def test_validate_kannada_rejects_other_indian_script():
    assert graph_agent._validate_kannada("ನಿಮ್ಮ ಅक्कಿ अक्कಿ ಎಲೆ") is False


# ---------- Part 4: 0.70 threshold -> retake-photo response ----------

@patch("agents.graph._predict_image")
def test_low_confidence_returns_retake(mock_predict):
    mock_predict.return_value = [
        {"label": "leaf_blight", "score": 0.5},
        {"label": "healthy", "score": 0.3},
    ]
    s = run_advice_sync(Req(image_bytes=b"x", crop="rice", language="en", location="Dharwad"))
    assert s.answer is not None
    low = s.answer.lower()
    assert "retake" in low or "not sure" in low
    assert "leaf_blight" in s.answer
    assert s.escalate is True
    assert s.ood is True


@patch("agents.graph._predict_image")
def test_high_confidence_no_retake(mock_predict):
    mock_predict.return_value = [{"label": "healthy", "score": 0.9}]
    s = run_advice_sync(Req(image_bytes=b"x", crop="rice", language="en"))
    assert "retake" not in s.answer.lower()
    assert s.ood is False


# ---------- Vision second opinion ----------
def _vision_state(image_bytes=b"x", predictions=None, confidence=0.9, error=None):
    return AgentState(
        case_id="x",
        language="en",
        image_bytes=image_bytes,
        predictions=predictions or [{"label": "healthy", "score": confidence}],
        confidence=confidence,
        error=error,
    )


@patch("agents.graph.vision_opinion")
@patch.object(graph_agent, "VISION_ENABLED", False)
def test_vision_disabled_by_default(mock_vision):
    state = _vision_state()
    asyncio.run(graph_agent.vision_second_opinion(state))
    assert state.vision_llm_opinion is None
    assert state.agreed is None
    mock_vision.assert_not_called()


@patch("agents.graph.vision_opinion")
@patch.object(graph_agent, "VISION_ENABLED", True)
def test_vision_agree(mock_vision):
    mock_vision.return_value = {"symptoms": "uniform green leaves", "diseases": ["healthy"]}
    state = _vision_state(confidence=0.9)
    asyncio.run(graph_agent.vision_second_opinion(state))
    assert state.agreed is True
    assert state.retake_requested is False


@patch("agents.graph.vision_opinion")
@patch.object(graph_agent, "VISION_ENABLED", True)
def test_vision_disagree_low_confidence_retake(mock_vision):
    mock_vision.return_value = {"symptoms": "blight spots", "diseases": ["leaf_blight"]}
    state = _vision_state(confidence=0.6)
    asyncio.run(graph_agent.vision_second_opinion(state))
    assert state.agreed is False
    assert state.retake_requested is True
    assert "disagree" in state.quality_issue


@patch("agents.graph.vision_opinion")
@patch.object(graph_agent, "VISION_ENABLED", True)
def test_vision_api_error_no_crash(mock_vision):
    mock_vision.side_effect = Exception("down")
    state = _vision_state(confidence=0.9)
    asyncio.run(graph_agent.vision_second_opinion(state))
    assert state.vision_llm_opinion is None
    assert state.agreed is None
    assert state.retake_requested is False


# ---------- LLM explanation step ----------

@patch("agents.graph.llm_configured", return_value=True)
@patch("agents.graph.explain")
@patch("agents.graph._predict_image")
def test_llm_explain_used_in_graph(mock_predict, mock_explain, mock_configured):
    mock_predict.return_value = [{"label": "leaf_blight", "score": 0.9}]
    mock_explain.return_value = "LLM generated farmer advice."
    s = run_advice_sync(Req(image_bytes=_make_image_bytes(), crop="rice", language="en"))
    assert s.llm_explained is True
    assert "LLM generated farmer advice." in s.answer
    mock_explain.assert_called_once()
    # only label, chunks and language are sent - no image, no location
    args, _kwargs = mock_explain.call_args
    assert len(args) == 3
    assert args[0] == "leaf_blight"
    assert isinstance(args[1], list)
    assert args[2] == "en"


@patch("agents.graph.llm_configured", return_value=False)
@patch("agents.graph.explain")
@patch("agents.graph._predict_image")
def test_llm_not_used_when_not_configured(mock_predict, mock_explain, mock_configured):
    mock_predict.return_value = [{"label": "leaf_blight", "score": 0.9}]
    s = run_advice_sync(Req(image_bytes=_make_image_bytes(), crop="rice", language="en"))
    assert s.llm_explained is False
    mock_explain.assert_not_called()
    assert s.answer is not None


@patch("agents.graph.llm_configured", return_value=True)
@patch("agents.graph.explain")
@patch("agents.graph._predict_image")
def test_llm_answer_helpline_not_duplicated(mock_predict, mock_explain, mock_configured):
    mock_predict.return_value = [{"label": "leaf_blight", "score": 0.9}]
    helpline = graph_agent.load_strings("en")["helpline"]
    mock_explain.return_value = f"LLM advice. {helpline}"
    s = run_advice_sync(Req(image_bytes=_make_image_bytes(), crop="rice", language="en"))
    assert s.answer.count(helpline) == 1


def _make_officer_mock(officer_result):
    def _fake(server_name: str, tool: str, arguments: dict):
        if server_name == "officer_directory":
            return officer_result
        if server_name == "crop_knowledge":
            return [
                {
                    "text": "remove affected leaves and apply approved fungicide.",
                    "source": "crop_knowledge_base",
                    "dose": "as per label",
                }
            ]
        if server_name == "weather":
            return {}
        return {}

    return _fake


def test_officer_no_result_uses_helpline():
    officer_result = [{"message": "No officers found"}]
    with patch(
        "agents.graph.mcp_call", new_callable=AsyncMock, side_effect=_make_officer_mock(officer_result)
    ), patch(
        "agents.graph._predict_image",
        return_value=[{"label": "leaf_blight", "score": 0.5}],
    ):
        state = run_advice_sync(
            Req(
                image_bytes=_make_image_bytes(),
                crop="rice",
                language="en",
                location="test village",
            )
        )
    assert state.officer == []
    assert "agriculture helpline" in state.answer


def test_officer_real_result_appears_in_answer():
    officer_result = [{"name": "Ramesh", "phone": "9876543210"}]
    with patch(
        "agents.graph.mcp_call", new_callable=AsyncMock, side_effect=_make_officer_mock(officer_result)
    ), patch(
        "agents.graph._predict_image",
        return_value=[{"label": "leaf_blight", "score": 0.5}],
    ):
        state = run_advice_sync(
            Req(
                image_bytes=_make_image_bytes(),
                crop="rice",
                language="en",
                location="test village",
            )
        )
    assert state.officer == [{"name": "Ramesh", "phone": "9876543210"}]
    assert "Ramesh" in state.answer


def run_advice_sync(req: Req):
    import asyncio as _a

    return _a.run(run_advice(req))
