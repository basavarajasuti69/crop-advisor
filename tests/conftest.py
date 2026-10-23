"""Make the repo root importable for every test module.

CI runs a bare `pytest tests -q`, which (unlike `python -m pytest`) does not
put the current directory on sys.path, so `apps`, `agents`, `speech` and
`rag` would not be importable. Insert the repo root by absolute path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def onnx_available() -> bool:
    """Return True only if the ONNX model is present and loadable.

    Used by skipif guards so tests skip gracefully when the model is
    missing (CI without LFS) or corrupted, instead of failing with an
    InvalidProtobuf error.
    """
    from pathlib import Path

    model_path = Path("ml/export_edge/artifacts/classifier.onnx")
    if not model_path.exists():
        return False
    if model_path.stat().st_size < 1024:
        return False
    try:
        with model_path.open("rb") as f:
            header = f.read(32)
        if header.startswith(b"version https://git-lfs"):
            return False
    except OSError:
        return False
    try:
        import onnxruntime as ort

        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception:
        return False
    return True
