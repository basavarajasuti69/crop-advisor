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
