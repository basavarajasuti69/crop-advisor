"""RAG ingest stub: load crop knowledge docs into the knowledge base index.

Phase 1: reads markdown files from rag/source_docs and writes chunks
to a local JSONL index.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def chunk_text(text: str, size: int = 500) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [""]


def main() -> None:
    src = Path("rag/source_docs")
    out = Path("rag/index.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as fh:
        for path in sorted(src.glob("*.md")):
            text = path.read_text()
            for i, chunk in enumerate(chunk_text(text)):
                doc = {
                    "id": hashlib.sha1(f"{path.name}:{i}".encode()).hexdigest()[:16],
                    "source": path.name,
                    "chunk": i,
                    "text": chunk,
                }
                fh.write(json.dumps(doc) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
