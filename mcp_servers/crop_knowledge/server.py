"""crop_knowledge MCP server — exposes the RAG knowledge base as MCP tools."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("crop_knowledge-mcp")

INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "rag" / "index.jsonl"


def _load_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    docs = []
    with open(INDEX_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge",
            description="Search the crop disease knowledge base for guidance",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (crop, disease, symptom)"},
                    "crop": {"type": "string", "description": "Filter by crop name"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_crop_diseases",
            description="List known diseases for a crop",
            inputSchema={
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "Crop name (rice, wheat, cotton, maize, tomato)"},
                },
                "required": ["crop"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "search_knowledge":
        return await _search_knowledge(arguments)
    if name == "get_crop_diseases":
        return await _get_crop_diseases(arguments)
    raise ValueError(f"Unknown tool: {name}")


def _query_tokens(query: str) -> list[str]:
    """Split a query into the tokens that must be found in a chunk.

    The knowledge base is headed "<Crop> - <Disease>", but the graph asks
    for "<disease> <crop>" (for example "leaf blight rice"). A plain
    substring test never matches, so every token has to be found
    separately.
    """
    tokens = [token for token in re.split(r"[\s_]+", query.lower()) if token]
    return list(dict.fromkeys(tokens))


def _matches_token(text: str, token: str) -> bool:
    """Whole-word match, so "leaf" does not match "leaves"."""
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


async def _search_knowledge(args: dict[str, Any]) -> list[TextContent]:
    query = args["query"].lower().replace("_", " ")
    crop = args.get("crop", "").lower()
    tokens = _query_tokens(query)
    # The crop is usually part of the query already; the rest identify the
    # disease.
    disease_tokens = [token for token in tokens if token != crop] or tokens
    docs = _load_index()
    scored: list[tuple[bool, int, int, dict[str, Any]]] = []
    for doc in docs:
        text = doc.get("text", "")
        for part in text.split("## "):
            part = part.strip()
            if not part:
                continue
            low = part.lower().replace("_", " ")
            if crop and not _matches_token(low, crop):
                continue
            positions = [
                match.start()
                for token in disease_tokens
                for match in [re.search(rf"\b{re.escape(token)}\b", low)]
                if match is not None
            ]
            if not positions:
                continue
            hits = len(positions)
            entry = {
                "text": part,
                "source": doc.get("source", ""),
                "chunk": doc.get("chunk", 0),
            }
            # Chunks that match every token win; then the most matches; then
            # the one where the disease word appears earliest, because the
            # heading comes first ("Wheat - Rust ..." beats a description
            # that merely mentions rust pustules).
            scored.append((hits == len(disease_tokens), hits, -min(positions), entry))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    chunks = [entry for _all, _hits, _pos, entry in scored[:5]]
    return [TextContent(type="text", text=json.dumps(chunks, indent=2))]


async def _get_crop_diseases(args: dict[str, Any]) -> list[TextContent]:
    crop = args["crop"].lower()
    docs = _load_index()
    diseases = set()
    for doc in docs:
        source = doc.get("source", "").lower()
        if crop in source:
            text = doc.get("text", "")
            for line in text.split("\n"):
                if "-" in line and any(w in line.lower() for w in ["blight", "rust", "curl", "healthy"]):
                    diseases.add(line.split("-")[0].strip())
    result = {"crop": crop, "diseases": sorted(diseases) or ["No diseases cataloged"]}
    return [TextContent(type="text", text=json.dumps(result))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
