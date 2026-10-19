"""officer_directory MCP server — provides agriculture officer contacts by location."""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("officer_directory-mcp")

# Sample officer directory — in production this comes from a database/API
OFFICERS = [
    {
        "id": "off_001",
        "name": "Ramesh Kumar",
        "title": "Agriculture Officer",
        "district": "Karimnagar",
        "state": "Telangana",
        "phone": "+91-87112-XXXXX",
        "email": "ramesh.kumar@agri.telangana.gov.in",
        "crops": ["rice", "cotton", "maize"],
        "languages": ["en", "hi", "te"],
    },
    {
        "id": "off_002",
        "name": "Sita Reddy",
        "title": "Senior Agronomist",
        "district": "Nizamabad",
        "state": "Telangana",
        "phone": "+91-8462-XXXXX",
        "email": "sita.reddy@agri.telangana.gov.in",
        "crops": ["rice", "turmeric", "maize"],
        "languages": ["en", "hi", "te"],
    },
    {
        "id": "off_003",
        "name": "Vikram Singh",
        "title": "Crop Protection Officer",
        "district": "Wardha",
        "state": "Maharashtra",
        "phone": "+91-7152-XXXXX",
        "email": "vikram.singh@agri.mh.gov.in",
        "crops": ["cotton", "wheat", "soybean"],
        "languages": ["en", "hi", "mr"],
    },
    {
        "id": "off_004",
        "name": "Lakshmi Devi",
        "title": "Horticulture Officer",
        "district": "Chittoor",
        "state": "Andhra Pradesh",
        "phone": "+91-8572-XXXXX",
        "email": "lakshmi.devi@agri.ap.gov.in",
        "crops": ["tomato", "mango", "groundnut"],
        "languages": ["en", "hi", "te"],
    },
    {
        "id": "off_005",
        "name": "Arjun Patel",
        "title": "Agriculture Officer",
        "district": "Kheda",
        "state": "Gujarat",
        "phone": "+91-268-XXXXX",
        "email": "arjun.patel@agri.gujarat.gov.in",
        "crops": ["cotton", "wheat", "maize"],
        "languages": ["en", "hi", "gu"],
    },
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="find_officer",
            description="Find agriculture officers near a location for a crop",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "District or state name"},
                    "crop": {"type": "string", "description": "Crop name (optional filter)"},
                    "language": {"type": "string", "description": "Preferred language code (en, hi, te, mr, gu)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="get_officer",
            description="Get officer details by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "officer_id": {"type": "string", "description": "Officer ID"},
                },
                "required": ["officer_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "find_officer":
        return await _find_officer(arguments)
    if name == "get_officer":
        return await _get_officer(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def _find_officer(args: dict[str, Any]) -> list[TextContent]:
    location = args.get("location", "").lower()
    crop = args.get("crop", "").lower()
    language = args.get("language", "en").lower()

    results = []
    for off in OFFICERS:
        match = True
        if location and location not in (off["district"] + off["state"]).lower():
            match = False
        if crop and crop not in off["crops"]:
            match = False
        if language and language not in off["languages"]:
            match = False
        if match:
            results.append(off)

    if not results:
        results = [{"message": "No officers found matching criteria. Contact state agriculture helpline."}]

    return [TextContent(type="text", text=json.dumps(results, indent=2))]


async def _get_officer(args: dict[str, Any]) -> list[TextContent]:
    officer_id = args["officer_id"]
    for off in OFFICERS:
        if off["id"] == officer_id:
            return [TextContent(type="text", text=json.dumps(off, indent=2))]
    return [TextContent(type="text", text=json.dumps({"error": f"Officer {officer_id} not found"}))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
