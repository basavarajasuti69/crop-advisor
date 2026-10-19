"""Weather MCP server — provides weather and season context for crop advisory.

Exposes tools:
  - get_weather(location, date) -> current/forecast weather
  - get_season(location, month) -> crop season guidance
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("weather-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="Get current weather and forecast for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or village name"},
                    "date": {
                        "type": "string",
                        "description": "Target date (YYYY-MM-DD), defaults to today",
                    },
                },
                "required": ["location"],
            },
        ),
        Tool(
            name="get_season",
            description="Get crop season guidance for a location and month",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or village name"},
                    "month": {
                        "type": "integer",
                        "description": "Month number 1-12",
                    },
                },
                "required": ["location", "month"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "get_weather":
        return await _get_weather(arguments)
    if name == "get_season":
        return await _get_season(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def _get_weather(args: dict[str, Any]) -> list[TextContent]:
    location = args["location"]
    date = args.get("date", "today")
    async with httpx.AsyncClient(timeout=15) as client:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
        )
        geo.raise_for_status()
        results = geo.json().get("results", [])
        if not results:
            return [TextContent(type="text", text=json.dumps({"error": f"Location '{location}' not found"}))]
        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        forecast = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        data = forecast.json()
        result = {
            "location": f"{place['name']}, {place.get('country_code', '')}",
            "current": data.get("current", {}),
            "daily": data.get("daily", {}),
            "source": "open-meteo",
        }
        return [TextContent(type="text", text=json.dumps(result))]


SEASON_GUIDANCE = {
    "rice": {6: "Kharif sowing", 7: "Transplanting", 8: "Vegetative stage", 9: "Flowering", 10: "Harvest"},
    "wheat": {10: "Sowing", 11: "Early growth", 12: "Tillering", 1: "Jointing", 2: "Heading", 3: "Harvest"},
    "cotton": {5: "Sowing", 6: "Seedling", 7: "Squaring", 8: "Flowering", 9: "Boll development", 10: "Harvest"},
    "maize": {6: "Sowing", 7: "Vegetative", 8: "Tasseling", 9: "Grain fill", 10: "Harvest"},
    "tomato": {1: "Transplanting", 2: "Vegetative", 3: "Flowering", 4: "Fruiting", 5: "Harvest"},
}


async def _get_season(args: dict[str, Any]) -> list[TextContent]:
    location = args["location"]
    month = args["month"]
    guidance = []
    for crop, months in SEASON_GUIDANCE.items():
        if month in months:
            guidance.append(f"{crop}: {months[month]}")
    result = {
        "location": location,
        "month": month,
        "guidance": guidance or ["No specific season guidance for this month"],
        "source": "crop_calendar",
    }
    return [TextContent(type="text", text=json.dumps(result))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
