"""Web search tool via DuckDuckGo Instant Answer API (zero-dependency)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..tools import ToolError, ToolRegistry


def search_tools(timeout: int = 15) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def web_search(query: str) -> str:
        """Search the web for a query. Returns a short abstract and related
        topics. Best for factual lookups, not deep browsing."""
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
        })
        req = urllib.request.Request(url, headers={"User-Agent": "picoagent/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except (OSError, ValueError, KeyError) as exc:
            raise ToolError(f"web_search failed: {exc}")

        lines: list[str] = []
        if data.get("AbstractText"):
            lines.append(data["AbstractText"])
            if data.get("AbstractURL"):
                lines.append(f"Source: {data['AbstractURL']}")
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                lines.append(f"- {topic['Text']} ({topic.get('FirstURL', '')})")
        return "\n".join(lines) or f"No instant answer for: {query}"

    return reg
