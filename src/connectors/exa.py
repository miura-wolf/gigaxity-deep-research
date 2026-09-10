"""Exa connector — neural web-search lane.

Exa runs its own neural index tuned for LLM consumers (high-precision
retrieval with page text) behind a keyed API. Like Brave, an API lane cannot
be served a CAPTCHA or bot-block page the way a self-hosted SearXNG's scraped
engines can, so it stays available under sustained automated load. Its free
tier ($20 signup credits + $10/month, no payment method) makes it the
recommended general-web lane while self-hosted SearXNG is unavailable.

There is no official Python SDK for /search, so this speaks HTTP directly
(same pattern as the Brave connector).
"""

import hashlib
import logging
import httpx
from .base import Connector, SearchResult, Source
from ..config import settings

logger = logging.getLogger(__name__)

API_URL = "https://api.exa.ai/search"

# Exa bills per-result above the 10 included in the base price; clamping here
# keeps a top_k request from silently multiplying cost.
MAX_RESULTS = 10

# Floor for the page-text cap — a zero/negative value would request no text
# and degrade the source to a bare title/URL.
MIN_TEXT_CHARS = 200


class ExaConnector(Connector):
    """Exa neural search connector."""

    name = "exa"

    def __init__(
        self,
        api_key: str | None = None,
        search_type: str | None = None,
        text_max_chars: int | None = None,
    ):
        self.api_key = api_key or settings.exa_api_key
        self.search_type = search_type or settings.exa_type
        if text_max_chars is not None:
            self.text_max_chars = text_max_chars
        else:
            self.text_max_chars = settings.exa_text_max_chars

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _probe_url(self) -> str | None:
        # Reachability only; key validity would cost a billed call.
        return "https://api.exa.ai"

    async def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Execute an Exa search."""
        if not self.is_configured():
            return SearchResult(sources=[], query=query, connector_name=self.name)

        payload = {
            "query": query,
            "numResults": max(1, min(top_k, MAX_RESULTS)),
            "type": self.search_type,
            # Page text is what the synthesis stages consume; without it Exa
            # returns only title/URL and this connector contributes weak sources.
            "contents": {
                "text": {"maxCharacters": max(MIN_TEXT_CHARS, int(self.text_max_chars))}
            },
        }

        headers = {
            "Accept": "application/json",
            "x-api-key": self.api_key,
        }

        sources = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])[:top_k]

            for idx, result in enumerate(results):
                url = result.get("url", "")
                source_id = f"ex_{hashlib.md5(url.encode()).hexdigest()[:8]}"

                sources.append(Source(
                    id=source_id,
                    title=result.get("title", ""),
                    url=url,
                    content=result.get("text", ""),
                    score=1.0 / (idx + 1),  # Rank-based score
                    connector=self.name,
                    metadata={
                        "published_date": result.get("publishedDate"),
                        "author": result.get("author"),
                    },
                ))

        except Exception as e:
            logger.warning("Exa search error: %s", e)

        return SearchResult(
            sources=sources,
            query=query,
            connector_name=self.name,
            total_results=len(sources),
        )