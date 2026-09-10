"""SerpApi connector — classic SERP lane (Google/Bing/DuckDuckGo).

SerpApi scrapes and normalizes traditional search-engine result pages behind
a keyed REST API, so like the other API lanes it cannot be served a CAPTCHA
or bot-block page under automated load. The free plan (~100 searches/month)
plus the broad engine coverage (Google, Bing, DuckDuckGo, ...) make it a
complementary lane to Exa's neural index: SerpApi answers keyword-shaped
queries with familiar SERP ranking, Exa answers semantic ones.

Unlike Brave (subscription header) or Exa (x-api-key header), SerpApi's
documented transport carries the key as a `api_key` query parameter — over
HTTPS only, and never written to logs.
"""

import hashlib
import logging
import httpx
from .base import Connector, SearchResult, Source
from ..config import settings

logger = logging.getLogger(__name__)

API_URL = "https://serpapi.com/search"

# Google's `num` parameter is rejected above 100; clamping keeps the
# connector inside the documented band for every engine SerpApi offers.
MAX_NUM = 100


class SerpApiConnector(Connector):
    """SerpApi SERP search connector."""

    name = "serpapi"

    def __init__(
        self,
        api_key: str | None = None,
        engine: str | None = None,
    ):
        self.api_key = api_key or settings.serpapi_api_key
        self.engine = engine or settings.serpapi_engine

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _probe_url(self) -> str | None:
        # Reachability only; key validity would cost a billed call.
        return "https://serpapi.com"

    async def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Execute a SerpApi search."""
        if not self.is_configured():
            return SearchResult(sources=[], query=query, connector_name=self.name)

        params: dict[str, str] = {
            "engine": self.engine,
            "q": query,
            "num": str(max(1, min(top_k, MAX_NUM))),
            # Documented transport: api_key is a query parameter for SerpApi.
            "api_key": self.api_key,
        }

        headers = {"Accept": "application/json"}

        sources = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(API_URL, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            # A query SerpApi answers with only an error payload (or with no
            # organic band, e.g. a pure knowledge-panel answer) carries no
            # `organic_results` key at all — zero results, not an error, so it
            # fuses as a quiet non-contributor.
            results = data.get("organic_results", [])[:top_k]

            for idx, result in enumerate(results):
                url = result.get("link", "")
                source_id = f"sp_{hashlib.md5(url.encode()).hexdigest()[:8]}"

                sources.append(Source(
                    id=source_id,
                    title=result.get("title", ""),
                    url=url,
                    content=result.get("snippet", ""),
                    score=1.0 / (idx + 1),  # Rank-based score
                    connector=self.name,
                    metadata={
                        "position": result.get("position"),
                        "published_date": result.get("date"),
                    },
                ))

        except Exception as e:
            logger.warning("SerpApi search error: %s", e)

        return SearchResult(
            sources=sources,
            query=query,
            connector_name=self.name,
            total_results=len(sources),
        )