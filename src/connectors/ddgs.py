"""ddgs metasearch connector — the keyless scraped lane.

`ddgs` (deedy5/duckduckgo_search, renamed) is a metasearch LIBRARY, not a
keyed API: it fans a `text()` query out to several scraped engines (bing,
duckduckgo, google, mojeek, startpage, yandex, yahoo, wikipedia, ...) and
aggregates the answers — which is exactly the role the dead public-SearXNG
lane used to fill, minus the self-hosted instance. No credential exists to
check, so the opt-in toggle IS the gate.

Trade-off, stated plainly: a scraped lane is less durable under sustained
automated load than a keyed API (the upstream README itself recommends a
rate-limiter for heavy use) — backends can serve blocks or empty bands. That
is why the lane ships OFF by default and degrades to zero sources on any
failure, exactly like every other connector, so RRF fusion survives without
it. The keyed lanes (Tavily/LinkUp/Exa/SerpAPI) remain the durable ones;
ddgs is the lane that works out of the box while keys are being obtained.

The library is synchronous, so calls run through `asyncio.to_thread` —
the event loop must never block for the seconds a scraped metasearch takes.
"""

import asyncio
import hashlib
import logging

from .base import Connector, SearchResult, Source
from ..config import settings

logger = logging.getLogger(__name__)

# No liveness probe is defined: `text()` fans out to several scraped engines
# with no single canonical endpoint, so a probe would certify one backend and
# say nothing about the lane. The base class reports "configured (no probe
# defined)" for enabled connectors, which is the honest answer.

# Courtesy cap on a keyless scraped lane. Unlike a billed API lane there is no
# documented per-request ceiling to respect, but a large `max_results` makes
# the library page through result pages repeatedly against the same scraped
# engines — the exact sustained-automated-load pattern that earns blocks.
# 30 bounds one request's fan-out while comfortably covering default_top_k.
MAX_RESULTS = 30


class DDGSConnector(Connector):
    """ddgs metasearch connector (keyless, opt-in)."""

    name = "ddgs"

    def __init__(
        self,
        enabled: bool | None = None,
        backend: str | None = None,
        region: str | None = None,
        safesearch: str | None = None,
    ):
        # bool settings cannot use the `param or settings.x` idiom (False is
        # falsy), so the toggle goes through an explicit None check.
        self.enabled = bool(settings.ddgs_enabled) if enabled is None else bool(enabled)
        self.backend = backend or settings.ddgs_backend
        self.region = region or settings.ddgs_region
        self.safesearch = safesearch or settings.ddgs_safesearch

    @staticmethod
    def _package_available() -> bool:
        """Is the ddgs package importable?

        It is a declared dependency, but a deployment that skipped it must
        degrade to "unconfigured" (lane absent) rather than error at import —
        `find_spec` checks presence without executing the package.
        """
        try:
            import importlib.util

            return importlib.util.find_spec("ddgs") is not None
        except Exception:  # noqa: BLE001 — presence check must never raise
            return False

    def is_configured(self) -> bool:
        # Keyless lane: "configured" means the operator opted in AND the
        # package is importable. Off by default so existing deployments see
        # no behaviour change on upgrade.
        return bool(self.enabled) and self._package_available()

    async def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Execute a ddgs metasearch."""
        if not self.is_configured():
            return SearchResult(sources=[], query=query, connector_name=self.name)

        sources = []
        try:
            # Clamp at the connector boundary — same policy spot as the other
            # lanes (Brave's count, SerpApi's num) — so the cap is visible to
            # callers and `_text` stays a pure transport function.
            clamped_k = max(1, min(top_k, MAX_RESULTS))
            results = await asyncio.to_thread(self._text, query, clamped_k)

            for idx, result in enumerate(results[:top_k]):
                # Result shape drifted across library generations: current
                # ddgs reports `url`, the old duckduckgo_search reported
                # `href`. Read both; title/body are stable across both.
                url = result.get("url") or result.get("href") or ""
                source_id = f"dg_{hashlib.md5(url.encode()).hexdigest()[:8]}"

                sources.append(Source(
                    id=source_id,
                    title=result.get("title", ""),
                    url=url,
                    content=result.get("body", ""),
                    score=1.0 / (idx + 1),  # Rank-based score
                    connector=self.name,
                    metadata={},
                ))

        except Exception as e:
            logger.warning("ddgs search error: %s", e)

        return SearchResult(
            sources=sources,
            query=query,
            connector_name=self.name,
            total_results=len(sources),
        )

    def _text(self, query: str, max_results: int) -> list[dict]:
        """Synchronous ddgs call, executed via asyncio.to_thread()."""
        # Imported lazily (inside the caller's try) so an absent package
        # degrades this lane to zero sources instead of breaking import of
        # the whole connectors package — same pattern as the Tavily SDK.
        from ddgs import DDGS

        return DDGS().text(
            query=query,
            region=self.region,
            safesearch=self.safesearch,
            max_results=max_results,
            backend=self.backend,
        )
