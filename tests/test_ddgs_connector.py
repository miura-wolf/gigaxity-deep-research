"""Tests for the ddgs metasearch connector.

ddgs is the keyless scraped lane — a library metasearch (deedy5/ddgs) across
scraped engines, the role the dead public-SearXNG lane used to fill minus the
instance. Keyless means the opt-in toggle IS the configuration gate, so the
Basics section pins: off-by-default must stay off unless the operator opts in
AND the package is importable. Every failure must degrade to zero sources so
RRF fusion survives, exactly like the keyed lanes.
"""

from unittest.mock import patch

import pytest

from src.connectors import DDGSConnector
from src.connectors.ddgs import MAX_RESULTS


DDGS_LIVE_SHAPE = [
    # Current ddgs (9.x) result shape: title/url/body.
    {
        "title": "Retrieval-Augmented Generation survey",
        "url": "https://arxiv.org/abs/2405.07437",
        "body": "A survey of RAG evaluation approaches.",
    },
    {
        "title": "Seven RAG benchmarks",
        "url": "https://www.evidentlyai.com/blog/rag-benchmarks",
        "body": "We highlight seven RAG benchmarks and trade-offs.",
    },
]

# Old duckduckgo_search result shape: title/href/body. Read defensively so a
# library generation bump does not silently zero the lane.
DDGS_LEGACY_SHAPE = [
    {
        "title": "Legacy result",
        "href": "https://legacy.example.com/page",
        "body": "Old-generation result shape.",
    },
]


class TestDDGSConnectorBasics:

    @pytest.mark.unit
    def test_connector_name(self):
        assert DDGSConnector(enabled=True).name == "ddgs"

    @pytest.mark.unit
    def test_is_configured_when_enabled(self):
        assert DDGSConnector(enabled=True).is_configured() is True

    @pytest.mark.unit
    def test_is_not_configured_when_disabled(self):
        """Keyless lane: the toggle is the gate, so off must mean absent."""
        assert DDGSConnector(enabled=False).is_configured() is False

    @pytest.mark.unit
    def test_is_not_configured_by_default(self):
        """Off by default: upgrading must not silently enable a scraped lane."""
        assert DDGSConnector().is_configured() is False

    @pytest.mark.unit
    def test_settings_enable_the_lane(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.ddgs_enabled", True)
        # The lane reads the toggle at construction time, not call time.
        assert DDGSConnector().is_configured() is True

    @pytest.mark.unit
    def test_is_not_configured_without_package(self, monkeypatch):
        """A deployment that skipped `pip install ddgs` sees 'unconfigured',
        not an ImportError at request time."""
        monkeypatch.setattr("src.config.settings.ddgs_enabled", True)
        with patch.object(DDGSConnector, "_package_available", return_value=False):
            assert DDGSConnector().is_configured() is False


class TestDDGSConnectorSearch:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_disabled_search_returns_empty(self):
        result = await DDGSConnector(enabled=False).search("query")
        assert result.sources == []
        assert result.connector_name == "ddgs"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_maps_live_shape(self):
        with patch.object(DDGSConnector, "_text", return_value=DDGS_LIVE_SHAPE):
            result = await DDGSConnector(enabled=True).search("RAG survey", top_k=10)

        assert len(result.sources) == 2
        first = result.sources[0]
        assert first.title == "Retrieval-Augmented Generation survey"
        assert first.url == "https://arxiv.org/abs/2405.07437"
        assert first.content == "A survey of RAG evaluation approaches."
        assert first.connector == "ddgs"
        assert first.id.startswith("dg_")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_maps_legacy_href_shape(self):
        """The old duckduckgo_search shape (href, not url) must still map."""
        with patch.object(DDGSConnector, "_text", return_value=DDGS_LEGACY_SHAPE):
            result = await DDGSConnector(enabled=True).search("q")

        assert result.sources[0].url == "https://legacy.example.com/page"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rank_based_scores(self):
        with patch.object(DDGSConnector, "_text", return_value=DDGS_LIVE_SHAPE):
            result = await DDGSConnector(enabled=True).search("q")

        assert result.sources[0].score == pytest.approx(1.0)
        assert result.sources[1].score == pytest.approx(0.5)
        assert result.total_results == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_results_are_truncated_to_top_k(self):
        many = [
            {"title": f"r{i}", "url": f"https://example.com/{i}", "body": "b"}
            for i in range(50)
        ]
        with patch.object(DDGSConnector, "_text", return_value=many):
            result = await DDGSConnector(enabled=True).search("q", top_k=5)

        assert len(result.sources) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_band_is_zero_not_error(self):
        """A scraped backend serving an empty answer fuses as a quiet
        non-contributor — zero sources, never an exception."""
        with patch.object(DDGSConnector, "_text", return_value=[]):
            result = await DDGSConnector(enabled=True).search("q")

        assert result.sources == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failure_is_absorbed_not_raised(self):
        """A failing scraped lane must degrade to zero sources so RRF fusion
        survives — same contract as every keyed lane."""
        with patch.object(
            DDGSConnector, "_text", side_effect=Exception("backend served a block")
        ):
            result = await DDGSConnector(enabled=True).search("q")

        assert result.sources == []
        assert result.connector_name == "ddgs"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_package_degrades_at_search_time_too(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.ddgs_enabled", True)
        with patch.object(DDGSConnector, "_package_available", return_value=False):
            result = await DDGSConnector().search("q")

        assert result.sources == []


class TestDDGSConnectorRequestContract:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_results_is_clamped(self):
        """A keyless scraped lane gets a courtesy cap: a large top_k must not
        make the library page repeatedly against the same engines — the exact
        sustained-automated-load pattern that earns blocks."""
        seen = {}

        def fake_text(query, top_k):
            seen["max_results"] = top_k
            return []

        with patch.object(DDGSConnector, "_text", side_effect=fake_text):
            await DDGSConnector(enabled=True).search("q", top_k=500)

        assert seen["max_results"] == MAX_RESULTS

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_settings_reach_the_ddgs_call(self, monkeypatch):
        """region/backend/safesearch must flow from settings to DDGS().text()."""
        monkeypatch.setattr("src.config.settings.ddgs_enabled", True)
        monkeypatch.setattr("src.config.settings.ddgs_backend", "duckduckgo,bing")
        monkeypatch.setattr("src.config.settings.ddgs_region", "es-es")
        monkeypatch.setattr("src.config.settings.ddgs_safesearch", "off")

        calls = {}

        def fake_text(self, query, top_k):
            calls["region"] = self.region
            calls["backend"] = self.backend
            calls["safesearch"] = self.safesearch
            return []

        with patch.object(DDGSConnector, "_text", fake_text):
            await DDGSConnector().search("q")

        assert calls["region"] == "es-es"
        assert calls["backend"] == "duckduckgo,bing"
        assert calls["safesearch"] == "off"


class TestDDGSConnectorRegistration:

    @pytest.mark.unit
    def test_exported_from_connectors_package(self):
        import src.connectors as pkg

        assert "DDGSConnector" in pkg.__all__
        assert pkg.DDGSConnector is DDGSConnector

    @pytest.mark.unit
    def test_aggregator_default_roster_includes_ddgs(self, monkeypatch):
        """The connector joins the default roster (keyless check via the
        settings toggle): when enabled, it must be selectable by name."""
        from src.search.aggregator import SearchAggregator

        monkeypatch.setattr("src.config.settings.ddgs_enabled", True)
        agg = SearchAggregator(connectors=[DDGSConnector()])
        assert agg.get_active_connectors() == ["ddgs"]

    @pytest.mark.unit
    def test_doctor_roster_includes_ddgs(self):
        from src.connectors.doctor import known_connectors

        names = [c.name for c in known_connectors()]
        assert "ddgs" in names

    @pytest.mark.unit
    def test_doctor_reports_unconfigured_when_disabled(self, monkeypatch):
        """Disabled (the default) must surface as a distinct 'unconfigured'
        row in the doctor report — the operator opted out, nothing is broken."""
        import asyncio

        from src.connectors.doctor import check_connectors

        monkeypatch.setattr("src.config.settings.ddgs_enabled", False)

        async def _run():
            all_health = await check_connectors(timeout_s=1)
            return [h for h in all_health if h.name == "ddgs"]

        rows = asyncio.run(_run())
        assert len(rows) == 1
        assert rows[0].configured is False
        assert rows[0].status == "unconfigured"
