"""Tests for the Exa and SerpApi connectors.

Exa is the neural web-search lane (own index, keyed API — CAPTCHA-immune
like every API lane), SerpApi the classic SERP lane (Google/Bing/DDG
normalized behind a keyed API). Both join the aggregator's RRF fusion and
must degrade to zero sources on any failure so fusion survives.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors import ExaConnector, SerpApiConnector
from src.connectors.exa import API_URL as EXA_API_URL, MAX_RESULTS as EXA_MAX_RESULTS
from src.connectors.serpapi import API_URL as SERPAPI_API_URL, MAX_NUM as SERPAPI_MAX_NUM


def _response(payload):
    """Build a mock httpx response returning `payload` from .json()."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _client_returning(resp):
    """Patchable async context manager whose .post()/.get() returns `resp`."""
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.get = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, client


# Shapes mirror real api.exa.ai/search and serpapi.com/search responses.
EXA_LIVE_SHAPE = {
    "requestId": "b5947044c4b78efa9552a7c89b306d95",
    "results": [
        {
            "title": "Retrieval-Augmented Generation survey",
            "url": "https://arxiv.org/abs/2405.07437",
            "text": "A survey of RAG evaluation approaches.",
            "publishedDate": "2024-05-13T00:00:00.000Z",
            "author": "arxiv.org",
        },
        {
            "title": "Seven RAG benchmarks",
            "url": "https://www.evidentlyai.com/blog/rag-benchmarks",
            "text": "We highlight seven RAG benchmarks and trade-offs.",
            "publishedDate": "2025-05-06T00:00:00.000Z",
            "author": "",
        },
    ],
    "costDollars": {"total": 0.007},
}

SERPAPI_LIVE_SHAPE = {
    "search_metadata": {"id": "test", "status": "Success"},
    "organic_results": [
        {
            "position": 1,
            "title": "Retrieval-Augmented Generation survey",
            "link": "https://arxiv.org/abs/2405.07437",
            "snippet": "A survey of RAG evaluation approaches.",
            "date": "May 13, 2024",
        },
        {
            "position": 2,
            "title": "Seven RAG benchmarks",
            "link": "https://www.evidentlyai.com/blog/rag-benchmarks",
            "snippet": "We highlight seven RAG benchmarks.",
            "date": "",
        },
    ],
}


# ---------------------------------------------------------------------------
# Exa
# ---------------------------------------------------------------------------


class TestExaConnectorBasics:

    @pytest.mark.unit
    def test_connector_name(self):
        assert ExaConnector(api_key="k").name == "exa"

    @pytest.mark.unit
    def test_is_configured_with_key(self):
        assert ExaConnector(api_key="k").is_configured() is True

    @pytest.mark.unit
    def test_is_not_configured_without_key(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.exa_api_key", "")
        assert ExaConnector(api_key="").is_configured() is False

    @pytest.mark.unit
    def test_probe_url_does_not_spend_a_query(self):
        """Health probe must hit the API root, never the billed search path."""
        probe = ExaConnector(api_key="k")._probe_url()
        assert probe == "https://api.exa.ai"
        assert "/search" not in probe

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unconfigured_returns_empty_without_network(self):
        """An unset key must short-circuit before any HTTP call."""
        with patch("src.connectors.exa.httpx.AsyncClient") as client_cls:
            result = await ExaConnector(api_key="").search("anything")
        client_cls.assert_not_called()
        assert result.sources == []
        assert result.connector_name == "exa"


class TestExaConnectorParsing:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_maps_live_response_shape(self):
        ctx, _ = _client_returning(_response(EXA_LIVE_SHAPE))
        with patch("src.connectors.exa.httpx.AsyncClient", return_value=ctx):
            result = await ExaConnector(api_key="k").search("q")

        assert len(result.sources) == 2
        first = result.sources[0]
        assert first.id.startswith("ex_")
        assert first.title == "Retrieval-Augmented Generation survey"
        assert first.url == "https://arxiv.org/abs/2405.07437"
        assert first.content == "A survey of RAG evaluation approaches."
        assert first.connector == "exa"
        # Rank-based scores: 1.0, 0.5
        assert first.score == pytest.approx(1.0)
        assert result.sources[1].score == pytest.approx(0.5)
        assert result.total_results == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_results_band_is_zero_not_error(self):
        """No `results` key (e.g. error payload) fuses as a quiet non-contributor."""
        ctx, _ = _client_returning(_response({}))
        with patch("src.connectors.exa.httpx.AsyncClient", return_value=ctx):
            result = await ExaConnector(api_key="k").search("q")

        assert result.sources == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_text_max_chars_floor_is_enforced(self):
        """A configured 0 must not request empty page text (degrades sources)."""
        ctx, _ = _client_returning(_response(EXA_LIVE_SHAPE))
        with patch("src.connectors.exa.httpx.AsyncClient", return_value=ctx):
            await ExaConnector(api_key="k", text_max_chars=0).search("q")

        payload = ctx.__aenter__.return_value.post.call_args.kwargs["json"]
        assert payload["contents"]["text"]["maxCharacters"] >= 200


class TestExaConnectorRequestContract:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_numresults_is_clamped_to_billed_base(self):
        """Exa bills per-result above the 10 included; clamp, never forward."""
        ctx, client = _client_returning(_response(EXA_LIVE_SHAPE))
        with patch("src.connectors.exa.httpx.AsyncClient", return_value=ctx):
            await ExaConnector(api_key="k").search("q", top_k=500)

        payload = client.post.call_args.kwargs["json"]
        assert payload["numResults"] == EXA_MAX_RESULTS

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_key_is_sent_as_api_key_header(self):
        ctx, client = _client_returning(_response(EXA_LIVE_SHAPE))
        with patch("src.connectors.exa.httpx.AsyncClient", return_value=ctx):
            await ExaConnector(api_key="secret-key").search("q")

        kwargs = client.post.call_args.kwargs
        assert kwargs["headers"]["x-api-key"] == "secret-key"
        assert kwargs["headers"]["Accept"] == "application/json"
        # The key must never leak into the request body.
        assert "secret-key" not in str(kwargs["json"])
        assert client.post.call_args.args[0] == EXA_API_URL


class TestExaConnectorFailureIsolation:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_http_error_is_absorbed_not_raised(self):
        """A failing connector must degrade to zero sources so RRF fusion survives."""
        ctx = MagicMock()
        client = MagicMock()
        client.post = AsyncMock(side_effect=Exception("429 rate limited"))
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.connectors.exa.httpx.AsyncClient", return_value=ctx):
            result = await ExaConnector(api_key="k").search("q")

        assert result.sources == []
        assert result.connector_name == "exa"


# ---------------------------------------------------------------------------
# SerpApi
# ---------------------------------------------------------------------------


class TestSerpApiConnectorBasics:

    @pytest.mark.unit
    def test_connector_name(self):
        assert SerpApiConnector(api_key="k").name == "serpapi"

    @pytest.mark.unit
    def test_is_configured_with_key(self):
        assert SerpApiConnector(api_key="k").is_configured() is True

    @pytest.mark.unit
    def test_is_not_configured_without_key(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.serpapi_api_key", "")
        assert SerpApiConnector(api_key="").is_configured() is False

    @pytest.mark.unit
    def test_probe_url_does_not_spend_a_query(self):
        """Health probe must hit the API root, never the billed search path."""
        probe = SerpApiConnector(api_key="k")._probe_url()
        assert probe == "https://serpapi.com"
        assert "/search" not in probe

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unconfigured_returns_empty_without_network(self):
        """An unset key must short-circuit before any HTTP call."""
        with patch("src.connectors.serpapi.httpx.AsyncClient") as client_cls:
            result = await SerpApiConnector(api_key="").search("anything")
        client_cls.assert_not_called()
        assert result.sources == []
        assert result.connector_name == "serpapi"


class TestSerpApiConnectorParsing:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_maps_live_response_shape(self):
        ctx, _ = _client_returning(_response(SERPAPI_LIVE_SHAPE))
        with patch("src.connectors.serpapi.httpx.AsyncClient", return_value=ctx):
            result = await SerpApiConnector(api_key="k").search("q")

        assert len(result.sources) == 2
        first = result.sources[0]
        assert first.id.startswith("sp_")
        assert first.title == "Retrieval-Augmented Generation survey"
        assert first.url == "https://arxiv.org/abs/2405.07437"
        assert first.content == "A survey of RAG evaluation approaches."
        assert first.connector == "serpapi"
        # Rank-based scores: 1.0, 0.5
        assert first.score == pytest.approx(1.0)
        assert result.sources[1].score == pytest.approx(0.5)
        assert result.total_results == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_organic_band_is_zero_not_error(self):
        """No `organic_results` key (knowledge-panel-only answer) fuses quietly."""
        ctx, _ = _client_returning(_response({"answer_box": {}}))
        with patch("src.connectors.serpapi.httpx.AsyncClient", return_value=ctx):
            result = await SerpApiConnector(api_key="k").search("q")

        assert result.sources == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_null_organic_band_is_zero_not_error(self):
        ctx, _ = _client_returning(_response({"organic_results": None}))
        with patch("src.connectors.serpapi.httpx.AsyncClient", return_value=ctx):
            result = await SerpApiConnector(api_key="k").search("q")

        assert result.sources == []


class TestSerpApiConnectorRequestContract:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_num_is_clamped_to_engine_maximum(self):
        """Google rejects num > 100; the connector must clamp, not forward."""
        ctx, client = _client_returning(_response(SERPAPI_LIVE_SHAPE))
        with patch("src.connectors.serpapi.httpx.AsyncClient", return_value=ctx):
            await SerpApiConnector(api_key="k").search("q", top_k=500)

        params = client.get.call_args.kwargs["params"]
        assert int(params["num"]) == SERPAPI_MAX_NUM

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_key_is_sent_as_documented_query_param(self):
        """SerpApi's documented transport carries the key as a query parameter."""
        ctx, client = _client_returning(_response(SERPAPI_LIVE_SHAPE))
        with patch("src.connectors.serpapi.httpx.AsyncClient", return_value=ctx):
            await SerpApiConnector(api_key="secret-key").search("q")

        kwargs = client.get.call_args.kwargs
        assert kwargs["params"]["api_key"] == "secret-key"
        assert kwargs["headers"]["Accept"] == "application/json"
        assert client.get.call_args.args[0] == SERPAPI_API_URL

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_engine_is_forwarded(self):
        ctx, client = _client_returning(_response(SERPAPI_LIVE_SHAPE))
        with patch("src.connectors.serpapi.httpx.AsyncClient", return_value=ctx):
            await SerpApiConnector(api_key="k", engine="bing").search("q")

        assert client.get.call_args.kwargs["params"]["engine"] == "bing"


class TestSerpApiConnectorFailureIsolation:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_http_error_is_absorbed_not_raised(self):
        """A failing connector must degrade to zero sources so RRF fusion survives."""
        ctx = MagicMock()
        client = MagicMock()
        client.get = AsyncMock(side_effect=Exception("429 rate limited"))
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.connectors.serpapi.httpx.AsyncClient", return_value=ctx):
            result = await SerpApiConnector(api_key="k").search("q")

        assert result.sources == []
        assert result.connector_name == "serpapi"