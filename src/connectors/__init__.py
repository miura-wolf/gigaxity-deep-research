"""Search connectors for multi-source research."""

from .base import SearchResult, Source, Connector
from .searxng import SearXNGConnector
from .tavily import TavilyConnector
from .linkup import LinkUpConnector
from .brave import BraveConnector
from .exa import ExaConnector
from .serpapi import SerpApiConnector
from .ddgs import DDGSConnector

__all__ = [
    "SearchResult",
    "Source",
    "Connector",
    "SearXNGConnector",
    "TavilyConnector",
    "LinkUpConnector",
    "BraveConnector",
    "ExaConnector",
    "SerpApiConnector",
    "DDGSConnector",
]
