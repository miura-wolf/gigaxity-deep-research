"""Client-side sliding-window RPM limiter for the LLM endpoint.

Hosted LLM endpoints cap requests per minute — NVIDIA NIM's free tier rejects
bursts with 503 `ResourceExhausted` around ~45 RPM, and the OpenAI SDK's own
retry loop makes the burst WORSE: every rejected call is retried up to
`llm_max_retries` times, so hitting the cap mid-research converts one research
request (synthesis + RCS summaries + gate scorer = many LLM calls) into a burst
of retries against the very budget that is already exhausted.

The limiter is deliberately CLIENT-side and deliberately a single
process-wide instance shared by every call made through
`OpenRouterClient.chat_completion`:

* one server process = one request budget against the endpoint, regardless of
  how many client objects exist — every server-owned LLM call (synthesis
  stages, RCS summaries, the gate scorer) funnels through that one method;
* `RPM <= 0` disables it entirely (the shipped default): OpenRouter's default
  endpoint has no per-minute cap worth throttling for, so installing this
  release changes no behaviour on its own. Set e.g. `RESEARCH_LLM_RPM=40` to
  stay under NIM free-tier's ~45 RPM with headroom.

Why a SLIDING window and not a fixed one: a fixed window admits a full quota at
the end of minute N and another full quota at the start of minute N+1 — a 2x
burst exactly at the boundary, which is precisely the pattern that trips
per-minute caps. The sliding window spreads the allowance by waiting until the
oldest retained request ages out of the last 60 seconds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Deque

from .config import settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


class SlidingWindowRateLimiter:
    """Async sliding-window limiter for a per-minute request budget.

    The window is a deque of monotonic send-timestamps. `acquire()` admits a
    call when fewer than `rpm` entries are younger than the window; otherwise
    it sleeps until the oldest entry ages out and re-checks. All checking and
    recording happens under one asyncio lock, so concurrent callers queue
    instead of racing — N concurrent `acquire()`s can never admit more than
    the budget allows in any 60-second slice.

    `acquire()` is also deliberately cancel-safe: it only awaits `sleep` and
    the lock, so an aborted request (MCP client disconnect, wall-clock cap)
    stops waiting immediately and consumes nothing.
    """

    def __init__(self, rpm: int = 0, window_s: float = WINDOW_SECONDS):
        self.rpm = int(rpm)
        self.window_s = float(window_s)
        self._lock = asyncio.Lock()
        self._sent: Deque[float] = deque(maxlen=max(1, self.rpm)) if self.rpm > 0 else deque()
        # Cheap observability: how many calls had to wait, and for how long in
        # total. Surfaced to tests and ad-hoc debugging; no metric plumbing.
        self.waited_calls: int = 0
        self.waited_total_s: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.rpm > 0

    def _prune(self, now: float) -> None:
        """Drop timestamps that have aged out of the window."""
        cutoff = now - self.window_s
        sent = self._sent
        while sent and sent[0] <= cutoff:
            sent.popleft()

    async def acquire(self) -> float:
        """Wait until one request slot is free; return the seconds waited.

        Returns 0.0 immediately when disabled — the disabled path must stay
        allocation-free and lock-free so the shipped default is a true no-op.
        """
        if not self.enabled:
            return 0.0

        waited = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                if len(self._sent) < self.rpm:
                    self._sent.append(now)
                    if waited > 0:
                        self.waited_calls += 1
                        self.waited_total_s += waited
                        logger.info(
                            "llm rate limiter: admitted call after %.2fs wait "
                            "(rpm=%d, waited_calls=%d)",
                            waited, self.rpm, self.waited_calls,
                        )
                    return waited
                # Window full: sleep until the OLDEST request ages out. Sleep
                # OUTSIDE the lock — holding it while waiting would block even
                # the admissions that are already free.
                sleep_s = self._sent[0] + self.window_s - now
            await asyncio.sleep(max(0.0, sleep_s))
            waited += max(0.0, sleep_s)


_limiter: SlidingWindowRateLimiter | None = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Process-wide limiter for the LLM endpoint (lazily built).

    Lazy (not built at import) so tests that monkeypatch `settings.llm_rpm`
    can call `reset_rate_limiter()` and observe the new value, and so merely
    importing the module never allocates asyncio primitives.
    """
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindowRateLimiter(rpm=settings.llm_rpm)
    return _limiter


def reset_rate_limiter() -> None:
    """Drop the cached limiter so the next call re-reads settings.

    Test/deploy hook: settings are read once per process by design (a mid-run
    config change silently applying to some calls and not others is worse than
    a restart), so this exists for tests and deliberate reconfiguration.
    """
    global _limiter
    _limiter = None
