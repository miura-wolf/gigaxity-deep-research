"""Client-side RPM limiter — invariants for the sliding window and its wiring.

Why client-side: NVIDIA NIM free-tier rejects bursts around ~45 RPM with 503
`ResourceExhausted`, and the OpenAI SDK RETRIES the rejection, so a burst
converts one research request (synthesis + RCS + gate = many LLM calls) into a
burst of retries against an already-exhausted budget. The limiter waits for a
slot instead of letting the endpoint answer 429/503.

These tests pin the contract: disabled (rpm=0) is a true no-op and the
shipped default; the window slides (no 2x burst at a minute boundary); every
call through chat_completion draws from one process-wide budget; the wait
lives inside the heartbeat/wall-clock-cap coverage; and an aborted request
stops waiting immediately, consuming no slot.
"""

import asyncio
import time
import types

import pytest

from src import progress
from src.config import settings
from src.llm_client import OpenRouterClient
from src.rate_limit import (
    SlidingWindowRateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
    WINDOW_SECONDS,
)


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Every test starts from a clean limiter cache.

    The limiter is process-wide by design (one server = one request budget
    against the endpoint), so cross-test contamination is the one leak this
    fixture exists to close: without it, a test enabling rpm=10 would leave
    the next test's calls waiting behind stale timestamps.
    """
    reset_rate_limiter()
    yield
    reset_rate_limiter()


class TestDisabledByDefault:

    def test_shipped_default_is_zero(self):
        assert settings.llm_rpm == 0

    @pytest.mark.asyncio
    async def test_disabled_is_noop(self):
        lim = SlidingWindowRateLimiter(rpm=0)
        assert lim.enabled is False
        assert await lim.acquire() == 0.0


@pytest.mark.asyncio
class TestSlidingWindowArithmetic:

    async def test_admits_within_budget_without_waiting(self):
        lim = SlidingWindowRateLimiter(rpm=5, window_s=WINDOW_SECONDS)
        waited = [await lim.acquire() for _ in range(5)]
        assert waited == [0.0] * 5

    async def test_sixth_call_waits_for_the_oldest_to_age_out(self):
        lim = SlidingWindowRateLimiter(rpm=2, window_s=0.2)
        await lim.acquire()
        await lim.acquire()
        start = time.monotonic()
        waited = await lim.acquire()
        elapsed = time.monotonic() - start
        # The third call waited ~one window for the first to age out.
        assert waited > 0.05
        assert elapsed >= waited * 0.9

    async def test_window_slides_no_boundary_burst(self):
        """The sliding window must NOT admit a fresh quota at a minute
        boundary the way a fixed window does — 2rpm means at most 2 sends in
        ANY 60s slice, not 2 at the end of minute N plus 2 more right after.
        """
        lim = SlidingWindowRateLimiter(rpm=2, window_s=0.2)
        await lim.acquire()          # send A at t=0
        await asyncio.sleep(0.1)
        await lim.acquire()          # send B at t=0.1
        # At t≈0.21: A (t=0) has aged out, B (t=0.1) is still inside until
        # t=0.3. So a third call admits immediately (A's slot freed), but a
        # fourth must WAIT for B — a fixed window would admit both at once.
        await asyncio.sleep(0.11)
        w3 = await lim.acquire()
        assert w3 == pytest.approx(0.0, abs=0.02)
        w4 = await lim.acquire()
        assert w4 > 0.0

    async def test_concurrent_calls_never_exceed_budget(self):
        """N concurrent acquires must queue, not race: in any window slice at
        most `rpm` timestamps are recorded."""
        lim = SlidingWindowRateLimiter(rpm=3, window_s=10.0)
        await asyncio.gather(*(lim.acquire() for _ in range(3)))
        assert len(lim._sent) == 3
        # The 4th must wait a full window — assert it does NOT record a 4th
        # timestamp immediately (it is still queued, no slot consumed).
        task = asyncio.create_task(lim.acquire())
        await asyncio.sleep(0.05)
        assert len(lim._sent) == 3
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancelled_waiter_consumes_no_slot(self):
        lim = SlidingWindowRateLimiter(rpm=1, window_s=10.0)
        await lim.acquire()  # budget consumed
        task = asyncio.create_task(lim.acquire())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The aborted waiter recorded nothing — the budget is intact.
        assert len(lim._sent) == 1


class TestProcessWideInstance:

    def test_get_rate_limiter_is_shared(self):
        a = get_rate_limiter()
        b = get_rate_limiter()
        assert a is b

    def test_reset_rate_limiter_rebuilds_from_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_rpm", 0)
        first = get_rate_limiter()
        reset_rate_limiter()
        monkeypatch.setattr(settings, "llm_rpm", 40)
        second = get_rate_limiter()
        assert first is not second
        assert second.rpm == 40

    def test_openrouter_client_uses_the_shared_limiter(self):
        """Wiring, not coincidence: the client must draw from the one
        process-wide budget, so synthesis stages, RCS summaries and the gate
        scorer all pace through the same window."""
        c = OpenRouterClient(api_key="k")
        assert c._rate_limiter is get_rate_limiter()


class _FakeSDK:
    """Minimal AsyncOpenAI stand-in: calls are recorded, never hit network."""

    def __init__(self):
        self.calls = 0

        class _Completions:
            async def create(inner_self, **kwargs):
                self.calls += 1
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="answer", reasoning_content=None, reasoning=None),
                        finish_reason="stop",
                    )]
                )

        self.chat = types.SimpleNamespace(completions=_Completions())


class TestChatCompletionIntegration:

    @pytest.mark.asyncio
    async def test_every_call_passes_the_limiter(self, monkeypatch):
        """A capped free tier must see at most rpm calls per window even when
        the pipeline fires several in quick succession."""
        monkeypatch.setattr(settings, "llm_rpm", 2)
        reset_rate_limiter()
        client = OpenRouterClient(api_key="k")
        sdk = _FakeSDK()
        monkeypatch.setattr(client, "_client", sdk)

        # Two calls land inside the budget.
        await client.chat_completion(messages=[{"role": "user", "content": "x"}])
        await client.chat_completion(messages=[{"role": "user", "content": "x"}])
        assert sdk.calls == 2

        # The third queues: give it a moment, it must NOT have gone through.
        task = asyncio.create_task(
            client.chat_completion(messages=[{"role": "user", "content": "x"}])
        )
        await asyncio.sleep(0.05)
        assert sdk.calls == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_disabled_limiter_never_blocks_a_call(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_rpm", 0)
        reset_rate_limiter()
        client = OpenRouterClient(api_key="k")
        sdk = _FakeSDK()
        monkeypatch.setattr(client, "_client", sdk)

        for _ in range(5):
            await client.chat_completion(messages=[{"role": "user", "content": "x"}])
        assert sdk.calls == 5

    @pytest.mark.asyncio
    async def test_wait_is_inside_the_wall_clock_cap(self, monkeypatch):
        """The rate-limit wait must count against the wall-clock cap — a
        queued request that would exceed a deliberate ceiling must be cut,
        not silently overdraw it."""
        monkeypatch.setattr(settings, "llm_rpm", 1)
        monkeypatch.setattr(settings, "llm_wall_clock_cap", 1)
        reset_rate_limiter()
        client = OpenRouterClient(api_key="k")
        sdk = _FakeSDK()
        monkeypatch.setattr(client, "_client", sdk)

        # First call consumes the budget.
        await client.chat_completion(messages=[{"role": "user", "content": "x"}])
        # Second queues for a full window (60s) under a 1s cap -> TimeoutError.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                client.chat_completion(messages=[{"role": "user", "content": "x"}]),
                timeout=5.0,
            )

    @pytest.mark.asyncio
    async def test_wait_is_inside_the_heartbeat_coverage(self, monkeypatch):
        """A queued request must not read as a dead connection to the MCP
        client: while the limiter waits, the heartbeat keeps ticking. Pin the
        wiring: the 'waiting' tick is emitted before the acquire starts."""
        monkeypatch.setattr(settings, "llm_rpm", 1)
        reset_rate_limiter()
        client = OpenRouterClient(api_key="k")
        sdk = _FakeSDK()
        monkeypatch.setattr(client, "_client", sdk)

        messages = [{"role": "user", "content": "x"}]

        class _Recorder:
            def __init__(self):
                self.messages = []

            async def tick(self, message):
                self.messages.append(message)

        recorder = _Recorder()
        token = progress.install(recorder)
        try:
            await client.chat_completion(messages=messages)
            task = asyncio.create_task(
                client.chat_completion(messages=messages)
            )
            await asyncio.sleep(0.05)
            assert "rate limiter: waiting for a free slot" in recorder.messages
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            progress.reset(token)


class TestNIMFreeTierProfile:

    def test_documented_profile_stays_under_the_cap(self):
        """NIM free-tier rejects bursts around ~45 RPM; the recommended
        profile (40) must stay under it with headroom, and the knob must be
        wired so a deployment can set it without editing source."""
        lim = SlidingWindowRateLimiter(rpm=40)
        assert lim.rpm < 45
        assert settings.llm_rpm == 0  # documented, off by default
