"""Provider endpoint catalog: periodic refresh from OpenRouter plus metrics measured by the proxy."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import Config, ModelConfig

log = logging.getLogger("orproxy.catalog")

PER_M = 1_000_000


def _price(v: Any) -> float | None:
    """OpenRouter prices are $/token strings; convert to $/1M tokens."""
    if v in (None, ""):
        return None
    try:
        return float(v) * PER_M
    except (TypeError, ValueError):
        return None


@dataclass
class Prices:
    input: float | None
    output: float | None
    cache_read: float | None


@dataclass
class Endpoint:
    tag: str                       # e.g. "deepinfra/fp8"; used in provider.order / provider.only
    provider_name: str             # e.g. "DeepInfra"; what OpenRouter reports back in responses
    quantization: str
    context_length: int | None
    max_completion_tokens: int | None
    prices: Prices
    discount: float
    overrides: list[dict[str, Any]]
    status: int | None
    uptime: dict[str, float | None]
    latency: dict[str, float] | None
    throughput: dict[str, float] | None
    supported_parameters: set[str]
    supports_caching: bool
    zdr: bool | None = None        # None = unknown (ZDR list not fetched)

    @property
    def slug(self) -> str:
        return self.tag.split("/", 1)[0]

    def matches(self, names: list[str]) -> bool:
        keys = {self.tag.lower(), self.slug.lower(), self.provider_name.lower()}
        return any(n in keys for n in names)

    def prices_for(self, prompt_tokens: int, now: datetime | None = None) -> Prices:
        """Base prices with OpenRouter's overrides applied (time-of-day windows, long-context tiers)."""
        if not self.overrides:
            return self.prices
        now = now or datetime.now(timezone.utc)
        hhmm = now.hour * 100 + now.minute
        p = Prices(self.prices.input, self.prices.output, self.prices.cache_read)
        for o in self.overrides:
            if "min_prompt_tokens" in o and prompt_tokens < int(o["min_prompt_tokens"]):
                continue
            if "utc_start" in o or "utc_end" in o:
                start, end = int(o.get("utc_start", 0)), int(o.get("utc_end", 0))
                inside = start <= hhmm < end if start < end else (hhmm >= start or hhmm < end)
                if not inside:
                    continue
            if "prompt" in o:
                p.input = _price(o["prompt"])
            if "completion" in o:
                p.output = _price(o["completion"])
            if "input_cache_read" in o:
                p.cache_read = _price(o["input_cache_read"])
        return p

    @classmethod
    def from_api(cls, e: dict[str, Any]) -> "Endpoint":
        pr = e.get("pricing") or {}
        cache_read = _price(pr.get("input_cache_read"))
        return cls(
            tag=e.get("tag") or e.get("provider_name", "?").lower(),
            provider_name=e.get("provider_name") or "?",
            quantization=(e.get("quantization") or "unknown").lower(),
            context_length=e.get("context_length"),
            max_completion_tokens=e.get("max_completion_tokens"),
            prices=Prices(_price(pr.get("prompt")), _price(pr.get("completion")), cache_read),
            discount=float(pr.get("discount") or 0),
            overrides=list(pr.get("overrides") or []),
            status=e.get("status"),
            uptime={"5m": e.get("uptime_last_5m"), "30m": e.get("uptime_last_30m"), "1d": e.get("uptime_last_1d")},
            latency=e.get("latency_last_30m") if isinstance(e.get("latency_last_30m"), dict) else None,
            throughput=e.get("throughput_last_30m") if isinstance(e.get("throughput_last_30m"), dict) else None,
            supported_parameters=set(e.get("supported_parameters") or []),
            supports_caching=bool(e.get("supports_implicit_caching")) or cache_read is not None,
        )


@dataclass
class Ewma:
    value: float | None = None
    samples: int = 0
    updated: float = 0.0

    def add(self, x: float, alpha: float = 0.2) -> None:
        self.value = x if self.value is None else alpha * x + (1 - alpha) * self.value
        self.samples += 1
        self.updated = time.time()


@dataclass
class Measured:
    """What the proxy itself observed for one provider of one model."""
    cache_hit_rate: Ewma = field(default_factory=Ewma)
    throughput: Ewma = field(default_factory=Ewma)
    ttft_ms: Ewma = field(default_factory=Ewma)
    requests: int = 0
    errors: int = 0


@dataclass
class ModelState:
    cfg: ModelConfig
    endpoints: list[Endpoint] = field(default_factory=list)
    refreshed_at: float = 0.0
    last_error: str | None = None
    measured: dict[str, Measured] = field(default_factory=dict)  # key: provider_name

    def measured_for(self, provider_name: str) -> Measured:
        m = self.measured.get(provider_name)
        if m is None:
            m = self.measured[provider_name] = Measured()
        return m


class Catalog:
    def __init__(self, cfg: Config, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self.models: dict[str, ModelState] = {name: ModelState(mc) for name, mc in cfg.models.items()}
        self._tasks: list[asyncio.Task[None]] = []
        self._zdr: set[tuple[str, str]] = set()   # (model_id, tag)
        self._zdr_at = 0.0
        self._zdr_lock = asyncio.Lock()

    def resolve(self, model: str) -> ModelState | None:
        """Finds the configured alias for a requested model name. Clients such as LiteLLM's openrouter
        provider keep the "openrouter/" prefix on names without an author part, so
        "openrouter/deepseek-v4-flash" resolves to the alias "deepseek-v4-flash"."""
        state = self.models.get(model)
        if state is None and model.startswith("openrouter/"):
            state = self.models.get(model[len("openrouter/"):])
        return state

    async def _zdr_endpoints(self) -> set[tuple[str, str]]:
        async with self._zdr_lock:
            if time.time() - self._zdr_at > 300:
                r = await self.client.get(f"{self.cfg.base_url}/endpoints/zdr",
                                          headers={"Authorization": f"Bearer {self.cfg.api_key}"}, timeout=30)
                r.raise_for_status()
                self._zdr = {(e.get("model_id"), e.get("tag")) for e in r.json().get("data") or []}
                self._zdr_at = time.time()
            return self._zdr

    async def refresh(self, state: ModelState) -> None:
        url = f"{self.cfg.base_url}/models/{state.cfg.model}/endpoints"
        try:
            r = await self.client.get(url, headers={"Authorization": f"Bearer {self.cfg.api_key}"}, timeout=30)
            r.raise_for_status()
            data = r.json().get("data") or {}
            eps = [Endpoint.from_api(e) for e in data.get("endpoints") or []]
            if not eps:
                raise ValueError("OpenRouter returned no endpoints")
            if state.cfg.filters.zdr_only:
                zdr = await self._zdr_endpoints()
                model_id = data.get("id") or state.cfg.model
                for ep in eps:
                    ep.zdr = (model_id, ep.tag) in zdr
            state.endpoints = eps
            state.refreshed_at = time.time()
            state.last_error = None
            log.info("refreshed %s: %d endpoints", state.cfg.name, len(eps))
        except Exception as exc:  # keep serving the last known catalog
            state.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("refresh of %s failed: %s", state.cfg.name, state.last_error)

    async def _loop(self, state: ModelState) -> None:
        while True:
            await self.refresh(state)
            # retry sooner while we have nothing to route with
            delay = state.cfg.refresh_interval if state.endpoints else min(60.0, state.cfg.refresh_interval)
            await asyncio.sleep(delay)

    async def start(self) -> None:
        await asyncio.gather(*(self.refresh(s) for s in self.models.values()))
        for s in self.models.values():
            # first refresh already done above
            self._tasks.append(asyncio.create_task(self._delayed_loop(s)))

    async def _delayed_loop(self, state: ModelState) -> None:
        await asyncio.sleep(state.cfg.refresh_interval if state.endpoints else 60.0)
        await self._loop(state)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
