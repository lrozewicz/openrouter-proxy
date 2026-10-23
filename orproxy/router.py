"""Choosing providers for a request: filtering, priority ranking and sticky sessions."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .catalog import Endpoint, ModelState
from .config import CRITERIA, QUANT_RANK, Criterion, ModelConfig


class NoProviderError(Exception):
    pass


@dataclass
class RequestInfo:
    prompt_tokens: int
    max_tokens: int
    required_params: set[str]
    session_key: str


@dataclass
class Candidate:
    ep: Endpoint
    features: dict[str, float | None]
    rejected: str | None = None


@dataclass
class Decision:
    mode: str                                   # ranked | sticky | openrouter | best_effort
    reason: str
    order: list[str] = field(default_factory=list)
    only: list[str] = field(default_factory=list)
    allow_fallbacks: bool = True
    candidates: list[Candidate] = field(default_factory=list)

    def provider_prefs(self) -> dict[str, Any] | None:
        if not self.order:
            return None
        return {"order": self.order, "only": self.only, "allow_fallbacks": self.allow_fallbacks}


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_text_len(p.get("text") if isinstance(p, dict) else p) for p in content)
    if content is None:
        return 0
    return len(json.dumps(content, ensure_ascii=False))


def _first_text(content: Any, limit: int = 4000) -> str:
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))[:limit]
    return ""


def analyze_request(body: dict[str, Any], chars_per_token: float, session_header: str | None) -> RequestInfo:
    chars = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict):
                chars += _text_len(m.get("content")) + 8
                if m.get("tool_calls"):
                    chars += _text_len(m["tool_calls"])
    elif "prompt" in body:
        chars += _text_len(body["prompt"])
    if body.get("tools"):
        chars += _text_len(body["tools"])
    prompt_tokens = int(chars / chars_per_token)

    required: set[str] = set()
    if body.get("tools"):
        required.add("tools")
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        required.add("structured_outputs")
    elif rf:
        required.add("response_format")

    if session_header:
        seed = session_header
    else:
        # a coding session is identified well enough by its system prompt + first user message
        system = first_user = ""
        for m in messages if isinstance(messages, list) else []:
            if not isinstance(m, dict):
                continue
            if m.get("role") in ("system", "developer") and not system:
                system = _first_text(m.get("content"))
            elif m.get("role") == "user":
                first_user = _first_text(m.get("content"))
                break
        seed = f"{body.get('user', '')}\x00{system}\x00{first_user}"
    session_key = hashlib.sha256(seed.encode("utf-8", "replace")).hexdigest()[:24]

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 0
    return RequestInfo(prompt_tokens, int(max_tokens) if isinstance(max_tokens, (int, float)) else 0,
                       required, session_key)


def compute_features(ep: Endpoint, state: ModelState, prompt_tokens: int) -> dict[str, float | None]:
    cfg = state.cfg
    prices = ep.prices_for(prompt_tokens)
    measured = state.measured.get(ep.provider_name)

    hit = None
    if measured and measured.cache_hit_rate.samples >= cfg.min_cache_samples:
        hit = measured.cache_hit_rate.value
    assumed_hit = hit if hit is not None else (cfg.expected_cache_hit_rate if ep.supports_caching else 0.0)

    eff = None
    if prices.input is not None:
        cache_price = prices.cache_read if prices.cache_read is not None else prices.input
        eff = (1 - assumed_hit) * prices.input + assumed_hit * cache_price
    blended = None
    if eff is not None and prices.output is not None:
        s = cfg.blend_output_share
        blended = (1 - s) * eff + s * prices.output

    pct = cfg.metrics_percentile
    or_tps = ep.throughput.get(pct) if ep.throughput else None
    my_tps = measured.throughput.value if measured and measured.throughput.samples else None
    if cfg.throughput_source == "openrouter":
        tps = or_tps
    elif cfg.throughput_source == "measured":
        tps = my_tps
    else:
        tps = or_tps if or_tps is not None else my_tps

    latency = ep.latency.get(pct) if ep.latency else None
    if latency is None and measured and measured.ttft_ms.samples:
        latency = measured.ttft_ms.value

    return {
        "input_price": prices.input,
        "output_price": prices.output,
        "cache_read_price": prices.cache_read,
        "effective_input_price": eff,
        "blended_price": blended,
        "throughput": tps,
        "latency": latency,
        "uptime": ep.uptime.get(cfg.uptime_window),
        "cache_hit_rate": hit,
        "quantization": float(QUANT_RANK[ep.quantization]) if ep.quantization in QUANT_RANK else None,
        "discount": ep.discount,
        "context_length": float(ep.context_length) if ep.context_length else None,
    }


def compat_reason(ep: Endpoint, req: RequestInfo) -> str | None:
    """Hard requirements that are never relaxed: the request must physically fit the endpoint."""
    if ep.context_length and req.prompt_tokens + req.max_tokens > ep.context_length:
        return f"context {ep.context_length} too small"
    missing = req.required_params - ep.supported_parameters
    if missing and ep.supported_parameters:
        return f"unsupported params {sorted(missing)}"
    return None


def filter_reason(ep: Endpoint, f: dict[str, float | None], cfg: ModelConfig) -> str | None:
    flt = cfg.filters
    strict = flt.strict_metrics

    if cfg.exclude_unhealthy and ep.status is not None and ep.status < 0:
        return f"status {ep.status}"
    if flt.zdr_only and ep.zdr is False:
        return "not a ZDR endpoint"
    if flt.include_providers and not ep.matches(flt.include_providers):
        return "not in include_providers"
    if flt.exclude_providers and ep.matches(flt.exclude_providers):
        return "in exclude_providers"

    def over(value: float | None, limit: float | None, name: str, unit: str) -> str | None:
        if limit is None:
            return None
        if value is None:
            return f"{name} unknown" if strict else None
        return f"{name} {value:.4g}{unit} > {limit:g}" if value > limit else None

    def under(value: float | None, limit: float | None, name: str, unit: str) -> str | None:
        if limit is None:
            return None
        if value is None:
            return f"{name} unknown" if strict else None
        return f"{name} {value:.4g}{unit} < {limit:g}" if value < limit else None

    checks = (
        over(f["input_price"], flt.max_input_price, "input price", "$"),
        over(f["output_price"], flt.max_output_price, "output price", "$"),
        under(f["throughput"], flt.min_throughput, "throughput", "tok/s"),
        over(f["latency"], flt.max_latency, "latency", "ms"),
        under(f["uptime"], flt.min_uptime, "uptime", "%"),
        under(f["cache_hit_rate"], flt.min_cache_hit_rate, "cache hit rate", ""),
    )
    for c in checks:
        if c:
            return c

    if flt.max_cache_read_price is not None:
        if f["cache_read_price"] is None:
            return "no cache pricing"
        if f["cache_read_price"] > flt.max_cache_read_price:
            return f"cache read price {f['cache_read_price']:.4g}$ > {flt.max_cache_read_price:g}"
    if flt.require_cache and not ep.supports_caching:
        return "no prompt caching"

    q = ep.quantization
    if q == "unknown":
        if not flt.allow_unknown_quantization and (flt.quantizations or flt.min_quantization):
            return "quantization unknown"
    else:
        if flt.quantizations and q not in flt.quantizations:
            return f"quantization {q} not allowed"
        if flt.min_quantization and QUANT_RANK.get(q, 0) < QUANT_RANK[flt.min_quantization.lower()]:
            return f"quantization {q} below {flt.min_quantization}"
    return None


def rank(cands: list[Candidate], criteria: list[Criterion]) -> list[Candidate]:
    """Lexicographic ranking with tolerance: providers within `tolerance` of the best value
    on a criterion form one tier and are ordered by the next criterion."""
    if len(cands) <= 1 or not criteria:
        return list(cands)
    crit, rest = criteria[0], criteria[1:]
    higher = CRITERIA[crit.by]
    known = [c for c in cands if c.features.get(crit.by) is not None]
    unknown = [c for c in cands if c.features.get(crit.by) is None]
    known.sort(key=lambda c: c.features[crit.by], reverse=higher)  # type: ignore[arg-type, return-value]

    out: list[Candidate] = []
    while known:
        best = known[0].features[crit.by]
        assert best is not None
        margin = abs(best) * crit.tolerance
        tier = [c for c in known
                if (c.features[crit.by] >= best - margin if higher else c.features[crit.by] <= best + margin)]  # type: ignore[operator]
        out += rank(tier, rest)
        known = known[len(tier):]
    out += rank(unknown, rest)
    return out


class StickyStore:
    def __init__(self, max_entries: int = 20000):
        self._data: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.max_entries = max_entries

    def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if not item:
            return None
        tag, expires = item
        if expires < time.time():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return tag

    def set(self, key: str, tag: str, ttl: float) -> None:
        self._data[key] = (tag, time.time() + ttl)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def evaluate(state: ModelState, req: RequestInfo) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """Returns (ranked passing candidates, compatible candidates, all candidates with reasons)."""
    cfg = state.cfg
    all_c: list[Candidate] = []
    for ep in state.endpoints:
        c = Candidate(ep, compute_features(ep, state, req.prompt_tokens))
        c.rejected = compat_reason(ep, req) or filter_reason(ep, c.features, cfg)
        all_c.append(c)
    compatible = [c for c in all_c if compat_reason(c.ep, req) is None]
    passing = rank([c for c in all_c if c.rejected is None], cfg.priority)
    return passing, compatible, all_c


def decide(state: ModelState, req: RequestInfo, sticky: StickyStore) -> Decision:
    cfg = state.cfg
    r = cfg.routing
    if not state.endpoints:
        return Decision("openrouter", "provider catalog not loaded yet")

    passing, compatible, all_c = evaluate(state, req)
    mode, reason = "ranked", f"{len(passing)}/{len(all_c)} providers pass filters"
    if not passing:
        if r.on_no_match == "error":
            raise NoProviderError(f"no provider for {cfg.name} matches the configured filters")
        if r.on_no_match == "openrouter" or not compatible:
            return Decision("openrouter", "no provider matches filters; using OpenRouter default routing",
                            candidates=all_c)
        passing = rank(compatible, cfg.priority)
        mode, reason = "best_effort", "no provider matches filters; ranking all compatible providers"

    above = r.switch_below_context is not None and req.prompt_tokens >= r.switch_below_context
    if above:
        if r.above_context == "openrouter":
            return Decision("openrouter", f"~{req.prompt_tokens} tokens >= {r.switch_below_context}; "
                                          "auto-switching disabled", candidates=all_c)
        if r.above_context == "sticky":
            tag = sticky.get(f"{cfg.name}:{req.session_key}")
            pinned = next((c for c in passing if c.ep.tag == tag), None) if tag else None
            if pinned:
                passing = [pinned] + [c for c in passing if c is not pinned]
                mode = "sticky"
                reason = f"~{req.prompt_tokens} tokens >= {r.switch_below_context}; keeping {tag} (cache)"

    tags = [c.ep.tag for c in passing]
    order = tags[: max(1, r.max_providers)]
    if r.allow_fallbacks:
        return Decision(mode, reason, order, tags, True, all_c)
    return Decision(mode, reason, order, order, False, all_c)


def tag_for_provider(decision: Decision, provider_name: str | None) -> str | None:
    if not provider_name:
        return None
    for tag in decision.only or decision.order:
        for c in decision.candidates:
            if c.ep.tag == tag and c.ep.provider_name == provider_name:
                return tag
    return None
