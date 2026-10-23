"""Loading and validating the proxy configuration (YAML or JSON)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("orproxy.config")

# Order of quantization quality, used by `min_quantization` and the `quantization` criterion.
QUANT_RANK = {
    "int4": 1, "fp4": 2, "nvfp4": 2, "mxfp4": 2,
    "fp6": 3, "int8": 4, "fp8": 5,
    "bf16": 6, "fp16": 6, "fp32": 7,
}

# Criterion name -> True when a higher value is better.
CRITERIA = {
    "input_price": False,
    "output_price": False,
    "cache_read_price": False,
    "effective_input_price": False,
    "blended_price": False,
    "throughput": True,
    "latency": False,
    "uptime": True,
    "cache_hit_rate": True,
    "quantization": True,
    "discount": True,
    "context_length": True,
}

FILTER_KEYS = {
    "min_throughput", "max_latency", "max_input_price", "max_output_price",
    "max_cache_read_price", "min_uptime", "quantizations", "min_quantization",
    "allow_unknown_quantization", "require_cache", "min_cache_hit_rate",
    "include_providers", "exclude_providers", "strict_metrics", "zdr_only",
}
# Keys that count as "a real filter" (the spec requires at least one per model).
COUNTED_FILTERS = FILTER_KEYS - {"allow_unknown_quantization", "strict_metrics"}

ABOVE_CONTEXT_MODES = {"sticky", "ranked", "openrouter"}
NO_MATCH_MODES = {"best_effort", "openrouter", "error"}
METRIC_SOURCES = {"openrouter", "measured", "auto"}
PERCENTILES = {"p50", "p75", "p90", "p99"}
UPTIME_WINDOWS = {"5m": "uptime_last_5m", "30m": "uptime_last_30m", "1d": "uptime_last_1d"}

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?\s*$")
_DURATION_MULT = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, None: 1}


class ConfigError(ValueError):
    pass


def parse_duration(value: Any) -> float:
    """'5m' -> 300.0, '1h' -> 3600.0, 90 -> 90.0 (seconds)."""
    if isinstance(value, (int, float)):
        return float(value)
    m = _DURATION_RE.match(str(value))
    if not m:
        raise ConfigError(f"invalid duration: {value!r} (use e.g. 30s, 5m, 1h, 6h)")
    return float(m.group(1)) * _DURATION_MULT[m.group(2)]


def parse_tokens(value: Any) -> int:
    """'100k' -> 100000, '1M' -> 1000000, 64000 -> 64000."""
    if isinstance(value, int):
        return value
    s = str(value).strip().lower().replace("_", "")
    mult = 1
    if s.endswith("k"):
        mult, s = 1000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        raise ConfigError(f"invalid token count: {value!r}") from None


@dataclass
class Criterion:
    by: str
    tolerance: float  # relative, e.g. 0.1 = values within 10% of the best are a tie


@dataclass
class Filters:
    min_throughput: float | None = None        # tok/s
    max_latency: float | None = None           # ms (time to first token)
    max_input_price: float | None = None       # $ per 1M tokens
    max_output_price: float | None = None
    max_cache_read_price: float | None = None
    min_uptime: float | None = None            # percent
    quantizations: list[str] | None = None     # allowed list, e.g. [fp8, bf16]
    min_quantization: str | None = None        # e.g. fp8 -> rejects fp4/int4
    allow_unknown_quantization: bool = True
    require_cache: bool = False                # provider must offer prompt caching
    min_cache_hit_rate: float | None = None    # 0..1, measured by the proxy
    include_providers: list[str] | None = None
    exclude_providers: list[str] | None = None
    strict_metrics: bool = False               # reject providers whose filtered metric is unknown
    zdr_only: bool = False                     # only Zero Data Retention endpoints (account ZDR setting)


@dataclass
class Routing:
    switch_below_context: int | None = None    # auto-switching only below this many prompt tokens
    above_context: str = "sticky"              # sticky | ranked | openrouter
    sticky_ttl: float = 3600.0
    max_providers: int = 3                     # how many ranked providers go into provider.order
    allow_fallbacks: bool = True               # fall back to the remaining providers that passed filters
    on_no_match: str = "best_effort"           # best_effort | openrouter | error


@dataclass
class ModelConfig:
    name: str                                   # alias exposed to clients
    model: str                                  # OpenRouter id, e.g. deepseek/deepseek-v4-flash-0731
    filters: Filters
    priority: list[Criterion]
    routing: Routing
    refresh_interval: float = 900.0
    metrics_percentile: str = "p50"
    uptime_window: str = "1d"
    throughput_source: str = "auto"
    expected_cache_hit_rate: float = 0.5        # used for effective_input_price until measured
    blend_output_share: float = 0.1             # share of output tokens in blended_price
    min_cache_samples: int = 5
    exclude_unhealthy: bool = True
    extra_provider: dict[str, Any] = field(default_factory=dict)  # merged into provider{}


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    http_referer: str | None = None
    app_title: str | None = "openrouter-proxy"
    request_timeout: float = 600.0
    proxy_keys: set[str] = field(default_factory=set)
    allow_anonymous: bool = False
    passthrough_unknown_models: bool = True
    chars_per_token: float = 3.5
    models: dict[str, ModelConfig] = field(default_factory=dict)


def _read_secret(direct: Any, file: Any, env: str | None) -> str | None:
    if direct:
        return str(direct).strip()
    if file:
        p = Path(str(file))
        if p.is_file():
            return p.read_text().strip()
        log.warning("secret file %s does not exist", p)
    if env and os.environ.get(env):
        return os.environ[env].strip()
    return None


def _parse_priority(raw: Any, default_tol: float) -> list[Criterion]:
    if raw is None:
        raw = ["effective_input_price", "throughput", "uptime"]
    if not isinstance(raw, list) or not raw:
        raise ConfigError("priority must be a non-empty list")
    out = []
    for item in raw:
        if isinstance(item, str):
            by, tol = item, default_tol
        elif isinstance(item, dict) and "by" in item:
            by, tol = item["by"], float(item.get("tolerance", default_tol))
        else:
            raise ConfigError(f"invalid priority entry: {item!r}")
        if by not in CRITERIA:
            raise ConfigError(f"unknown priority criterion {by!r}; allowed: {', '.join(CRITERIA)}")
        out.append(Criterion(by, tol))
    return out


def _parse_filters(raw: dict[str, Any], name: str) -> Filters:
    unknown = set(raw) - FILTER_KEYS
    if unknown:
        raise ConfigError(f"model {name}: unknown filters {sorted(unknown)}")
    f = Filters(**raw)
    for key in ("quantizations", "include_providers", "exclude_providers"):
        val = getattr(f, key)
        if val is not None:
            setattr(f, key, [str(v).lower() for v in val])
    if f.min_quantization and f.min_quantization.lower() not in QUANT_RANK:
        raise ConfigError(f"model {name}: unknown min_quantization {f.min_quantization!r}")
    if f.min_cache_hit_rate is not None and f.min_cache_hit_rate > 1:
        f.min_cache_hit_rate /= 100.0  # accept 60 as 60%
    return f


def _parse_model(raw: dict[str, Any], defaults: dict[str, Any]) -> ModelConfig:
    merged = _deep_merge(defaults, raw)
    name = merged.get("name")
    model = merged.get("model")
    if not name or not model:
        raise ConfigError("each model needs `name` (alias) and `model` (OpenRouter id)")

    filters_raw = merged.get("filters") or {}
    if not any(filters_raw.get(k) not in (None, False, []) for k in COUNTED_FILTERS):
        raise ConfigError(f"model {name}: define at least one filter ({', '.join(sorted(COUNTED_FILTERS))})")
    filters = _parse_filters(filters_raw, name)

    r = merged.get("routing") or {}
    routing = Routing(
        switch_below_context=parse_tokens(r["switch_below_context"]) if r.get("switch_below_context") else None,
        above_context=r.get("above_context", "sticky"),
        sticky_ttl=parse_duration(r.get("sticky_ttl", "1h")),
        max_providers=int(r.get("max_providers", 3)),
        allow_fallbacks=bool(r.get("allow_fallbacks", True)),
        on_no_match=r.get("on_no_match", "best_effort"),
    )
    if routing.above_context not in ABOVE_CONTEXT_MODES:
        raise ConfigError(f"model {name}: above_context must be one of {sorted(ABOVE_CONTEXT_MODES)}")
    if routing.on_no_match not in NO_MATCH_MODES:
        raise ConfigError(f"model {name}: on_no_match must be one of {sorted(NO_MATCH_MODES)}")

    mc = ModelConfig(
        name=str(name),
        model=str(model),
        filters=filters,
        priority=_parse_priority(merged.get("priority"), float(merged.get("tolerance", 0.05))),
        routing=routing,
        refresh_interval=parse_duration(merged.get("refresh_interval", "15m")),
        metrics_percentile=merged.get("metrics_percentile", "p50"),
        uptime_window=str(merged.get("uptime_window", "1d")),
        throughput_source=merged.get("throughput_source", "auto"),
        expected_cache_hit_rate=float(merged.get("expected_cache_hit_rate", 0.5)),
        blend_output_share=float(merged.get("blend_output_share", 0.1)),
        min_cache_samples=int(merged.get("min_cache_samples", 5)),
        exclude_unhealthy=bool(merged.get("exclude_unhealthy", True)),
        extra_provider=dict(merged.get("provider") or {}),
    )
    if mc.metrics_percentile not in PERCENTILES:
        raise ConfigError(f"model {name}: metrics_percentile must be one of {sorted(PERCENTILES)}")
    if mc.uptime_window not in UPTIME_WINDOWS:
        raise ConfigError(f"model {name}: uptime_window must be one of {sorted(UPTIME_WINDOWS)}")
    if mc.throughput_source not in METRIC_SOURCES:
        raise ConfigError(f"model {name}: throughput_source must be one of {sorted(METRIC_SOURCES)}")
    if mc.refresh_interval < 30:
        raise ConfigError(f"model {name}: refresh_interval must be at least 30s")
    return mc


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike[str]) -> Config:
    """Loads the config from ORPROXY_CONFIG_YAML (inline YAML/JSON, e.g. set in a Portainer stack)
    or, when that is empty, from the file at `path`."""
    inline = os.environ.get("ORPROXY_CONFIG_YAML", "").strip()
    if inline:
        return build_config(yaml.safe_load(inline) or {})
    text = Path(path).read_text()
    raw = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
    return build_config(raw or {})


def build_config(raw: dict[str, Any]) -> Config:
    server = raw.get("server") or {}
    orc = raw.get("openrouter") or {}
    auth = raw.get("auth") or {}

    cfg = Config(
        host=server.get("host", "0.0.0.0"),
        port=int(server.get("port", 8080)),
        log_level=str(server.get("log_level", "info")),
        base_url=str(orc.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        api_key=_read_secret(orc.get("api_key"), orc.get("api_key_file"), "OPENROUTER_API_KEY") or "",
        http_referer=orc.get("http_referer"),
        app_title=orc.get("app_title", "openrouter-proxy"),
        request_timeout=parse_duration(orc.get("request_timeout", "10m")),
        allow_anonymous=bool(auth.get("allow_anonymous", False)),
        passthrough_unknown_models=bool(raw.get("passthrough_unknown_models", True)),
        chars_per_token=float(raw.get("chars_per_token", 3.5)),
    )
    if not cfg.api_key:
        raise ConfigError("OpenRouter API key missing (openrouter.api_key_file, openrouter.api_key or OPENROUTER_API_KEY)")

    keys = list(auth.get("api_keys") or [])
    keys_blob = _read_secret(None, auth.get("api_keys_file"), "PROXY_API_KEYS")
    if keys_blob:
        keys += re.split(r"[\s,]+", keys_blob)
    cfg.proxy_keys = {k.strip() for k in keys if k and k.strip()}
    if not cfg.proxy_keys and not cfg.allow_anonymous:
        raise ConfigError("no client API keys configured (auth.api_keys_file / auth.api_keys); "
                          "set auth.allow_anonymous: true to run without authentication")

    defaults = raw.get("defaults") or {}
    for m in raw.get("models") or []:
        mc = _parse_model(m, defaults)
        if mc.name in cfg.models:
            raise ConfigError(f"duplicate model alias {mc.name!r}")
        cfg.models[mc.name] = mc
    if not cfg.models:
        raise ConfigError("no models configured")
    return cfg
