from datetime import datetime, timezone

import pytest

from orproxy.catalog import Catalog, Endpoint, ModelState
from orproxy.config import ConfigError, build_config, parse_duration, parse_tokens
from orproxy.router import StickyStore, analyze_request, decide


def ep(tag, prompt, completion, cache=None, tps=None, lat=None, uptime=100.0, quant="fp8", ctx=1_000_000,
       params=("tools",), overrides=None):
    return Endpoint.from_api({
        "tag": tag, "provider_name": tag.split("/")[0].title(), "quantization": quant, "context_length": ctx,
        "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6),
                    **({"input_cache_read": str(cache / 1e6)} if cache is not None else {}),
                    "overrides": overrides},
        "status": 0, "uptime_last_1d": uptime,
        "throughput_last_30m": {"p50": tps} if tps is not None else None,
        "latency_last_30m": {"p50": lat} if lat is not None else None,
        "supported_parameters": list(params),
    })


def make_cfg(**model):
    raw = {
        "openrouter": {"api_key": "k"},
        "auth": {"api_keys": ["t"]},
        "models": [{"name": "m", "model": "x/y", "filters": {"max_input_price": 1.0}, **model}],
    }
    return build_config(raw).models["m"]


def req(body, cpt=4.0):
    return analyze_request(body, cpt, None)


def test_parsers():
    assert parse_duration("5m") == 300
    assert parse_duration("6h") == 21600
    assert parse_tokens("100k") == 100_000
    assert parse_tokens("1M") == 1_000_000


def test_at_least_one_filter_required():
    with pytest.raises(ConfigError):
        build_config({"openrouter": {"api_key": "k"}, "auth": {"api_keys": ["t"]},
                      "models": [{"name": "m", "model": "x/y", "filters": {}}]})


def test_auth_required_unless_anonymous():
    base = {"openrouter": {"api_key": "k"}, "models": [{"name": "m", "model": "x/y",
                                                        "filters": {"min_uptime": 90}}]}
    with pytest.raises(ConfigError):
        build_config(base)
    assert build_config({**base, "auth": {"allow_anonymous": True}}).allow_anonymous


def test_filters_and_priority():
    cfg = make_cfg(filters={"max_input_price": 0.2, "min_throughput": 50, "min_quantization": "fp8"},
                   priority=["input_price", "throughput"])
    s = ModelState(cfg, [
        ep("cheap/fp4", 0.05, 0.2, tps=100, quant="fp4"),     # rejected: quantization
        ep("slow", 0.06, 0.2, tps=20),                         # rejected: throughput
        ep("pricey", 0.5, 1.0, tps=200),                       # rejected: price
        ep("a", 0.10, 0.3, tps=60),
        ep("b", 0.08, 0.3, tps=90),
    ])
    d = decide(s, req({"messages": [{"role": "user", "content": "hi"}]}), StickyStore())
    assert d.mode == "ranked"
    assert d.order == ["b", "a"]
    assert d.only == ["b", "a"]


def test_tolerance_lets_next_criterion_decide():
    cfg = make_cfg(priority=[{"by": "input_price", "tolerance": 0.2}, "throughput"])
    s = ModelState(cfg, [ep("cheapest", 0.100, 0.3, tps=50), ep("fast", 0.115, 0.3, tps=150),
                         ep("far", 0.5, 0.3, tps=500)])
    d = decide(s, req({"messages": [{"role": "user", "content": "hi"}]}), StickyStore())
    assert d.order == ["fast", "cheapest", "far"]


def test_context_and_tools_are_hard_requirements():
    cfg = make_cfg()
    s = ModelState(cfg, [ep("small", 0.1, 0.1, ctx=1000), ep("notools", 0.1, 0.1, params=("temperature",)),
                         ep("ok", 0.5, 0.5)])
    body = {"messages": [{"role": "user", "content": "x" * 8000}], "tools": [{"type": "function"}]}
    assert decide(s, req(body), StickyStore()).order == ["ok"]


def test_no_match_modes():
    s = ModelState(make_cfg(filters={"max_input_price": 0.01}), [ep("a", 0.1, 0.1), ep("b", 0.05, 0.1)])
    d = decide(s, req({"messages": []}), StickyStore())
    assert d.mode == "best_effort" and d.order[0] == "b"
    s = ModelState(make_cfg(filters={"max_input_price": 0.01}, routing={"on_no_match": "openrouter"}),
                   [ep("a", 0.1, 0.1)])
    assert decide(s, req({"messages": []}), StickyStore()).provider_prefs() is None


def test_sticky_above_context_threshold():
    cfg = make_cfg(priority=["input_price"], routing={"switch_below_context": "1k", "above_context": "sticky"})
    s = ModelState(cfg, [ep("a", 0.10, 0.1), ep("b", 0.20, 0.1)])
    sticky = StickyStore()
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    small = req({"messages": msgs})
    sticky.set(f"m:{small.session_key}", "b", 60)

    # below threshold: switch freely to the best provider
    assert decide(s, small, sticky).order[0] == "a"
    # above threshold: same session keeps its provider
    big = req({"messages": msgs + [{"role": "assistant", "content": "y" * 8000}]})
    assert big.session_key == small.session_key
    d = decide(s, big, sticky)
    assert d.mode == "sticky" and d.order[0] == "b"


def test_zdr_filter():
    cfg = make_cfg(filters={"zdr_only": True})
    a, b = ep("a", 0.1, 0.1), ep("b", 0.05, 0.1)
    a.zdr, b.zdr = True, False
    assert decide(ModelState(cfg, [a, b]), req({"messages": []}), StickyStore()).order == ["a"]


def test_price_overrides():
    e = ep("t", 0.1, 0.2, overrides=[
        {"utc_start": 0, "utc_end": 1400, "prompt": "0.0000002"},
        {"min_prompt_tokens": 200000, "completion": "0.000001"},
    ])
    morning = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    evening = datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)
    assert e.prices_for(1000, morning).input == pytest.approx(0.2)
    assert e.prices_for(1000, evening).input == pytest.approx(0.1)
    assert e.prices_for(300000, evening).output == pytest.approx(1.0)


def test_resolve_openrouter_prefixed_alias():
    cfg = build_config({"openrouter": {"api_key": "k"}, "auth": {"api_keys": ["t"]},
                        "models": [{"name": "deepseek-v4-flash", "model": "deepseek/deepseek-v4-flash-0731",
                                    "filters": {"min_uptime": 90}}]})
    cat = Catalog(cfg, client=None)  # type: ignore[arg-type]
    assert cat.resolve("deepseek-v4-flash") is cat.resolve("openrouter/deepseek-v4-flash") is not None
    assert cat.resolve("openrouter/auto") is None
    assert cat.resolve("openai/gpt-5-nano") is None
