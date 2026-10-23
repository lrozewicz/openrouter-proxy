"""HTTP layer: OpenAI-compatible endpoints proxied to OpenRouter with provider routing injected."""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .catalog import Catalog, ModelState
from .config import Config, load_config
from .router import (Decision, NoProviderError, RequestInfo, StickyStore, analyze_request, decide, evaluate,
                     tag_for_provider)

log = logging.getLogger("orproxy")
logging.getLogger("httpx").setLevel(logging.WARNING)

HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade",
               "proxy-authorization", "proxy-connection", "authorization", "x-api-key", "accept-encoding",
               "x-session-id"}
RESP_DROP = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
MIN_CACHE_PROMPT = 2048  # only requests this big say anything useful about cache behaviour


def _error(status: int, message: str, code: str = "proxy_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": code, "code": status}}, status_code=status)


class Proxy:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.request_timeout, connect=15.0),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=64, keepalive_expiry=60),
        )
        self.catalog = Catalog(cfg, self.client)
        self.sticky = StickyStore()

    # ---- auth -------------------------------------------------------------------------------
    def authorized(self, request: Request) -> bool:
        if not self.cfg.proxy_keys:
            return self.cfg.allow_anonymous
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        return any(hmac.compare_digest(token, k) for k in self.cfg.proxy_keys)

    def upstream_headers(self, request: Request) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        headers["authorization"] = f"Bearer {self.cfg.api_key}"
        if self.cfg.http_referer and "http-referer" not in request.headers:
            headers["http-referer"] = self.cfg.http_referer
        if self.cfg.app_title and "x-title" not in request.headers:
            headers["x-title"] = self.cfg.app_title
        return headers

    # ---- endpoints --------------------------------------------------------------------------
    async def health(self, request: Request) -> Response:
        loaded = all(s.endpoints for s in self.catalog.models.values())
        return JSONResponse({"status": "ok" if loaded else "degraded"})

    async def list_models(self, request: Request) -> Response:
        if not self.authorized(request):
            return _error(401, "invalid API key", "authentication_error")
        now = int(time.time())
        data = [{"id": name, "object": "model", "created": now, "owned_by": "openrouter-proxy",
                 "root": s.cfg.model,
                 "context_length": max((e.context_length or 0 for e in s.endpoints), default=None)}
                for name, s in self.catalog.models.items()]
        return JSONResponse({"object": "list", "data": data})

    async def status(self, request: Request) -> Response:
        if not self.authorized(request):
            return _error(401, "invalid API key", "authentication_error")
        tokens = int(request.query_params.get("tokens", "8000"))
        tools = request.query_params.get("tools", "1") != "0"
        out = {}
        for name, s in self.catalog.models.items():
            req = RequestInfo(tokens, 0, {"tools"} if tools else set(), "status")
            passing, _, all_c = evaluate(s, req)
            order = {c.ep.tag: i + 1 for i, c in enumerate(passing)}
            rows = sorted(all_c, key=lambda c: order.get(c.ep.tag, 10_000))
            out[name] = {
                "model": s.cfg.model,
                "refreshed_at": s.refreshed_at or None,
                "refresh_interval_s": s.cfg.refresh_interval,
                "last_error": s.last_error,
                "assumed_prompt_tokens": tokens,
                "ranking": [c.ep.tag for c in passing],
                "providers": [{
                    "rank": order.get(c.ep.tag),
                    "tag": c.ep.tag,
                    "provider": c.ep.provider_name,
                    "quantization": c.ep.quantization,
                    "rejected": c.rejected,
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.features.items()},
                    "measured": _measured_dict(s, c.ep.provider_name),
                } for c in rows],
            }
        return JSONResponse({"sticky_sessions": len(self.sticky), "models": out})

    async def proxy(self, request: Request) -> Response:
        if not self.authorized(request):
            return _error(401, "invalid API key", "authentication_error")

        path = request.path_params["path"]
        url = f"{self.cfg.base_url}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"
        headers = self.upstream_headers(request)
        raw = await request.body()

        state: ModelState | None = None
        decision: Decision | None = None
        req_info: RequestInfo | None = None
        body: dict[str, Any] | None = None

        if raw and "json" in request.headers.get("content-type", "json"):
            try:
                parsed = json.loads(raw)
                body = parsed if isinstance(parsed, dict) else None
            except ValueError:
                body = None
        if body is not None and isinstance(body.get("model"), str):
            state = self.catalog.resolve(body["model"])
            if state is None and not self.cfg.passthrough_unknown_models:
                return _error(404, f"model {body['model']!r} is not configured on this proxy", "invalid_request_error")
            if state is not None:
                req_info = analyze_request(body, self.cfg.chars_per_token, request.headers.get("x-session-id"))
                try:
                    decision = decide(state, req_info, self.sticky)
                except NoProviderError as exc:
                    return _error(503, str(exc), "no_provider")
                body["model"] = state.cfg.model
                prefs = dict(body.get("provider") or {})
                prefs.update(state.cfg.extra_provider)
                if state.cfg.filters.zdr_only:
                    prefs["zdr"] = True
                if (dp := decision.provider_prefs()) is not None:
                    prefs.update(dp)
                if prefs:
                    body["provider"] = prefs
                body.setdefault("usage", {"include": True})
                raw = json.dumps(body, ensure_ascii=False).encode()

        upstream_req = self.client.build_request(request.method, url, headers=headers, content=raw)
        t0 = time.monotonic()
        try:
            upstream = await self.client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            log.warning("upstream error for %s: %s", path, exc)
            return _error(502, f"upstream error: {type(exc).__name__}: {exc}", "upstream_error")

        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in RESP_DROP}
        if decision is not None:
            resp_headers["x-orproxy-mode"] = decision.mode
            resp_headers["x-orproxy-order"] = ",".join(decision.order) or "openrouter-default"

        if state is None or decision is None or req_info is None:
            return StreamingResponse(_relay(upstream), status_code=upstream.status_code, headers=resp_headers)

        observer = Observation(self, state, decision, req_info, body or {}, t0, upstream.status_code)
        is_sse = "text/event-stream" in upstream.headers.get("content-type", "")
        return StreamingResponse(observer.wrap(upstream, is_sse), status_code=upstream.status_code,
                                 headers=resp_headers)


async def _relay(upstream: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()


def _measured_dict(state: ModelState, provider_name: str) -> dict[str, Any] | None:
    m = state.measured.get(provider_name)
    if not m:
        return None
    r = lambda v: round(v, 4) if v is not None else None  # noqa: E731
    return {"requests": m.requests, "errors": m.errors,
            "cache_hit_rate": r(m.cache_hit_rate.value), "cache_samples": m.cache_hit_rate.samples,
            "throughput": r(m.throughput.value), "ttft_ms": r(m.ttft_ms.value)}


class Observation:
    """Relays the upstream response untouched while watching it for provider, usage and timing."""

    def __init__(self, proxy: Proxy, state: ModelState, decision: Decision, req: RequestInfo,
                 body: dict[str, Any], t0: float, status: int):
        self.proxy, self.state, self.decision, self.req, self.t0, self.status = proxy, state, decision, req, t0, status
        self.is_continuation = any(isinstance(m, dict) and m.get("role") == "assistant"
                                   for m in body.get("messages") or [])
        self.provider: str | None = None
        self.usage: dict[str, Any] | None = None
        self.first_token: float | None = None
        self.error: str | None = None

    async def wrap(self, upstream: httpx.Response, is_sse: bool) -> AsyncIterator[bytes]:
        buf = b""
        whole: list[bytes] = []
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
                if is_sse:
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        self._sse_line(line)
                elif len(whole) < 4096:
                    whole.append(chunk)
            if not is_sse:
                self._json_body(b"".join(whole))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await upstream.aclose()
            self._finish()

    def _sse_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        if self.first_token is None:
            self.first_token = time.monotonic()
        if self.provider is not None and b'"usage"' not in payload and b'"error"' not in payload:
            return
        try:
            data = json.loads(payload)
        except ValueError:
            return
        self._absorb(data)

    def _json_body(self, raw: bytes) -> None:
        try:
            data = json.loads(raw)
        except ValueError:
            return
        if isinstance(data, dict):
            self._absorb(data)

    def _absorb(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        if data.get("provider"):
            self.provider = data["provider"]
        if data.get("usage"):
            self.usage = data["usage"]
        if data.get("error"):
            err = data["error"]
            self.error = err.get("message") if isinstance(err, dict) else str(err)

    def _finish(self) -> None:
        end = time.monotonic()
        usage = self.usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        ok = self.status < 400 and self.error is None

        tps = None
        if self.provider:
            m = self.state.measured_for(self.provider)
            m.requests += 1
            if not ok:
                m.errors += 1
            if ok and self.first_token is not None:
                m.ttft_ms.add((self.first_token - self.t0) * 1000)
                gen_time = end - self.first_token
                if completion >= 20 and gen_time > 0.05:
                    tps = completion / gen_time
                    m.throughput.add(tps)
            if ok and self.is_continuation and prompt >= MIN_CACHE_PROMPT:
                m.cache_hit_rate.add(cached / prompt)
            tag = tag_for_provider(self.decision, self.provider)
            if ok and tag:
                cfg = self.state.cfg
                self.proxy.sticky.set(f"{cfg.name}:{self.req.session_key}", tag, cfg.routing.sticky_ttl)

        log.info("%s ~%dtok mode=%s order=%s -> %s status=%s prompt=%d cached=%d out=%d%s%s",
                 self.state.cfg.name, self.req.prompt_tokens, self.decision.mode,
                 ",".join(self.decision.order[:3]) or "default", self.provider or "?", self.status,
                 prompt, cached, completion, f" tps={tps:.0f}" if tps else "",
                 f" error={self.error}" if self.error else "")


def create_app(cfg: Config | None = None) -> Starlette:
    cfg = cfg or load_config(os.environ.get("ORPROXY_CONFIG", "config.yaml"))
    proxy = Proxy(cfg)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await proxy.catalog.start()
        try:
            yield
        finally:
            await proxy.catalog.stop()
            await proxy.client.aclose()

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    routes = [
        Route("/health", proxy.health),
        Route("/status", proxy.status),
        Route("/v1/models", proxy.list_models),
        Route("/v1/{path:path}", proxy.proxy, methods=methods),
        Route("/api/v1/models", proxy.list_models),
        Route("/api/v1/{path:path}", proxy.proxy, methods=methods),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.proxy = proxy
    return app
