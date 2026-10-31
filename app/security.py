"""Access control, rate limiting and structured access logs for the HTTP service.

Roles (from API_TOKENS="admin:<token>,viewer:<token>,service:<token>", several tokens per role allowed):
  * service — the game backend (Node.js) calling POST /triage and /triage/debug;
  * viewer  — reads the queue, tickets, statistics, the dashboard and the moderation tool;
  * admin   — everything: edits, local send, escalation, ticket submission, demo data, export of edits, and /triage.
Tokens are sent as `Authorization: Bearer <token>` (or `X-API-Key`).

Modes, deliberately strict:
  * API_TOKENS unset or empty  -> explicit DEMO mode: authentication OFF, said by /health and at startup (a local demo needs no setup);
  * API_TOKENS set and VALID   -> every protected route needs a token of the right role;
  * API_TOKENS set but INVALID (an unknown role, a token shorter than 16 characters, an empty list after parsing) -> the service REFUSES
    TO START (`AuthConfigError`). A typo in the security configuration must never silently turn authentication off. Token values are
    never printed: only the role and the problem.

The rate limiter is in-process (one counter per token or client IP, fixed one-minute windows). Several instances behind a load balancer
need the limit at the gateway instead; this is stated in docs/INTEGRATION.md.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
import threading
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request

ROLE_GRANTS = {"admin": {"admin", "viewer", "service"}, "viewer": {"viewer"}, "service": {"service"}}

log = logging.getLogger("copilot.access")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def parse_tokens(raw: str) -> tuple[dict[str, str], list[str]]:
    """'admin:abc,viewer:def' -> ({token: role}, problems). Unknown roles and short tokens are rejected (and reported)."""
    tokens: dict[str, str] = {}
    problems: list[str] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        role, sep, token = part.partition(":")
        role, token = role.strip(), token.strip()
        if not sep or role not in ROLE_GRANTS:
            problems.append(f"API_TOKENS entry with unknown role {role!r} ignored (use admin|viewer|service)")
        elif len(token) < 16:
            problems.append(f"API_TOKENS: the {role} token is shorter than 16 characters and was ignored")
        else:
            tokens[token] = role
    return tokens, problems


class AuthConfigError(RuntimeError):
    pass


def load_tokens(raw: str) -> dict[str, str]:
    """{} for the explicit demo mode (nothing configured); the token map when the configuration is valid; AuthConfigError otherwise."""
    if not (raw or "").strip():
        return {}
    tokens, problems = parse_tokens(raw)
    if problems or not tokens:
        raise AuthConfigError("API_TOKENS is set but invalid, the service will not start with authentication off: "
                              + "; ".join(problems or ["no usable token"]) + ". Fix API_TOKENS or remove it for the demo mode.")
    return tokens


def _presented_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def role_of(request: Request) -> Optional[str]:
    tokens: dict[str, str] = request.app.state.api_tokens
    if not tokens:
        return "admin"  # demo mode: authentication is off (and /health says so)
    presented = _presented_token(request)
    if not presented:
        return None
    for token, role in tokens.items():  # constant-time comparison against every configured token
        if hmac.compare_digest(token.encode(), presented.encode()):
            return role
    return None


def require(needed: str):
    """FastAPI dependency: the caller's role must grant `needed`."""

    def dep(request: Request) -> str:
        role = role_of(request)
        if role is None:
            raise HTTPException(401, "missing or invalid API token", headers={"WWW-Authenticate": "Bearer"})
        if needed not in ROLE_GRANTS[role]:
            raise HTTPException(403, f"role {role!r} may not do this (needs {needed!r})")
        request.state.role = role
        return role

    return dep


def limit_key(request: Request) -> str:
    """WHO is rate-limited. Only a VERIFIED token identifies a client (hashed, never the token itself); an unknown, invalid or - in demo
    mode - any presented token does not: then the client's IP is the key, so inventing a new `Authorization` header per request does not
    reset the limit."""
    tokens: dict[str, str] = request.app.state.api_tokens
    presented = _presented_token(request)
    if tokens and presented:
        for token in tokens:
            if hmac.compare_digest(token.encode(), presented.encode()):
                return "tok:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    return "ip:" + (request.client.host if request.client else "unknown")


class RateLimiter:
    """Fixed one-minute windows per key. Memory is bounded: at most MAX_KEYS distinct keys per window; further keys share one overflow
    bucket (so a flood of new IPs is limited together instead of growing the table)."""

    MAX_KEYS = 10_000

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._window = 0
        self._counts: dict[str, int] = {}

    def hit(self, key: str, weight: int = 1) -> Optional[int]:
        """Add `weight` units (1 per HTTP request; a batch adds one per extra ticket). None if allowed, otherwise seconds to the next window."""
        if self.per_minute <= 0:
            return None
        now = time.time()
        window = int(now // 60)
        with self._lock:
            if window != self._window:
                self._window, self._counts = window, {}
            if key not in self._counts and len(self._counts) >= self.MAX_KEYS:
                key = "overflow"
            n = self._counts.get(key, 0) + max(weight, 0)
            self._counts[key] = n
        return None if n <= self.per_minute else int(60 - now % 60) + 1

    def size(self) -> int:
        with self._lock:
            return len(self._counts)


async def access_middleware(request: Request, call_next):
    """Request id, rate limit for API routes, one JSON access-log line per request (never the ticket text)."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    t0 = time.perf_counter()
    path = request.url.path
    limited = path.startswith("/triage") or path.startswith("/api/")
    if limited:
        key = limit_key(request)
        retry = request.app.state.rate_limiter.hit(key)
        if retry is not None:
            from fastapi.responses import JSONResponse

            resp = JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": str(retry), "X-Request-ID": rid})
            _log(request, rid, 429, t0)
            return resp
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    from . import metrics

    route = request.scope.get("route")
    metrics.record_request(getattr(route, "path", "unmatched"), response.status_code)  # the template, not the raw path (no id explosion)
    if path != "/health":
        _log(request, rid, response.status_code, t0)
    return response


def _log(request: Request, rid: str, status: int, t0: float) -> None:
    log.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": rid, "method": request.method, "path": request.url.path,
                         "status": status, "ms": round((time.perf_counter() - t0) * 1000, 2),
                         "role": getattr(request.state, "role", None)}, ensure_ascii=False))
