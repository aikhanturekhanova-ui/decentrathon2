"""FastAPI app: POST /triage (contract), admin queue API and the admin web panel."""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Request, Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .config import ROOT, load_settings
from .dedup import DedupIndex
from .pipeline import TriageService
from . import metrics
from .schemas import Kind, Message, TriageRequest, TriageResponse
from .security import RateLimiter, access_middleware, load_tokens, require
from .store import Store

WEB_DIR = Path(__file__).parent / "web"
DEMO_FILE = ROOT / "tests" / "fixtures" / "demo_tickets.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    # the duplicate history lives in the same SQLite file as the queue, so a restart does not forget it
    dedup = DedupIndex(settings.db_path) if settings.dedup_persist else DedupIndex()
    app.state.service = TriageService(settings, dedup=dedup)
    app.state.store = Store(settings.db_path)
    # an invalid API_TOKENS stops the start-up (AuthConfigError): a broken security setting never means "authentication off"
    app.state.api_tokens = load_tokens(settings.api_tokens_raw)
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_min)
    if not app.state.api_tokens:
        print("[WARNING] API_TOKENS is not set: authentication is OFF (explicit demo mode). Set it before exposing the service.", file=sys.stderr)
    for w in settings.config_warnings:
        print(f"[config warning] {w}", file=sys.stderr)
    yield
    if app.state.service.llm is not None:
        app.state.service.llm.close()


class UTF8JSONResponse(JSONResponse):
    """JSON with an explicit charset: Windows clients (PowerShell 5.1, some HTTP libraries) otherwise decode Cyrillic as Latin-1."""

    media_type = "application/json; charset=utf-8"


app = FastAPI(
    default_response_class=UTF8JSONResponse,
    title="Grand Mobile Support Copilot",
    version="0.1.0",
    description="Triage service: language, category, priority, extraction, rule-grounded draft reply, human-handoff.",
    lifespan=lifespan,
)
app.middleware("http")(access_middleware)
VIEWER, ADMIN, SERVICE = Depends(require("viewer")), Depends(require("admin")), Depends(require("service"))


def _service() -> TriageService:
    return app.state.service


def _store() -> Store:
    return app.state.store


@app.get("/health")
def health() -> dict:
    svc = _service()
    return {
        "status": "ok",
        "version": app.version,
        "auth": "on" if app.state.api_tokens else "off (demo mode: set API_TOKENS)",
        "rate_limit_per_min": app.state.settings.rate_limit_per_min,
        "mode": svc.mode,
        "classifier": svc.classifier,
        "model_meta": {k: v for k, v in (svc.model.meta if svc.model else {}).items() if k in {"trained_at", "n_train", "classes", "blend_weight"}},
        "llm_enabled": svc.llm is not None,
        "rules_points": len(svc.rules.docs),
        "taxonomy_source": svc.taxonomy.source,
        "persist_triage": app.state.settings.persist_triage,
        "faq_entries": len(svc.faq.entries),
        "faq_problems": svc.faq.problems,
        "dedup_history_entries": svc.dedup.size(),
        "dedup_persistent": app.state.settings.dedup_persist,
        "config_warnings": list(app.state.settings.config_warnings),
    }


@app.post("/triage", response_model=TriageResponse, dependencies=[SERVICE])
def triage(req: TriageRequest, response: Response) -> TriageResponse:
    """The contract endpoint. Body: {kind, messages:[{role,text,attachments}]} (+ optional context)."""
    resp, dbg = _service().triage(req)
    metrics.record_triage(resp, dbg)
    response.headers["X-Triage-Mode"] = dbg.mode
    response.headers["X-Latency-Ms"] = str(dbg.latency_ms)
    if dbg.cost_usd is not None:
        response.headers["X-Cost-Usd"] = f"{dbg.cost_usd:.6f}"
    if app.state.settings.persist_triage:
        _store().add(req.model_dump(), resp.model_dump(), source="api", external_id=req.ticket_id)
    return resp


@app.post("/triage/debug", dependencies=[SERVICE])
def triage_debug(req: TriageRequest) -> dict:
    """Same pipeline as /triage plus diagnostics (signals, retrieved rules, timings, cost)."""
    resp, dbg = _service().triage(req)
    return {"response": resp.model_dump(), "debug": dbg.model_dump()}


class BatchBody(BaseModel):
    # validated ONE BY ONE below: a single malformed ticket (a numeric created_at, a missing kind) gets its own error entry instead of
    # rejecting the 199 valid tickets next to it
    tickets: list[Any] = Field(min_length=1, max_length=200)


@app.post("/triage/batch", dependencies=[SERVICE])
def triage_batch(body: BatchBody, request: Request) -> dict:
    """Up to 200 tickets per call for backlogs. Processed in time order (as `make predict`), so duplicates are found whatever the order
    in the body; results come back in the order of the request. An invalid or failing ticket gets an error entry (never a fake answer),
    the others are unaffected. The rate limit counts TICKETS here, not HTTP requests (one call of 200 = 200 units)."""
    from pydantic import ValidationError

    from .dedup import parse_ts
    from .security import limit_key

    retry = request.app.state.rate_limiter.hit(limit_key(request), weight=max(len(body.tickets) - 1, 0))
    if retry is not None:
        raise HTTPException(429, "rate limit exceeded (a batch counts each ticket)", headers={"Retry-After": str(retry)})
    tickets: list[Optional[TriageRequest]] = []
    errors: dict[int, str] = {}
    for i, raw in enumerate(body.tickets):
        try:
            tickets.append(TriageRequest.model_validate(raw))
        except ValidationError as exc:
            tickets.append(None)
            errors[i] = "invalid ticket: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
    order = sorted((i for i, t in enumerate(tickets) if t is not None),
                   key=lambda i: (parse_ts(tickets[i].created_at) or parse_ts("1970-01-01T00:00:00"), i))
    results: list[Optional[dict]] = [None] * len(body.tickets)
    for i, why in errors.items():
        raw = body.tickets[i]
        results[i] = {"ticket_id": raw.get("ticket_id") if isinstance(raw, dict) else None, "error": why, "needs_human": True}
    for i in order:
        t = tickets[i]
        try:
            resp, dbg = _service().triage(t)
            metrics.record_triage(resp, dbg)
            results[i] = {"ticket_id": t.ticket_id, "response": resp.model_dump()}
        except Exception as exc:  # noqa: BLE001 - one bad ticket must not fail the batch; the caller sends it to a human
            results[i] = {"ticket_id": t.ticket_id, "error": f"{type(exc).__name__}: {str(exc)[:200]}", "needs_human": True}
    return {"results": results}


@app.get("/metrics", dependencies=[VIEWER])
def prometheus_metrics() -> PlainTextResponse:
    """Prometheus text format: request counts, triage latency histogram, needs_human share, categories, priorities, languages."""
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


# ------------------------------------------------------------------------------ administrators' decisions (human labels)
class FeedbackBody(BaseModel):
    """The administrator's final decision on a ticket, sent by the game's admin tool (or this panel)."""

    ticket_id: str = Field(min_length=1, max_length=200)
    kind: Optional[Kind] = None
    messages: Optional[list[Message]] = Field(default=None, max_length=50)  # needed only if this service has not stored the ticket
    predicted_category: Optional[str] = None  # what the copilot returned at triage time (looked up or recomputed when absent)
    predicted_priority: Optional[str] = None
    final_category: str
    final_priority: str
    final_reply: Optional[str] = Field(default=None, max_length=4000)
    draft_rating: Optional[int] = Field(default=None, ge=1, le=5)  # how useful the draft was, 1..5 (the case's own scale)


def _record_decision(body: FeedbackBody, source: str) -> dict:
    svc = _service()
    if body.final_category not in svc.taxonomy.categories:
        raise HTTPException(422, f"final_category {body.final_category!r} is not in the taxonomy")
    if body.final_priority not in svc.taxonomy.priorities:
        raise HTTPException(422, f"final_priority {body.final_priority!r} is not in the taxonomy")
    stored = _store().find_by_external_id(body.ticket_id)
    if body.messages:
        req = TriageRequest(kind=body.kind or (stored["kind"] if stored else body.final_category.split(".")[0]), messages=body.messages)
    elif stored:
        req = TriageRequest.model_validate(stored["request"])
    else:
        raise HTTPException(422, "unknown ticket_id: send `kind` and `messages` with the decision")
    pred_cat, pred_pri = body.predicted_category, body.predicted_priority
    if stored and not pred_cat:
        pred_cat, pred_pri = stored["response"]["category"], stored["response"]["priority"]
    if not pred_cat:  # recompute without side effects (no duplicate history is touched)
        from .classify import analyze

        a = analyze(req, svc.model)
        pred_cat, pred_pri, _ = svc.taxonomy.fit(a.category, a.priority, a.kind)
    from .classify import player_text
    from .textutil import detect_lang

    text = player_text(req)
    _store().add_decision({"ticket_ref": body.ticket_id, "kind": req.kind, "player_text": text, "predicted_category": pred_cat,
                           "predicted_priority": pred_pri, "final_category": body.final_category, "final_priority": body.final_priority,
                           "final_reply": body.final_reply, "draft_rating": body.draft_rating, "lang": detect_lang(text, req.server_lang),
                           "source": source})
    return {"stored": True, "ticket_id": body.ticket_id, "predicted_category": pred_cat, "predicted_priority": pred_pri,
            "category_correct": pred_cat == body.final_category, "priority_correct": pred_pri == body.final_priority}


@app.post("/api/feedback", dependencies=[SERVICE])
def feedback(body: FeedbackBody) -> dict:
    """Store the administrator's final category/priority (and optionally the reply sent and a 1–5 rating of the draft).
    These human decisions measure the real accuracy (`GET /api/feedback/summary`) and are training data for `make train`."""
    return _record_decision(body, "api")


@app.get("/api/feedback/summary", dependencies=[VIEWER])
def feedback_summary() -> dict:
    """Accuracy of the copilot against the administrators' own decisions (the only trustworthy quality measure in production)."""
    return _store().decisions_summary()


class DecisionBody(BaseModel):
    final_category: str
    final_priority: str
    draft_rating: Optional[int] = Field(default=None, ge=1, le=5)


@app.post("/api/tickets/{ticket_id}/decision", dependencies=[ADMIN])
def ticket_decision(ticket_id: int, body: DecisionBody) -> dict:
    """The admin panel's «Решение администратора»: confirm or correct category and priority, rate the draft."""
    t = _store().get(ticket_id)
    if t is None:
        raise HTTPException(404, "ticket not found")
    ref = t["external_id"] or f"local:{ticket_id}"
    fb = FeedbackBody(ticket_id=ref, kind=t["kind"], messages=t["request"]["messages"], predicted_category=t["response"]["category"],
                      predicted_priority=t["response"]["priority"], final_category=body.final_category, final_priority=body.final_priority,
                      final_reply=t.get("final_text"), draft_rating=body.draft_rating)
    return _record_decision(fb, "admin-ui")


# ------------------------------------------------------------------------------ admin queue API
class SubmitBody(TriageRequest):
    source: str = "admin-ui"


class SaveBody(BaseModel):
    text: Optional[str] = Field(default=None, max_length=4000)
    note: Optional[str] = Field(default=None, max_length=1000)


@app.get("/api/queue", dependencies=[VIEWER])
def queue(category: str = "", lang: str = "", priority: str = "", needs_human: str = "", status: str = "", q: str = "",
          limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> dict:
    return _store().list(category, lang, priority, needs_human, status, q, limit, offset)


@app.post("/api/tickets", dependencies=[ADMIN])
def submit(body: SubmitBody) -> dict:
    req = TriageRequest.model_validate(body.model_dump(exclude={"source"}))
    if req.reporter_id and not req.created_at:
        # an author without a time cannot be compared inside the 60-minute window: use the moment of acceptance and say so
        req.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    resp, dbg = _service().triage(req)
    metrics.record_triage(resp, dbg)
    tid = _store().add(req.model_dump(), resp.model_dump(), source=body.source, external_id=req.ticket_id)
    return {"id": tid, "response": resp.model_dump(), "latency_ms": dbg.latency_ms, "mode": dbg.mode}


@app.get("/api/tickets/{ticket_id}", dependencies=[VIEWER])
def ticket(ticket_id: int) -> dict:
    t = _store().get(ticket_id)
    if t is None:
        raise HTTPException(404, "ticket not found")
    rb = _service().rules
    t["rules_detail"] = [
        {"cite": c, "section": d.section_title, "ref": d.ref, "text": d.body}
        for c in t["response"]["draft_reply"]["cited"]
        if (d := rb.resolve_cite(c)) is not None
    ]
    t["terms_detail"] = [
        {"abbr": x.abbr, "name": x.name, "definition": x.definition, "section": x.section_title}
        for abbr in t["response"]["draft_reply"].get("terms", []) if (x := rb.terms.get(abbr)) is not None
    ]
    t["moderation"] = moderation_view(t["request"]["messages"], t["request"].get("server_lang"))
    return t


def moderation_view(messages: list, server_lang: Optional[str] = None) -> dict:
    """Moderation panel data: per-message language segments and abusive-language findings with spans in the original text."""
    from .langseg import profile
    from .toxicity import analyze_messages

    tox = analyze_messages(messages)
    per_msg = []
    for i, m in enumerate(messages):
        role, text = (m.get("role"), m.get("text")) if isinstance(m, dict) else (m.role, m.text)
        if role == "admin" or not text:
            continue
        per_msg.append({"index": i, "role": role, "text": text, "languages": profile(text, server_lang).as_dict(),
                        "findings": [f.__dict__ for f in tox.findings if f.message == i]})
    summary = tox.as_dict()
    summary.pop("findings")
    return {"summary": summary, "messages": per_msg}


class ModerationBody(BaseModel):
    text: str = Field(max_length=4000)
    role: str = "player"
    server_lang: Optional[str] = None


@app.post("/api/moderation/analyze", dependencies=[VIEWER])
def moderation_analyze(body: ModerationBody) -> dict:
    """Analyse any text (the «Модерация» tool of the admin panel): languages by segment and abusive language with evidence spans."""
    return moderation_view([{"role": body.role, "text": body.text}], body.server_lang)


def _act(ticket_id: int, status: str, body: SaveBody) -> dict:
    t = _store().update_status(ticket_id, status, body.text, body.note)
    if t is None:
        raise HTTPException(404, "ticket not found")
    return t


@app.post("/api/tickets/{ticket_id}/save", dependencies=[ADMIN])
def save_edit(ticket_id: int, body: SaveBody) -> dict:
    """Save an edited draft (status 'edited'); a changed text is stored as feedback."""
    return _act(ticket_id, "edited", body)


@app.post("/api/tickets/{ticket_id}/approve", dependencies=[ADMIN])
def approve_local(ticket_id: int, body: SaveBody) -> dict:
    """Confirm locally. The game is not connected, so NOTHING is sent to the player."""
    return _act(ticket_id, "approved_local", body)


@app.post("/api/tickets/{ticket_id}/escalate", dependencies=[ADMIN])
def escalate(ticket_id: int, body: SaveBody) -> dict:
    return _act(ticket_id, "escalated", body)


@app.get("/api/stats", dependencies=[VIEWER])
def stats() -> dict:
    return _store().stats()


HISTORY_FILE = ROOT / "docs" / "metrics" / "history.json"


@app.get("/api/insights", dependencies=[VIEWER])
def insights() -> dict:
    """Dashboard data: the live queue + aggregated historical patterns from TRAIN (scripts/history_insights.py)."""
    hist = None
    if HISTORY_FILE.is_file():
        try:
            hist = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except ValueError:
            hist = None
    return {"queue": _store().stats(), "history": hist, "classifier": _service().classifier}


@app.post("/api/demo/seed", dependencies=[ADMIN])
def seed_demo() -> dict:
    """Load the artificial demo tickets from tests/fixtures (no real player data)."""
    if not DEMO_FILE.is_file():
        raise HTTPException(404, "demo file missing")
    n = 0
    for line in DEMO_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        req = TriageRequest.model_validate(json.loads(line))
        resp, _ = _service().triage(req)
        _store().add(req.model_dump(), resp.model_dump(), source="demo", external_id=req.ticket_id)
        n += 1
    return {"added": n}


@app.get("/api/template-suggestions", dependencies=[VIEWER])
def template_suggestions(min_size: int = Query(3, ge=2, le=100)) -> list[dict]:
    """Bonus: reply-template proposals from clusters of similar tickets (for human review, never auto-applied)."""
    from .clusters import suggest

    return suggest(_store().all_full(), min_size=min_size)


@app.get("/api/edits", dependencies=[ADMIN])
def edits() -> JSONResponse:
    return JSONResponse(_store().export_edits())


@app.get("/")
@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
