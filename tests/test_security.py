"""Access control (roles), rate limiting, request ids and configurable thresholds."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module  # resolved at call time: tests/test_review3_http.py reloads app.main
from app.security import parse_tokens

ADMIN, VIEWER, SERVICE = "admin-token-0123456789", "viewer-token-0123456789", "service-token-0123456789"
BODY = {"kind": "complaint", "messages": [{"role": "player", "text": "Arman_Petrov оскорбил меня", "attachments": 1}]}


@pytest.fixture()
def secured(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("API_TOKENS", f"admin:{ADMIN},viewer:{VIEWER},service:{SERVICE}")
    with TestClient(main_module.app) as c:
        yield c


def h(token: str) -> dict:
    return {"authorization": f"Bearer {token}"}


def test_demo_mode_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "d.db"))
    monkeypatch.delenv("API_TOKENS", raising=False)
    with TestClient(main_module.app) as c:
        assert c.get("/health").json()["auth"].startswith("off")
        assert c.get("/api/queue").status_code == 200


def test_roles(secured):
    c = secured
    assert c.get("/health").json()["auth"] == "on"
    assert c.get("/api/queue").status_code == 401
    assert c.get("/api/queue", headers=h("wrong-token-0123456789")).status_code == 401
    assert c.get("/api/queue", headers=h(VIEWER)).status_code == 200
    assert c.post("/api/demo/seed", headers=h(VIEWER)).status_code == 403
    assert c.post("/api/demo/seed", headers=h(ADMIN)).status_code == 200
    assert c.post("/triage", json=BODY).status_code == 401
    assert c.post("/triage", json=BODY, headers=h(SERVICE)).status_code == 200
    assert c.post("/triage", json=BODY, headers={"x-api-key": ADMIN}).status_code == 200
    assert c.get("/api/queue", headers=h(SERVICE)).status_code == 403  # the game backend cannot read the admin queue
    assert c.get("/api/edits", headers=h(VIEWER)).status_code == 403
    assert c.get("/admin").status_code == 200  # the static page holds no data


def test_token_parsing_rejects_bad_entries():
    tokens, problems = parse_tokens("admin:short,root:" + "x" * 20 + ",viewer:" + "v" * 20)
    assert tokens == {"v" * 20: "viewer"} and len(problems) == 2


def test_rate_limit_and_request_id(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.delenv("API_TOKENS", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    with TestClient(main_module.app) as c:
        codes = [c.post("/triage", json=BODY).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and 429 in codes[3:]
        r = c.post("/triage", json=BODY)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert c.get("/health").status_code == 200  # health is never limited
        assert c.get("/health", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123"


def test_confidence_thresholds_are_configurable(rulebook, tmp_path, monkeypatch):
    from app.config import load_settings
    from app.pipeline import TriageService
    from tests.conftest import make_req

    req = make_req("что-то непонятное случилось со мной вчера", kind="report", att=0)
    monkeypatch.setenv("HUMAN_CONF_MIN", "0.0")
    monkeypatch.setenv("HUMAN_AMBIGUOUS_MIN", "0.0")
    relaxed = TriageService(load_settings(), rules=rulebook).triage(req)[0]
    monkeypatch.setenv("HUMAN_CONF_MIN", "0.99")
    strict = TriageService(load_settings(), rules=rulebook).triage(req)[0]
    assert strict.needs_human is True
    assert not any("уверенность" in x.lower() for x in relaxed.reasons)
