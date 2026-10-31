"""HTTP-level checks of the third review: duplicate history after a restart, the admin queue with an author, local statuses."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "http.db"))
    monkeypatch.setenv("PERSIST_TRIAGE", "0")
    monkeypatch.setenv("DEDUP_PERSIST", "1")

    def make() -> TestClient:
        import app.main as m

        importlib.reload(m)  # a new "process": new settings, new service, same database file
        return TestClient(m.app)

    return make


BODY = {"kind": "complaint", "reporter_id": "player-77", "reported_nick": "Max_Orlov", "created_at": "2026-10-30T10:00:00",
        "messages": [{"role": "player", "text": "Max_Orlov оскорбил меня у банка", "attachments": 1}]}


def test_duplicate_history_survives_a_restart_over_http(make_client):
    with make_client() as c:
        first = c.post("/triage", json=BODY).json()
        again = c.post("/triage", json={**BODY, "created_at": "2026-10-30T10:05:00"}).json()
        assert first["is_spam_or_duplicate"] is False and again["is_spam_or_duplicate"] is True
        assert c.get("/health").json()["dedup_history_entries"] == 2
    with make_client() as c:  # restart
        assert c.get("/health").json()["dedup_history_entries"] == 2
        third = c.post("/triage", json={**BODY, "created_at": "2026-10-30T10:10:00"}).json()
        assert third["is_spam_or_duplicate"] is True and any("Дубликат" in x for x in third["reasons"])


def test_without_author_nothing_is_compared_over_http(make_client):
    body = {"kind": "complaint", "messages": [{"role": "player", "text": "Max_Orlov оскорбил меня у банка", "attachments": 1}]}
    with make_client() as c:
        assert c.post("/triage", json=body).json()["is_spam_or_duplicate"] is False
        assert c.post("/triage", json=body).json()["is_spam_or_duplicate"] is False


def test_queue_ticket_with_author_gets_a_duplicate_check_and_local_statuses(make_client):
    with make_client() as c:
        a = c.post("/api/tickets", json={**BODY, "created_at": None, "source": "admin-ui"}).json()
        b = c.post("/api/tickets", json={**BODY, "created_at": None, "source": "admin-ui"}).json()
        assert a["response"]["is_spam_or_duplicate"] is False and b["response"]["is_spam_or_duplicate"] is True  # acceptance time was used
        draft = b["response"]["draft_reply"]["text"]
        saved = c.post(f"/api/tickets/{b['id']}/save", json={"text": draft + " Дополнение администратора."}).json()
        assert saved["status"] == "edited"
        sent = c.post(f"/api/tickets/{b['id']}/approve", json={"text": saved["final_text"]}).json()
        assert sent["status"] == "approved_local"  # local only: the UI says nothing is delivered to the player
        assert c.post(f"/api/tickets/{b['id']}/escalate", json={}).json()["status"] == "escalated"
        edits = c.get("/api/edits").json()
        assert len(edits) == 1 and edits[0]["original_text"] == draft and "Дополнение" in edits[0]["edited_text"]


def test_sensitive_ticket_in_the_queue_is_p0_and_needs_a_human(make_client):
    with make_client() as c:
        r = c.post("/api/tickets", json={"kind": "report", "messages": [{"role": "player", "text": "Верните 500 рублей за покупку. Можно ли подать жалобу?", "attachments": 0}]}).json()
        assert r["response"]["priority"] == "p0" and r["response"]["needs_human"] is True
        assert c.get("/api/queue", params={"priority": "p0"}).json()["total"] == 1
