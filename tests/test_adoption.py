"""Integration features for the game backend: administrators' decisions (human labels) and live accuracy, batch triage, Prometheus metrics."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module  # resolved at call time: tests/test_review3_http.py reloads app.main

T1 = {"kind": "complaint", "ticket_id": "g-1", "messages": [{"role": "player", "text": "Arman_Petrov оскорбил мою семью в чате", "attachments": 2}]}
T2 = {"kind": "report", "ticket_id": "g-2", "messages": [{"role": "player", "text": "оплатил донат, коины не пришли", "attachments": 0}]}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.delenv("API_TOKENS", raising=False)
    with TestClient(main_module.app) as c:
        yield c


def test_feedback_measures_real_accuracy(client):
    assert client.get("/api/feedback/summary").json() == {"decisions": 0, "with_prediction": 0}
    # the game's admin tool reports the final decisions; the copilot looks up / recomputes what it predicted
    r1 = client.post("/api/feedback", json={"ticket_id": "g-1", "kind": "complaint", "messages": T1["messages"],
                                            "final_category": "complaint.insult", "final_priority": "p2", "draft_rating": 4}).json()
    assert r1["category_correct"] and r1["priority_correct"]
    r2 = client.post("/api/feedback", json={"ticket_id": "g-2", "kind": "report", "messages": T2["messages"], "predicted_category": "report.payment",
                                            "predicted_priority": "p0", "final_category": "report.account", "final_priority": "p0"}).json()
    assert not r2["category_correct"]
    s = client.get("/api/feedback/summary").json()
    assert s["decisions"] == 2 and s["category_accuracy"] == 0.5 and s["priority_accuracy"] == 1.0
    assert s["top_category_corrections"] == [["report.payment -> report.account", 1]] and s["draft_rating_mean"] == 4.0
    # the latest decision for a ticket wins (no double counting)
    client.post("/api/feedback", json={"ticket_id": "g-2", "kind": "report", "messages": T2["messages"], "final_category": "report.payment", "final_priority": "p0"})
    assert client.get("/api/feedback/summary").json()["category_accuracy"] == 1.0


def test_feedback_validation(client):
    bad = {"ticket_id": "x", "kind": "complaint", "messages": T1["messages"], "final_priority": "p2"}
    assert client.post("/api/feedback", json={**bad, "final_category": "complaint.unknown"}).status_code == 422
    assert client.post("/api/feedback", json={**bad, "final_category": "complaint.insult", "final_priority": "p9"}).status_code == 422
    assert client.post("/api/feedback", json={"ticket_id": "never-seen", "final_category": "complaint.insult", "final_priority": "p2"}).status_code == 422
    assert client.post("/api/feedback", json={**bad, "final_category": "complaint.insult", "draft_rating": 7}).status_code == 422


def test_decision_from_the_admin_panel(client):
    tid = client.post("/api/tickets", json={**T1, "source": "test"}).json()["id"]
    r = client.post(f"/api/tickets/{tid}/decision", json={"final_category": "complaint.dm", "final_priority": "p2", "draft_rating": 3}).json()
    assert r["category_correct"] is False and r["ticket_id"] == "g-1"
    assert client.get("/api/feedback/summary").json()["with_prediction"] == 1


def test_batch_keeps_order_and_finds_duplicates_regardless_of_order(client):
    a = {"kind": "complaint", "ticket_id": "b", "reporter_id": "r1", "reported_id": "t1", "created_at": "2026-10-01T10:10:00",
         "messages": [{"role": "player", "text": "Arman_Petrov оскорбил меня в чате", "attachments": 1}]}
    b = {**a, "ticket_id": "a", "created_at": "2026-10-01T10:00:00"}
    res = client.post("/triage/batch", json={"tickets": [a, b]}).json()["results"]
    assert [x["ticket_id"] for x in res] == ["b", "a"]  # request order
    assert res[0]["response"]["is_spam_or_duplicate"] is True and res[1]["response"]["is_spam_or_duplicate"] is False  # time order decides
    assert client.post("/triage/batch", json={"tickets": []}).status_code == 422


def test_prometheus_metrics(client):
    client.post("/triage", json=T2)
    text = client.get("/metrics").text
    assert 'copilot_triage_total{dimension="category",value="report.payment"} 1' in text
    assert 'copilot_triage_total{dimension="needs_human",value="true"}' in text
    assert "copilot_triage_latency_ms_count" in text and 'route="/triage",status="200"' in text


def test_training_uses_admin_decisions(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import sys

    sys.path.insert(0, "scripts")
    from app.config import load_settings
    from app.store import Store
    from train_classifier import GOLD_WEIGHT, gold_rows

    monkeypatch.setenv("DB_PATH", str(tmp_path / "g.db"))
    st = Store(load_settings().db_path)
    st.add_decision({"ticket_ref": "1", "kind": "complaint", "player_text": "он оскорбил меня", "final_category": "complaint.insult", "final_priority": "p2"})
    st.add_decision({"ticket_ref": "2", "kind": "report", "player_text": "помогите", "final_category": "report.help", "final_priority": "p2"})
    st.add_decision({"ticket_ref": "3", "kind": "complaint", "player_text": "жб", "final_category": "complaint.invalid", "final_priority": "p3"})
    rows = gold_rows(load_settings(), {"complaint.insult", "complaint.dm"})
    assert [y for _, _, y in rows] == ["complaint.insult"] and GOLD_WEIGHT >= 1  # reports and classes the model lacks are skipped
