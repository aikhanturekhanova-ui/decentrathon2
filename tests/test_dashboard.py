"""Dashboard API: live queue counters, critical alerts, historical aggregates; /health exposes the classifier mode."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import app.main as main_module  # resolved at call time: tests/test_review3_http.py reloads app.main


LEXICON_ONLY = pytest.mark.skipif(os.environ.get("TEST_USE_MODEL") == "1", reason="asserts the hermetic lexicon-only mode")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "dash.db"))
    with TestClient(main_module.app) as c:
        yield c


@LEXICON_ONLY
def test_empty_queue_dashboard(client):
    d = client.get("/api/insights").json()
    assert d["queue"]["total"] == 0 and d["queue"]["open_critical"] == [] and d["queue"]["low_confidence"] == 0
    assert "lexicon only" in d["classifier"]  # tests never load the locally trained model


def test_dashboard_after_demo_seed(client):
    assert client.post("/api/demo/seed").json()["added"] > 0
    q = client.get("/api/insights").json()["queue"]
    assert q["total"] > 0 and q["needs_human"] <= q["total"]
    assert {t["category"] for t in q["open_critical"]} >= {"report.payment", "report.account"}
    assert all(set(t) == {"id", "category", "preview"} for t in q["open_critical"])
    # a handled critical ticket leaves the alert list
    first = q["open_critical"][0]["id"]
    client.post(f"/api/tickets/{first}/approve", json={"text": "ok"})
    after = client.get("/api/insights").json()["queue"]["open_critical"]
    assert first not in {t["id"] for t in after}


@LEXICON_ONLY
def test_health_names_the_classifier(client):
    h = client.get("/health").json()
    assert "classifier" in h and h["model_meta"] == {}


def test_history_aggregates_publish_no_rare_strings(client):
    hist = client.get("/api/insights").json()["history"]
    if hist is None:
        pytest.skip("docs/metrics/history.json not generated")
    assert all(n >= hist["min_count_for_strings"] for _, n in hist["top_punishments"])
