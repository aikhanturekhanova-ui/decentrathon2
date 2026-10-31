"""API contract, taxonomy conformity, regression on the synthetic gold set, CLI == API."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.cli import predict
from app.main import app
from app.pipeline import CATEGORY_CODES, PRIORITY_CODES
from app.schemas import TriageResponse

FIX = Path(__file__).parent / "fixtures"
GOLD = [json.loads(l) for l in (FIX / "gold_synthetic.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
EXAMPLE = {
    "kind": "complaint",
    "messages": [{"role": "player", "text": "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП", "attachments": 2}],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    with TestClient(app) as c:
        yield c


def gold_body(g: dict) -> dict:
    body = {"kind": g["kind"], "messages": [{"role": "player", "text": g["text"], "attachments": g["att"]}]}
    if g.get("server_lang"):
        body["server_lang"] = g["server_lang"]
    return body


def test_contract_example(client):
    r = client.post("/triage", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    TriageResponse.model_validate(body)  # exact schema
    assert set(body) == {"lang", "category", "priority", "is_spam_or_duplicate", "extracted", "draft_reply", "needs_human", "confidence", "reasons"}
    assert body["category"] == "complaint.insult" and body["priority"] == "p2" and body["lang"] == "ru"
    assert body["extracted"]["nickname"] == "Arman_Petrov"
    assert body["extracted"]["rule_refs"] == ["2.1"]
    assert body["extracted"]["has_evidence"] is True
    assert body["extracted"]["evidence_verified"] is False  # attachments are never verified
    assert body["draft_reply"]["cited"] == ["2.1"]
    assert body["needs_human"] is True
    assert 0.0 <= body["confidence"] <= 1.0 and body["reasons"]
    assert r.headers["X-Triage-Mode"] == "template"


def test_contract_example_does_not_copy_the_dangerous_sample(client):
    """The case statement's sample reply promises a mute and says the screenshots show a violation."""
    text = client.post("/triage", json=EXAMPLE).json()["draft_reply"]["text"].lower()
    assert "получит мут" not in text and "по скриншотам видно" not in text
    assert "не просматриваем" in text  # we say explicitly that attachments were not viewed


@pytest.mark.parametrize("bad", [{}, {"kind": "complaint"}, {"kind": "nope", "messages": [{"text": "x"}]}, {"kind": "report", "messages": []}])
def test_invalid_requests_are_422(client, bad):
    assert client.post("/triage", json=bad).status_code == 422


def test_empty_and_huge_text_do_not_crash(client):
    assert client.post("/triage", json={"kind": "report", "messages": [{"role": "player", "text": ""}]}).status_code == 200
    r = client.post("/triage", json={"kind": "complaint", "messages": [{"role": "player", "text": "оск " * 5000, "attachments": 1}]})
    assert r.status_code == 200


def test_every_output_is_inside_the_taxonomy(client):
    for g in GOLD:
        body = client.post("/triage", json=gold_body(g)).json()
        assert body["category"] in CATEGORY_CODES and body["priority"] in PRIORITY_CODES
        assert body["category"].startswith(g["kind"] + ".")  # kind and category family agree


def test_synthetic_gold_regression(service, mk):
    """Regression guard on artificial examples written by the authors.

    NOT an accuracy estimate: the labels are ours, the examples were written next to the lexicon.
    """
    bad_cat, bad_pri, bad_lang = [], [], []
    for g in GOLD:
        req = mk(g["text"], g["kind"], g["att"], server_lang=g.get("server_lang"))
        resp, _ = service.triage(req)
        e = g["expect"]
        if resp.category != e["category"]:
            bad_cat.append((g["text"], e["category"], resp.category))
        if resp.priority != e["priority"]:
            bad_pri.append((g["text"], e["priority"], resp.priority))
        if "lang" in e and resp.lang != e["lang"]:
            bad_lang.append((g["text"], e["lang"], resp.lang))
    n = len(GOLD)
    assert len(bad_cat) / n <= 0.03, bad_cat
    assert len(bad_pri) / n <= 0.04, bad_pri
    assert len(bad_lang) / n <= 0.05, bad_lang


def test_cli_and_api_share_one_pipeline(client, service, tmp_path):
    """make predict and POST /triage must give identical answers for the same ticket."""
    sample = GOLD[:40]
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text("\n".join(json.dumps({**gold_body(g), "ticket_id": f"t{i}"}) for i, g in enumerate(sample)), encoding="utf-8")
    summary = predict(str(inp), str(out), service=service)
    assert summary["errors"] == 0 and summary["records"] == len(sample)
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    for i, (g, line) in enumerate(zip(sample, lines)):
        assert line.pop("ticket_id") == f"t{i}"
        api = client.post("/triage", json=gold_body(g)).json()
        assert line == api, g["text"]


def test_predict_accepts_full_dataset_records_and_ignores_admin_text(service, tmp_path):
    rec = {
        "ticket_id": "x1", "kind": "complaint", "created_at": "2026-03-14T19:02:00", "region": "kz", "server_lang": "kk", "lang": "ru",
        "reporter_hash": "a1", "reported_hash": "b1", "reported_nick": "Arman_Petrov", "rule_refs_raw": "2.1", "images_count": 2, "status": 2,
        "messages": [
            {"role": "player", "ts": "t", "text": "оскорбил меня и семью, скрины есть"},
            {"role": "admin", "ts": "t", "text": "Игрок Arman_Petrov получит: мут на 30 минут за оскорбление."},
        ],
        "verdict": "approved", "punishment": "мут на 30 минут", "violation": "2.1 ОП",
    }
    other = json.loads(json.dumps(rec))
    other["messages"][1]["text"] = "Нарушения нет, жалоба отклонена."
    other["verdict"], other["punishment"], other["violation"] = "rejected", None, None
    other["ticket_id"], other["reporter_hash"] = "x2", "a2"  # a different author: otherwise it is (correctly) a duplicate
    inp, out = tmp_path / "i.jsonl", tmp_path / "o.jsonl"
    inp.write_text(json.dumps(rec, ensure_ascii=False) + "\n" + json.dumps(other, ensure_ascii=False) + "\n", encoding="utf-8")
    predict(str(inp), str(out), service=service)
    a, b = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    a.pop("ticket_id"), b.pop("ticket_id")
    # administrator answer / verdict must not influence anything
    for k in ("lang", "category", "priority", "extracted", "draft_reply", "needs_human", "confidence"):
        assert a[k] == b[k], k
    assert "мут на 30" not in json.dumps(a, ensure_ascii=False)
