"""Regression tests for defects found in an independent review of the first version."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from app.cli import predict
from app.config import load_settings
from app.dedup import DedupIndex
from app.guard import find_promises
from app.llm import LLMClient
from app.pipeline import TriageService

ROOT = Path(__file__).resolve().parent.parent


# 1. mandatory handoff must not depend on the final category or on attachments --------------------
@pytest.mark.parametrize("text,att", [
    ("Меня забанили несправедливо, прошу пересмотреть наказание", 0),
    ("Меня забанили несправедливо, прошу пересмотреть наказание", 2),
    ("не согласен с баном", 0),
    ("опру мут", 0),
    ("дали варн не за что", 0),
    ("unban qilishingizni so‘rayman, nohaq ban berishdi", 0),
    ("please unban me, the ban was unfair", 0),
])
def test_appeals_go_to_a_human_with_or_without_attachments(service, mk, text, att):
    r, _ = service.triage(mk(text, att=att))
    assert r.needs_human is True, (text, r.category)
    assert r.category != "complaint.invalid"  # an appeal is not "a complaint without evidence"


def test_the_exact_reported_case(service, mk):
    r, _ = service.triage(mk("Меня забанили несправедливо, прошу пересмотреть наказание", att=0))
    assert r.category == "complaint.admin" and r.needs_human and r.priority == "p2"
    assert "скриншот наказания" in r.draft_reply.text  # asks for the penalty screenshot instead of rejecting


@pytest.mark.parametrize("text,kind,att", [
    ("продаёт аккаунт, я оплатил донат и не получил, верните деньги", "complaint", 0),
    ("вот он", "complaint", 1),
])
def test_payment_and_hack_flags_are_independent_of_category(service, mk, text, kind, att):
    r, dbg = service.triage(mk(text, kind, att))
    if "payment" in dbg.signals["flags"]:
        assert r.needs_human is True


def test_hack_or_payment_inside_any_ticket_forces_human(service, mk):
    for text in ["мой аккаунт взломали", "списали деньги за донат, верните", "my account was hacked and I paid for coins"]:
        for kind, att in (("complaint", 0), ("complaint", 2), ("report", 0)):
            r, _ = service.triage(mk(text, kind, att))
            assert r.needs_human is True, (text, kind, r.category)


# 2. negation must apply to the promise it is attached to -------------------------------------------
@pytest.mark.parametrize("text", [
    "Не волнуйтесь, мы вернём деньги.",
    "Не волнуйтесь мы вернём деньги.",
    "Не можем обещать сроки, но вернём деньги.",
    "Мы не можем обещать сроки, а деньги вернём.",
    "Деньги вернут в течение недели.",
    "Don't worry, we will refund you.",
    "We cannot promise a date, but we will refund you.",
    "No worries, you will get your money back.",
    "Игрок не уйдёт от ответственности: он будет наказан.",
    "Ақшаңызды қайтарамыз, алаңдамаңыз.",
    "Pulingizni qaytaramiz, xavotir olmang.",
])
def test_negation_does_not_hide_a_promise(text):
    assert find_promises(text), text


@pytest.mark.parametrize("text", [
    "Мы не можем обещать возврат.",
    "Не можем обещать, что вернём деньги.",
    "Мы не обещаем, что игрок будет наказан.",
    "We cannot promise that we will refund you.",
    "We do not promise a refund.",
    "Ақшаны қайтарамыз деп уәде бере алмаймыз.",
    "Pulni qaytaramiz deb va’da bera olmaymiz.",
    "Заранее сообщить результат или наказание мы не можем.",
])
def test_a_negated_promise_is_not_a_promise(text):
    assert find_promises(text) == [], text


def test_llm_reply_with_negation_trick_is_rejected(rulebook, tmp_path, mk):
    s = dataclasses.replace(load_settings(), llm_api_key="k")
    bad = {"text": "Здравствуйте! Не волнуйтесь, мы вернём деньги. [[R1]]", "cited": ["R1"]}
    handler = lambda req: httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(bad, ensure_ascii=False)}], "usage": {"input_tokens": 5, "output_tokens": 5}})  # noqa: E731
    svc = TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(handler)), dedup=DedupIndex())
    r, dbg = svc.triage(mk("Max_Orlov оск родни"))
    assert dbg.mode == "template" and "вернём" not in r.draft_reply.text and dbg.llm_error


# 3. a question about a rule is answered from the rule, not treated as a complaint ----------------
def test_rule_question_gets_a_content_answer(service, mk):
    r, _ = service.triage(mk("Можно ли оскорблять других игроков в чате?", "report", 0))
    assert r.category == "report.question" and r.draft_reply.cited == ["2.1"]
    assert "ник игрока" not in r.draft_reply.text.lower()  # does not ask for a violator's nickname
    assert "оскорблен" in r.draft_reply.text and r.needs_human is False
    r2, _ = service.triage(mk("Разрешено ли использовать читы?", "report", 0))
    assert r2.category == "report.question" and r2.draft_reply.cited == ["5.2"]
    r3, _ = service.triage(mk("Is it allowed to sell an account for real money?", "report", 0))
    assert r3.category == "report.question" and r3.draft_reply.cited and r3.draft_reply.lang == "en"


def test_victims_and_mechanics_questions_are_not_hijacked(service, mk):
    assert service.triage(mk("меня оскорбляют, можно что-то сделать?", "report", 0))[0].category == "report.complaint"
    r, _ = service.triage(mk("как включить музыку в машине?", "report", 0))
    assert r.category == "report.question" and r.draft_reply.cited == [] and r.needs_human is True  # mechanics: no rule attached
    assert service.triage(mk("можно вытащить меня из текстур?", "report", 0))[0].category == "report.help"


# 5. unknown cost is not zero ------------------------------------------------------------------------
def test_unknown_cost_is_reported_as_unknown(rulebook, tmp_path, mk):
    s = dataclasses.replace(load_settings(), llm_api_key="k", llm_price_in_per_mtok=0.0, llm_price_out_per_mtok=0.0)
    good = {"text": "Здравствуйте! По вашим словам, игрок оскорблял вас. [[R1]] Администратор проверит жалобу.", "cited": ["R1"]}
    handler = lambda req: httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(good, ensure_ascii=False)}], "usage": {"input_tokens": 700, "output_tokens": 90}})  # noqa: E731
    svc = TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(handler)), dedup=DedupIndex())
    inp, out = tmp_path / "i.jsonl", tmp_path / "o.jsonl"
    inp.write_text(json.dumps({"kind": "complaint", "messages": [{"role": "player", "text": "Max_Orlov оск родни", "attachments": 1}]}) + "\n", encoding="utf-8")
    summary = predict(str(inp), str(out), service=svc)
    assert summary["llm_calls"] == 1
    assert summary["cost_status"] == "unknown" and summary["cost_usd_total"] is None and summary["tickets_with_unknown_cost"] == 1
    assert summary["cost_usd_known_sum"] is None and summary["cost_usd_per_ticket"] is None
    stats = json.loads(Path(str(out) + ".stats.json").read_text(encoding="utf-8"))
    assert stats["cost_usd_total"] is None  # NOT 0.0


def test_no_external_calls_is_a_real_zero(service, tmp_path):
    inp, out = tmp_path / "i.jsonl", tmp_path / "o.jsonl"
    inp.write_text(json.dumps({"kind": "report", "messages": [{"role": "player", "text": "помогите"}]}) + "\n", encoding="utf-8")
    s = predict(str(inp), str(out), service=service)
    assert s["cost_status"] == "no_external_calls" and s["cost_usd_total"] == 0.0


# 7. robustness found by the full-data audit ------------------------------------------------------------------
def test_ticket_with_145_images_and_null_fields_do_not_crash(service, tmp_path):
    from app.adapter import to_request

    rec = {"ticket_id": "big", "kind": "complaint", "reporter_hash": "h", "images_count": 145, "messages": [{"role": "player", "text": "оск родни"}]}
    r, _ = service.triage(to_request(rec))
    assert r.extracted.attachments == 145 and r.category == "complaint.insult"
    r2, _ = service.triage(to_request({"kind": "report", "messages": [{"role": "player", "text": None, "attachments": None}]}))
    assert r2.category == "report.offtopic"
    r3, _ = service.triage(to_request({"kind": "report", "messages": None}))
    assert r3.category == "report.offtopic"
    inp, out = tmp_path / "i.jsonl", tmp_path / "o.jsonl"
    inp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert predict(str(inp), str(out), service=service)["errors"] == 0


# 6. container ships the demo tickets --------------------------------------------------------------------
def test_dockerfile_copies_demo_fixture():
    assert "tests/fixtures/demo_tickets.jsonl" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
