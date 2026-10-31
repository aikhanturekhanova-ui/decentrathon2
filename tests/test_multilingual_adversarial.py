"""Adversarial multilingual suite (artificial cases in tests/fixtures/multilingual_cases.jsonl): language segments, Kazakh look-alikes,
Latin Kazakh/Russian, obfuscated swearing, quoted vs own abuse, threats in and out of the game, jokes, false-positive traps.
Known limitations are xfail: they document what the modules cannot do yet instead of hiding it."""
from __future__ import annotations

import pytest

from scripts.multilingual_eval import check, load_cases
from tests.conftest import make_req

CASES = load_cases()


@pytest.mark.parametrize("case", [
    pytest.param(c, id=c["group"], marks=pytest.mark.xfail(reason=c["known_limitation"], strict=True)) if c.get("known_limitation")
    else pytest.param(c, id=c["group"]) for c in CASES])
def test_case(case):
    assert check(case) == []


def test_offsets_point_into_the_original_text():
    from app.toxicity import analyze_messages
    text = "Он написал мне «с.у.к.а» и «3ае6ал», а ещё ЕБАНЫЙ"
    for f in analyze_messages([{"role": "player", "text": text}]).findings:
        assert text[f.start:f.end] == f.text


def test_reporting_victim_is_not_the_offender_in_triage(service):
    r, dbg = service.triage(make_req("Arman_Petrov написал мне в чате «ты сука, мать твою», скрины есть", att=2))
    tox = dbg.signals["toxicity"]
    assert tox["reported_abuse"] and not tox["author_abusive"]
    assert r.category == "complaint.insult" and r.priority == "p2"  # abuse never raises the priority by itself
    assert any(x.startswith("Модерация:") for x in r.reasons)


def test_real_life_threat_escalates_but_game_threat_does_not(service):
    r, _ = service.triage(make_req("он пишет что найдёт меня в реальной жизни и убьёт: «найду тебя irl, убью»", att=1))
    assert r.priority == "p0" and r.needs_human
    r, _ = service.triage(make_req("Arman_Petrov написал «убью тебя в игре» и убил у банка", att=1))
    assert r.priority != "p0"


def test_profanity_alone_does_not_raise_priority(service):
    r, _ = service.triage(make_req("бля помогите застрял в текстурах", kind="report", att=0))
    assert r.priority in {"p2", "p3"}


def test_mixed_language_reason(service):
    r, dbg = service.triage(make_req("Сәлем бро, help me please, застрял в текстурах", kind="report", att=0))
    assert dbg.signals["languages"]["mixed"] is True
    assert any(x.startswith("Сообщение на нескольких языках") for x in r.reasons)


def test_moderation_endpoint_and_ticket_card(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app
    monkeypatch.setenv("DB_PATH", str(tmp_path / "m.db"))
    with TestClient(app) as c:
        d = c.post("/api/moderation/analyze", json={"text": "он написал «с.у.к.а», help pls"}).json()
        assert d["summary"]["reported_abuse"] and d["messages"][0]["findings"][0]["obfuscated"] is True
        tid = c.post("/api/tickets", json={"kind": "complaint", "messages": [{"role": "player", "text": "он написал «дебил»", "attachments": 1}]}).json()["id"]
        card = c.get(f"/api/tickets/{tid}").json()
        assert card["moderation"]["messages"][0]["findings"][0]["text"] == "дебил"
        assert c.post("/api/moderation/analyze", json={"text": "x" * 5000}).status_code == 422


@pytest.mark.parametrize("text,category", [
    ("Arman_Petrov маған көлік сатамын деп ақшамды алды да, көлікті бермеді", "complaint.scam"),
    ("маған бан бекер берілді, мен ешкімді өлтірген жоқпын, қайта қараңызшы", "complaint.admin"),
    ("Arman_Petrov мені банктің жанында себепсіз өлтірді, видео бар", "complaint.dm"),
])
def test_kazakh_scam_and_appeal_are_recognised(service, text, category):
    r, _ = service.triage(make_req(text, att=1))
    assert r.category == category and r.lang == "kk"
    if category == "complaint.admin":
        assert r.needs_human  # an appeal always goes to a person


@pytest.mark.parametrize("text,lang,form", [
    ("он написал «сен котак»", "kk", "казахское слово русскими буквами (без казахских букв)"),
    ("ол «qotaq» деп жазды", "kk", "казахское слово латиницей"),
    ("он пишет «фак ю»", "en", "английское слово кириллицей"),
    ("он написал «blyad»", "ru", "русское слово латиницей"),
    ("он написал «пiзда»", "ru", "смешение алфавитов внутри слова"),
    ("он написал «қотақ»", "kk", None),  # Kazakh in Kazakh letters: nothing unusual to report
])
def test_foreign_alphabet_is_reported(text, lang, form):
    from app.toxicity import analyze_messages
    f = analyze_messages([{"role": "player", "text": text}]).findings[0]
    assert f.lang == lang and f.form == form


def test_mixed_language_abuse_is_explained():
    from app.toxicity import analyze_messages
    rep = analyze_messages([{"role": "player", "text": "он написал «ебаный qotaq, fuck you»"}])
    assert rep.mixed_language_abuse and set(rep.languages) >= {"ru", "kk", "en"}
    assert any(x.startswith("Грубые слова на нескольких языках") for x in rep.explanation)
    assert any(x.startswith("Слова записаны чужим алфавитом") for x in rep.explanation)


def test_one_stem_matches_every_alphabet():
    from app.toxicity import normalize, skeleton
    for spelling in ("қотақ", "котак", "qotaq", "kotak", "КОТАК", "q.o.t.a.q", "қооотақ"):
        assert skeleton(normalize(spelling))[0] == "kotak", spelling


def test_quoted_plain_insult_is_evidence_for_an_insult_complaint():
    # regression: the toxicity type names were shadowed by classify's INSULT/PROFANITY regexes, so only family/hate/sexual abuse counted
    from app.classify import analyze
    a = analyze(make_req("он мне написал «дебил»", att=1))
    assert "toxicity:reported" in a.matched.get("complaint.insult", [])
    a = analyze(make_req("он мне написал «бля»", att=1))
    assert "toxicity:reported" in a.matched.get("complaint.insult", [])


@pytest.mark.parametrize("text,kind,category", [
    ("Arman_Petrov mashina sotaman deb pulimni oldi, mashinani bermadi", "complaint", "complaint.scam"),  # «oldi» = took, not died
    ("Arman_Petrov meni bank yonida sababsiz o'ldirdi, video bor", "complaint", "complaint.dm"),
    ("Arman_Petrov meni oldirdi", "complaint", "complaint.dm"),
    ("teksturada qolib ketdim, yordam bering", "report", "report.help"),
])
def test_uzbek_scam_dm_and_stuck_are_recognised(service, text, kind, category):
    r, _ = service.triage(make_req(text, kind=kind, att=1))
    assert r.category == category
