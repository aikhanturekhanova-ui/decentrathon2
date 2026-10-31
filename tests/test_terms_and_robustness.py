"""Game terms from the rulebook section «RP термины» (DM, DB, SK, TK, NonRP, MG), missing rules.json, renamed taxonomy."""
from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from app.config import load_settings
from app.dedup import DedupIndex
from app.guard import check_draft
from app.llm import LLMClient
from app.phrases import TERM_GISTS
from app.pipeline import Taxonomy, TriageService
from tests.conftest import FIXTURES, make_req


@pytest.mark.parametrize("text,term", [
    ("убил меня без причины", "DM"), ("дм", "DM"), ("дб на машине", "DB"), ("ск у спавна", "SK"), ("тк", "TK"),
    ("нон рп коп", "NonRP"), ("мг", "MG"),
])
def test_slang_maps_to_the_term_defined_in_the_rulebook(service, text, term):
    r, _ = service.triage(make_req(text))
    assert r.draft_reply.terms == [term] and f"{term} —" in r.draft_reply.text
    assert "от места зависит" not in r.draft_reply.text  # the answer is grounded: no need to ask which rule applies
    assert r.needs_human is True
    assert "DEMORGAN" not in r.draft_reply.text and "демморган" not in r.draft_reply.text.lower()  # sanctions are never quoted


def test_numbered_rule_and_term_can_both_be_used(service):
    r, _ = service.triage(make_req("дм в зелёной зоне"))
    assert r.draft_reply.cited == ["GZ 1.1"] and r.draft_reply.terms == ["DM"]


def test_report_complaint_also_names_the_term_and_other_categories_do_not(service):
    r, _ = service.triage(make_req("339 дм в зз", "report", 0))
    assert r.category == "report.complaint" and r.draft_reply.terms == ["DM"]
    assert service.triage(make_req("оск родни"))[0].draft_reply.terms == []
    assert service.triage(make_req("летает по воздуху"))[0].draft_reply.terms == []


@pytest.mark.parametrize("lang,text", [("ru", "убил меня без причины"), ("en", "he killed me for no reason"), ("kk", "себепсіз өлтірді"),
                                       ("uz", "sababsiz o'ldirdi"), ("es", "me mató sin motivo"), ("pt", "me matou sem motivo")])
def test_term_sentence_is_in_the_players_language(service, rulebook, lang, text):
    r, _ = service.triage(make_req(text))
    assert r.draft_reply.lang == lang and r.draft_reply.terms == ["DM"], text
    assert TERM_GISTS["DM"][lang] in r.draft_reply.text
    assert check_draft(r.draft_reply.text, r.draft_reply.cited, rulebook, r.needs_human, terms=r.draft_reply.terms).ok


def test_guard_rejects_unknown_or_unmentioned_terms(rulebook):
    assert not check_draft("Текст.", [], rulebook, terms=["XYZ"]).ok          # not in the rulebook
    assert not check_draft("Текст без термина.", [], rulebook, terms=["DM"]).ok  # claimed but not named
    assert check_draft("Из раздела правил «RP термины»: DM — убийство без основательной RP-причины.", [], rulebook, terms=["DM"]).ok


def test_llm_text_gets_the_verified_term_sentence(rulebook):
    s = dataclasses.replace(load_settings(), llm_api_key="k")
    body = {"text": "Здравствуйте! Администратор проверит жалобу.", "cited": []}  # no model-written "по вашим словам" (see test_review5)
    h = lambda req: httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}], "usage": {"input_tokens": 5, "output_tokens": 5}})  # noqa: E731
    svc = TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(h)), dedup=DedupIndex())
    r, dbg = svc.triage(make_req("убил меня без причины"))
    assert dbg.mode == "llm" and r.draft_reply.terms == ["DM"] and "DM —" in r.draft_reply.text


# --- rules.json missing / elsewhere ------------------------------------------------------------------------------------
def test_missing_rules_degrade_gracefully_instead_of_crashing(tmp_path):
    s = dataclasses.replace(load_settings(), rules_path=tmp_path / "nope.json", taxonomy_path=tmp_path / "nope2.json")
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)  # nothing to discover here either
    try:
        svc = TriageService(s, dedup=DedupIndex())
    finally:
        os.chdir(cwd)
    if svc.rules.docs:  # the project's own data/handout was found through ROOT: that is the intended discovery
        pytest.skip("organizers' files are present in the project; discovery found them")
    r, _ = svc.triage(make_req("оск родни, 2.1 ОП"))
    assert r.category == "complaint.insult" and r.draft_reply.cited == [] and r.needs_human
    assert any("rules.json" in x and "не найдена" in x for x in r.reasons)
    assert not any("нет в правилах" in x for x in r.reasons)  # the player's rule number is not called unknown when we have no rules at all


def test_rules_are_found_next_to_the_input_file(tmp_path, rulebook):
    (tmp_path / "rules.json").write_text((FIXTURES / "rules_synthetic.json").read_text(encoding="utf-8"), encoding="utf-8")
    s = dataclasses.replace(load_settings(), rules_path=tmp_path / "elsewhere" / "rules.json")
    svc = TriageService(s, hint_dirs=[tmp_path], dedup=DedupIndex())
    assert svc.rules.docs and svc.rules_source == str(tmp_path / "rules.json")
    assert svc.triage(make_req("оск родни"))[0].draft_reply.cited == ["2.1"]


# --- the organizers' final taxonomy may differ from the draft ---------------------------------------------------------------
def test_renamed_taxonomy_never_crashes_and_never_emits_unknown_codes(service, tmp_path):
    p = tmp_path / "taxonomy.json"
    p.write_text(json.dumps({"categories": [{"code": c} for c in ("complaint.insult", "complaint.other", "report.question", "report.offtopic")],
                             "priorities": [{"code": "p1"}, {"code": "p3"}]}), encoding="utf-8")
    service.taxonomy = Taxonomy.load(p)
    r, _ = service.triage(make_req("продаёт аккаунт за реал в тг"))      # complaint.rmt_ads is absent in this taxonomy
    assert r.category == "complaint.other" and r.priority in {"p1", "p3"} and any("отсутствует в таксономии" in x for x in r.reasons)
    r2, _ = service.triage(make_req("взломали аккаунт", "report", 0))   # report.account absent -> report.question
    assert r2.category == "report.question" and r2.needs_human is True   # mandatory handoff survives the renaming
    assert Taxonomy.load(tmp_path / "missing.json").categories and Taxonomy.load(tmp_path).categories  # missing file / a directory: built-in
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert Taxonomy.load(tmp_path / "bad.json").categories


def test_admin_card_returns_term_definitions(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    with TestClient(app) as c:
        tid = c.post("/api/tickets", json={"kind": "complaint", "messages": [{"role": "player", "text": "убил меня без причины", "attachments": 1}]}).json()["id"]
        t = c.get(f"/api/tickets/{tid}").json()
        assert t["terms_detail"] and t["terms_detail"][0]["abbr"] == "DM" and "основательной" in t["terms_detail"][0]["definition"]
