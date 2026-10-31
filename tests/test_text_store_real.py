"""Text normalization, language detection, SQLite queue, and checks against the real rulebook."""
from __future__ import annotations

import json

import pytest

from app.adapter import record_to_request, to_request
from app.classify import analyze, parse_rule_refs
from app.knowledge import RuleBook
from app.pipeline import CATEGORY_CODES, PRIORITY_CODES, Taxonomy, TriageService
from app.rag import ANCHORS, CANDIDATES
from app.store import Store
from app.textutil import GAME_KK_MAP, clean_for_matching, detect_lang, normalize_game_chars
from tests.conftest import REAL_DATA, requires_real_data

# ------------------------------------------------------------------ text
def test_game_symbols_become_kazakh_letters_but_original_is_kept(mk):
    text = "Ќайрат кґмек ѕиып ўй јз їс шаєым"
    out, n = normalize_game_chars(text)
    assert out == "Қайрат көмек ңиып ұй әз үс шағым" and n == 7
    a = analyze(mk("Ќайрат мені ќорлады", server_lang="kk"))
    assert "ќ" in a.raw_text and "қ" in a.text  # original preserved, normalized copy used for matching
    assert set(GAME_KK_MAP) >= set("ќґѕєјўї")


def test_game_symbol_table_follows_the_real_data_not_the_case_statement():
    """Real admin/player words found in the data: ј=ә, є=ғ, ґ=ө (the PDF/README table says є=ә, ґ=ғ, ј=ө)."""
    real = {"Сјлеметсіз": "Сәлеметсіз", "шаєымыѕызды": "шағымыңызды", "кґмек": "көмек", "дјлелдеріѕіз": "дәлелдеріңіз",
            "сўраєыѕызды": "сұрағыңызды", "баќылаудамын": "бақылаудамын", "жґндемейміз": "жөндемейміз", "їшін": "үшін", "ойынєа": "ойынға"}
    for raw, expected in real.items():
        assert normalize_game_chars(raw)[0] == expected, raw


def test_kazakh_game_encoding_round_trips():
    from app.textutil import to_game_chars

    s = "Сәлеметсіз, шағымыңызды дәлелдеріңіз көмек сұрағыңызды үшін бақылаудамын"
    assert normalize_game_chars(to_game_chars(s))[0] == s
    assert "ќ" in to_game_chars("қ") and to_game_chars("Ә") == "Ј"


def test_kk_draft_can_be_emitted_with_game_symbols_and_lang_can_be_restricted_to_the_contract(rulebook):
    import dataclasses

    from app.config import load_settings
    from app.dedup import DedupIndex
    from tests.conftest import make_req

    base = dataclasses.replace(load_settings(), kk_output_game_symbols=True, lang_strict_contract=True)
    svc = TriageService(base, rules=rulebook, dedup=DedupIndex())
    r, _ = svc.triage(make_req("Қайрат мені қорлады, скрин бар"))
    assert r.draft_reply.lang == "kk" and "ќ" in r.draft_reply.text and "қ" not in r.draft_reply.text and "ә" not in r.draft_reply.text
    assert r.lang == "kk"
    assert svc.triage(make_req("osk rodni"))[0].lang == "ru"            # lat -> ru in strict mode
    assert svc.triage(make_req("me insultó en el chat"))[0].lang == "und"  # es -> und in strict mode
    plain = TriageService(dataclasses.replace(load_settings()), rules=rulebook, dedup=DedupIndex())
    assert "қ" in plain.triage(make_req("Қайрат мені қорлады, скрин бар"))[0].draft_reply.text  # default: normal Kazakh
    assert plain.triage(make_req("osk rodni"))[0].lang == "lat"


def test_latin_text_is_not_touched_by_normalization():
    assert normalize_game_chars("hello world j u")[1] == 0


@pytest.mark.parametrize("text,expected", [
    ("оскорбил меня", "ru"), ("Jasur menga haqorat qildi, yordam bering", "uz"), ("Қайрат мені қорлады", "kk"),
    ("Ќайрат мені ќорлады", "kk"), ("he insulted me and the admin did not help", "en"), ("osk rodni", "lat"), ("+", "und"), ("12345", "und"),
    ("оск родни osk", "ru"),
])
def test_language_detection(text, expected):
    assert detect_lang(text) == expected


def test_clean_for_matching_squeezes_repeats_and_placeholders():
    assert clean_for_matching("ОСКОРБЛЕНИЕЕЕЕ <url> <digits>!!!") == "оскорблениее"  # 3+ repeats -> 2, tokens dropped


@pytest.mark.parametrize("raw,expected", [
    ("2.1 ОП", ["2.1"]), ("2,1 оп", ["2.1"]), ("1.1 гз", ["GZ 1.1"]), ("ЖБ 1.2.7", ["ЖБ 1.2.7"]), ("3,5,7", []), ("жб", []), ("", []),
    ("5.2op", ["5.2"]), ("5.3 и 5.4", ["5.3", "5.4"]),
])
def test_rule_reference_parsing(raw, expected):
    assert parse_rule_refs(raw) == expected


# ------------------------------------------------------------------ adapter
def test_adapter_drops_admin_and_outcome_fields():
    rec = {"ticket_id": "1", "kind": "complaint", "reporter_hash": "h", "images_count": 2, "reported_nick": "A_B",
           "messages": [{"role": "player", "text": "оск"}, {"role": "reported_player", "text": "это игра по роли"},
                        {"role": "admin", "text": "мут на 30 минут"}],
           "verdict": "approved", "punishment": "мут", "violation": "2.1 ОП", "category": "x", "priority": "p0"}
    req = record_to_request(rec)
    dumped = req.model_dump_json()
    assert "мут" not in dumped and "approved" not in dumped and "2.1 ОП" not in dumped
    assert [m.role for m in req.messages] == ["player", "reported_player"] and req.messages[0].attachments == 2
    compact = to_request({"kind": "report", "messages": [{"role": "player", "text": "hi"}, {"role": "admin", "text": "secret"}], "verdict": "x"})
    assert [m.role for m in compact.messages] == ["player"]


# ------------------------------------------------------------------ queue
def test_queue_filters_edits_and_local_approval(service, mk, tmp_path):
    st = Store(tmp_path / "q.db")
    ids = []
    for text, kind, att in [("оск родни", "complaint", 1), ("взломали аккаунт", "report", 0), ("помогите", "report", 0)]:
        req = mk(text, kind, att)
        resp, _ = service.triage(req)
        ids.append(st.add(req.model_dump(), resp.model_dump()))
    assert st.list()["total"] == 3
    assert st.list(category="report.account")["total"] == 1
    assert st.list(needs_human="0")["total"] == 1 and st.list(priority="p0")["items"][0]["category"] == "report.account"
    assert st.list(q="оск")["total"] == 1
    t = st.get(ids[0])
    original = t["response"]["draft_reply"]["text"]
    edited = st.update_status(ids[0], "edited", original + " Дополнено администратором.")
    assert edited["status"] == "edited" and edited["final_text"].endswith("администратором.")
    approved = st.update_status(ids[0], "approved_local", edited["final_text"])  # "confirm" re-sends the same text
    assert approved["status"] == "approved_local"
    feedback = st.export_edits()  # ... which must not create a second feedback row
    assert len(feedback) == 1 and feedback[0]["original_text"] == original and feedback[0]["edited_text"] != original
    assert st.update_status(999, "edited") is None
    with pytest.raises(ValueError):
        st.update_status(ids[0], "sent_to_player")  # there is no such status: nothing is ever sent


def test_admin_api_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "ui.db"))
    with TestClient(app) as c:
        assert c.post("/api/demo/seed").json()["added"] >= 10
        q = c.get("/api/queue", params={"needs_human": "1"}).json()
        assert q["total"] > 0
        tid = q["items"][0]["id"]
        t = c.get(f"/api/tickets/{tid}").json()
        assert t["response"]["draft_reply"]["text"]
        r = c.post(f"/api/tickets/{tid}/save", json={"text": "Правка администратора."}).json()
        assert r["status"] == "edited" and r["final_text"] == "Правка администратора."
        r = c.post(f"/api/tickets/{tid}/approve", json={}).json()
        assert r["status"] == "approved_local"
        assert c.post(f"/api/tickets/{tid}/escalate", json={"note": "нужен старший"}).json()["status"] == "escalated"
        assert c.get("/api/tickets/99999").status_code == 404
        assert c.get("/api/stats").json()["edits"] == 1
        assert c.get("/admin").status_code == 200 and "Support Copilot" in c.get("/admin").text


def test_template_suggestions_from_clusters(service, mk, tmp_path):
    from app.clusters import suggest

    st = Store(tmp_path / "c.db")
    ids = []
    for nick in ("Ali_Kim", "Bek_Orlov", "Dan_Fox"):
        req = mk(f"{nick} оск родни")
        resp, _ = service.triage(req)
        ids.append(st.add(req.model_dump(), resp.model_dump()))
    req = mk("летает по воздуху")
    st.add(req.model_dump(), service.triage(req)[0].model_dump())
    for i in ids[:2]:  # two admins independently saved the same improved text
        st.update_status(i, "edited", "Здравствуйте! Передали администратору (шаблон, подготовленный админом).")
    sug = suggest(st.all_full(), min_size=3)
    assert len(sug) == 1 and sug[0]["category"] == "complaint.insult" and sug[0]["size"] == 3
    assert sug[0]["source"] == "admin edits" and sug[0]["edit_support"] == 2 and "шаблон, подготовленный админом" in sug[0]["suggested_template"]
    assert suggest(st.all_full(), min_size=4) == []  # nothing is suggested without enough similar tickets


# ------------------------------------------------------------------ real organizers' files (skipped if absent)
@requires_real_data
def test_curated_rules_exist_and_pass_anchors_in_the_real_rulebook():
    rb = RuleBook.load(REAL_DATA / "rules.json")
    cites = {c for lst in CANDIDATES.values() for c, _ in lst} | {"ЖБ 1.2.2", "ЖБ 1.1.2", "ЖБ 1.3.4", "ЖБ 1.3.1", "ЖБ 1.2.9"}
    import re

    for c in cites:
        doc = rb.resolve_cite(c)
        assert doc is not None, f"curated rule {c} missing in rules.json"
        if c in ANCHORS:
            assert re.search(ANCHORS[c], doc.body, re.I), f"{c} no longer matches its anchor"


@requires_real_data
def test_real_rulebook_has_duplicate_refs_across_sections_and_we_address_by_section():
    rb = RuleBook.load(REAL_DATA / "rules.json")
    sections_with_2_1 = [d.section_id for d in rb.docs if d.ref == "2.1"]
    assert len(sections_with_2_1) > 1
    assert rb.resolve_cite("2.1").section_id == 30
    assert rb.context_docs, "section without points (RP terms) should be kept as context"


@requires_real_data
def test_builtin_taxonomy_matches_the_organizers_file():
    t = Taxonomy.load(REAL_DATA / "taxonomy.json")
    assert t.categories == CATEGORY_CODES and t.priorities == PRIORITY_CODES


@requires_real_data
def test_validation_file_runs_clean():
    svc = TriageService()
    valid = REAL_DATA / "tickets_valid.jsonl"
    if not valid.is_file():
        pytest.skip("tickets_valid.jsonl missing")
    for i, line in enumerate(valid.read_text(encoding="utf-8").splitlines()[:400]):
        resp, dbg = svc.triage(to_request(json.loads(line)))
        assert not dbg.guard_notes, (i, dbg.guard_notes)
        assert resp.category in CATEGORY_CODES
