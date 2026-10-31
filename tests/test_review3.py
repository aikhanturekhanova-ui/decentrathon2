"""Regression tests for the third independent review: sensitive topics in mixed tickets, priority order, LLM guard through the whole
service, place-bound rules, terms, evidence wording, honest handoff wording, language detection, dedup, ad complaints, LLM budget.

They use new phrasings, permutations and negative examples, not only the sentences that exposed the problems.
"""
from __future__ import annotations

import dataclasses
import json
import re

import httpx
import pytest

from app.config import load_settings
from app.dedup import DedupIndex
from app.guard import check_draft, check_free_text, find_promises
from app.llm import LLMClient
from app.pipeline import TriageService
from app.textutil import detect_lang
from tests.conftest import make_req


def run(service, text, kind="report", att=0, **extra):
    r, d = service.triage(make_req(text, kind=kind, att=att, **extra))
    return r, d


# ------------------------------------------------------------------------------------------------ 4. mandatory handoff
SENSITIVE = [
    # payment / refund (also hidden behind a rule question)
    "Верните 500 рублей за покупку. Можно ли подать жалобу?", "Можно ли подать жалобу? Верните 500 рублей за покупку.",
    "Я оплатил донат, коины не пришли, как оформить возврат?", "Списали деньги с карты за подписку, верните пожалуйста",
    "I paid for coins and they did not arrive, refund please", "Pagué una donación y no llegaron las monedas",
    "Paguei e não recebi, quero reembolso", "To'lov qildim, pulni qaytaring", "Төлем жасадым, қайтарыңыз",
    # hacking / lost access (also followed by an ordinary question)
    "Меня взломали, кто-то зашёл в мой аккаунт. Можно ли материться в чате?", "Можно ли материться в чате? Меня взломали.",
    "Менің аккаунтыма кіре алмаймын. Ойыншыны қорлауға рұқсат па?", "Ойыншыны қорлауға рұқсат па? Менің аккаунтыма кіре алмаймын.",
    "Аккаунтимни бузишди", "Akkauntimni buzishdi", "Akkauntga kira olmayman", "Someone hacked my account", "I can't log in to my account",
    "Hackearon mi cuenta", "No puedo entrar a mi cuenta", "Hackearam minha conta", "Não consigo entrar na minha conta",
    "Не могу зайти в аккаунт, забыл пароль", "Потерял доступ к аккаунту",
]


@pytest.mark.parametrize("kind", ["report", "complaint"])
@pytest.mark.parametrize("text", SENSITIVE)
def test_sensitive_topics_always_reach_a_person_and_p0(service, text, kind):
    r, _ = run(service, text, kind=kind, att=0)
    assert r.needs_human is True, (kind, text, r.category)
    assert r.priority == "p0", (kind, text, r.category, r.priority)
    assert not any(x in r.draft_reply.text for x in ("подайте жалобу заново", "нельзя принять", "Submit the complaint again"))


@pytest.mark.parametrize("text", SENSITIVE[:8])
def test_sensitive_topic_survives_attachments_and_other_category(service, text):
    r, _ = run(service, text + " Игрок Max_Orlov оскорбил меня в чате.", kind="complaint", att=3)
    assert r.needs_human and r.priority == "p0"


@pytest.mark.parametrize("text", [
    "Можно ли оскорблять в чате?", "Что такое DM?", "Что такое ДБ?", "Можно ли использовать читы?", "Is it allowed to advertise on the server?",
])
def test_ordinary_safe_questions_are_still_answered_automatically(service, text):
    r, _ = run(service, text)
    assert r.category == "report.question" and r.needs_human is False, (text, r.category, r.reasons)
    assert r.draft_reply.cited or r.draft_reply.terms


@pytest.mark.parametrize("text", ["Как оплатить донат?", "Что такое донат?"])
def test_pure_payment_questions_are_not_treated_as_refund_incidents(service, text):
    r, _ = run(service, text)
    assert "payment" not in " ".join(r.reasons).lower() or r.category == "report.payment"


# ------------------------------------------------------------------------------------------------ 5. priority order
@pytest.mark.parametrize("kind", ["complaint", "report"])
@pytest.mark.parametrize("text", [
    "Arman_Petrov взломал мой аккаунт", "Я оплатил донат, а игрок украл мои вещи", "Игрок обманул на машину и угрожает найти меня в реале",
    "Он угрожает убить меня в реальной жизни", "Угрожает найти мой адрес и приехать, скрины есть", "Игрок мошенник, обманул на 3 млн, я оплатил донат",
])
def test_critical_signals_give_p0_and_later_branches_do_not_lower_it(service, text, kind):
    r, _ = run(service, text, kind=kind, att=2)
    assert r.priority == "p0", (kind, text, r.category, r.priority)
    assert r.needs_human


def test_scam_without_critical_signal_keeps_its_own_priority(service):
    big, _ = run(service, "Max_Orlov обманул меня на 5 млн, скрины есть", kind="complaint", att=1)
    small, _ = run(service, "Max_Orlov обманул меня на ручку, скрины есть", kind="complaint", att=1)
    assert big.category == small.category == "complaint.scam" and (big.priority, small.priority) == ("p1", "p2")


# ------------------------------------------------------------------------------------------------ 6. guard through the service
def svc_llm(rulebook, answer, calls=None, **cfg):
    def handler(req):
        if calls is not None:
            calls.append(1)
        return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps({"text": answer, "cited": []})}], "usage": {"input_tokens": 100, "output_tokens": 40}})

    s = dataclasses.replace(load_settings(), llm_api_key="k", llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=5.0, **cfg)
    return TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(handler)), dedup=DedupIndex())


@pytest.mark.parametrize("answer", [
    "Здравствуйте! Игроку назначат блокировку. Проверка займёт 24 часа.",
    "Здравствуйте! Вы получите средства обратно.", "Здравствуйте! Средства поступят на ваш счёт.",
    "Здравствуйте! Администратор отправит вам компенсацию.", "Здравствуйте! Не волнуйтесь, деньги вам вернут.",
    "Здравствуйте! Нарушитель будет наказан в ближайшее время.", "Здравствуйте! Мы свяжемся с вами завтра.",
    "Здравствуйте! Обратитесь в поддержку по адресу help@example.com.", "Hello! The player will receive a penalty.",
    "Hello! We will review it within 2 hours.", "Здравствуйте! Мы разберёмся.[[R1]] Бан будет выдан.",
])
def test_llm_text_with_promises_or_invented_facts_is_rejected_by_the_whole_service(rulebook, answer):
    r, dbg = svc_llm(rulebook, answer).triage(make_req("Max_Orlov оск родни"))
    assert dbg.mode == "template" and dbg.llm_error, answer
    assert any("Черновик модели отклонён" in x for x in r.reasons)
    assert not find_promises(r.draft_reply.text)


@pytest.mark.parametrize("answer", [
    "Здравствуйте! Ваше сообщение получено. Мы не можем обещать наказание или возврат. Опишите, когда это произошло?",
    "Здравствуйте! Ваше сообщение получено. Администратор рассмотрит его. [[R1]]",
])
def test_llm_text_that_only_describes_the_process_is_accepted(rulebook, answer):
    r, dbg = svc_llm(rulebook, answer).triage(make_req("Max_Orlov оск родни"))
    assert dbg.mode == "llm", (dbg.llm_error, answer)


@pytest.mark.parametrize("text", [
    "Мы не можем обещать возврат средств.", "Мы не можем гарантировать, что средства поступят.", "Cannot promise a refund.",
    "No podemos prometer un reembolso.", "Não podemos prometer reembolso.", "Qaytarishni va'da qila olmaymiz.",
])
def test_negated_outcome_mentions_are_not_promises(text):
    assert check_free_text(text) == [] and not find_promises(text), text


@pytest.mark.parametrize("text", [
    "Не волнуйтесь, деньги вам вернут.", "Средства поступят на ваш счёт.", "Вы получите средства обратно.", "Администратор отправит вам компенсацию.",
    "Игроку назначат блокировку.", "Don't worry, you will get your money back.", "Los fondos llegarán a su cuenta.", "Receberá o dinheiro de volta.",
])
def test_promise_forms_are_blocked_by_the_free_text_check(text):
    assert check_free_text(text), text


@pytest.mark.parametrize("text", [
    "Вы получите средства обратно.", "Средства поступят на ваш счёт.", "Администратор отправит вам компенсацию.", "Игроку назначат блокировку.",
])
def test_check_draft_blocks_the_reported_promises(rulebook, text):
    assert not check_draft(text, [], rulebook).ok


def test_sensitive_tickets_never_reach_the_model(rulebook):
    calls: list = []
    svc = svc_llm(rulebook, "Здравствуйте! Мы поможем.", calls)
    for text in ("Верните 500 рублей за покупку", "Меня взломали", "Он угрожает убить меня в реальной жизни"):
        r, dbg = svc.triage(make_req(text, kind="report", att=0))
        assert dbg.mode == "template" and r.needs_human
    assert calls == []
    svc.triage(make_req("Max_Orlov оск родни"))
    assert calls  # an ordinary ticket does use the model


def test_check_draft_rejects_invented_contacts(rulebook):
    for t in ("Пишите на help@example.com.", "Звоните +7 999 123 45 67.", "Наш канал @grand_support_bot.", "Откройте /report в игре."):
        assert not check_draft(t, [], rulebook).ok, t


# ------------------------------------------------------------------------------------------------ 7. place-bound rules
KIDNAP = "Arman_Petrov похитил меня и связал верёвкой{place}"


def test_green_zone_rule_needs_the_green_zone(service):
    in_gz, _ = run(service, KIDNAP.format(place=" в Зелёной Зоне"), kind="complaint", att=1)
    nowhere, _ = run(service, KIDNAP.format(place=""), kind="complaint", att=1)
    forest, _ = run(service, KIDNAP.format(place=" в лесу"), kind="complaint", att=1)
    outside, _ = run(service, KIDNAP.format(place=" вне зелёной зоны"), kind="complaint", att=1)
    assert "GZ 1.2" in in_gz.draft_reply.cited
    for r in (nowhere, forest, outside):
        assert not any(c.startswith("GZ") for c in r.draft_reply.cited), r.draft_reply.cited
    assert "Зелёной Зоне или у дома" in nowhere.draft_reply.text  # the draft asks for the place instead of assuming one


def test_player_typed_green_zone_rule_number_is_not_trusted_without_the_place(service):
    r, _ = run(service, "Arman_Petrov связал меня верёвкой в лесу, GZ 1.2", kind="complaint", att=1)
    assert not any(c.startswith("GZ") for c in r.draft_reply.cited)
    assert r.extracted.rule_refs_unverified == [] and "GZ 1.2" in " ".join(r.reasons)


def test_killing_report_in_green_zone_cites_the_green_zone_rule(service):
    r, _ = run(service, "Меня убили в зелёной зоне", kind="report")
    assert r.category == "report.complaint" and "GZ 1.1" in r.draft_reply.cited and r.needs_human


def test_killing_report_without_a_place_does_not_cite_the_green_zone(service):
    r, _ = run(service, "Меня убили на улице", kind="report")
    assert not any(c.startswith("GZ") for c in r.draft_reply.cited)


# ------------------------------------------------------------------------------------------------ 8. terms in questions
@pytest.mark.parametrize("text,term", [("Что такое DM?", "DM"), ("что значит дб", "DB"), ("Что такое NonRP?", "NonRP"), ("dm это что?", "DM"), ("What is TK?", "TK")])
def test_questions_about_game_terms_are_answered_from_the_term_section(service, text, term):
    r, _ = run(service, text)
    assert r.category == "report.question" and r.draft_reply.terms == [term], (text, r.category, r.draft_reply.terms)
    assert r.draft_reply.cited == [] and not r.needs_human  # a term has no numbered point: it is NOT cited with an invented number
    assert term in r.draft_reply.text


# ------------------------------------------------------------------------------------------------ 10. evidence wording
def test_no_evidence_statement_is_not_evidence(service):
    r, _ = run(service, "Arman_Petrov оскорбил меня. Доказательств нет.", kind="complaint", att=0)
    assert r.extracted.has_evidence is False and r.extracted.evidence_status == "stated_absent" and r.extracted.evidence_verified is False
    assert r.category == "complaint.invalid"


@pytest.mark.parametrize("text", [
    "Arman_Petrov оскорбил меня, скринов нет", "Arman_Petrov оскорбил меня. Видео нет.", "Arman_Petrov insulted me, I have no proof",
    "Arman_Petrov me insultó, no tengo pruebas", "Arman_Petrov меня оскорбил, доказательств нет",
])
def test_negated_evidence_in_several_forms_and_languages(service, text):
    r, _ = run(service, text, kind="complaint", att=0)
    assert r.extracted.has_evidence is False and r.extracted.evidence_status == "stated_absent", text


@pytest.mark.parametrize("text,status", [
    ("Arman_Petrov оскорбил меня, скрины есть", "claimed_only"), ("Arman_Petrov оскорбил меня, видео нет, но скрины есть", "claimed_only"),
    ("Какие доказательства нужны для жалобы?", "asked_about"),
])
def test_claimed_and_asked_evidence_are_distinguished(service, text, status):
    r, _ = run(service, text, kind="complaint" if "оскорбил" in text else "report", att=0)
    assert r.extracted.evidence_status == status, (text, r.extracted.evidence_status)
    assert r.extracted.has_evidence is (status == "claimed_only") and r.extracted.evidence_verified is False


def test_attachments_win_over_the_words_but_the_conflict_is_reported(service):
    r, _ = run(service, "Arman_Petrov оскорбил меня. Доказательств нет.", kind="complaint", att=2)
    assert r.extracted.has_evidence is True and r.extracted.evidence_status == "attached" and r.extracted.evidence_verified is False
    assert any("доказательств нет, но вложений" in x for x in r.reasons)


def test_the_draft_never_claims_attachments_were_viewed(service):
    r, _ = run(service, "Arman_Petrov оскорбил меня, скрины есть", kind="complaint", att=2)
    assert "не просматриваем" in r.draft_reply.text


# ------------------------------------------------------------------------------------------------ 11. honest handoff wording
CLAIMS_TRANSFER = re.compile(
    r"передан[аоы]?\b|передали|переда[лё]м|have\s+passed|has\s+been\s+passed|passed\s+(?:it|your)|forwarded|encaminhamos|encaminhou|foi\s+encaminhad|hemos\s+pasado|se\s+ha\s+pasado|"
    r"жібердік|жіберілді|жібердік|yuborildi|yubordik|юбор", re.I)


def test_no_draft_claims_a_transfer_that_did_not_happen():
    from app.phrases import P

    for lang, T in P.items():
        for key, value in T.items():
            assert not CLAIMS_TRANSFER.search(value), (lang, key, value)


@pytest.mark.parametrize("text", [
    "Жалоба передана администратору.", "Мы передали ваше обращение сотруднику.", "Your complaint has been forwarded to an administrator.",
    "We passed it to a staff member.", "Hemos pasado su queja a un administrador.", "Encaminhamos sua mensagem à equipe.", "Shikoyatingiz administratorga yuborildi.",
    "Шағымыңыз әкімшіге жіберілді.",
])
def test_guard_rejects_claims_that_a_ticket_was_passed_on(rulebook, text):
    assert not check_draft(text, [], rulebook).ok, text


def test_llm_text_claiming_a_transfer_falls_back_to_the_template(rulebook):
    r, dbg = svc_llm(rulebook, "Здравствуйте! Ваша жалоба передана администратору.").triage(make_req("Max_Orlov оск родни"))
    assert dbg.mode == "template" and not CLAIMS_TRANSFER.search(r.draft_reply.text)


def test_demo_tickets_have_no_false_transfer_claims(service):
    for line in open("tests/fixtures/demo_tickets.jsonl", encoding="utf-8"):
        r, _ = service.triage(make_req(json.loads(line)["messages"][0]["text"], kind=json.loads(line)["kind"], att=1))
        assert not CLAIMS_TRANSFER.search(r.draft_reply.text), r.draft_reply.text


# ------------------------------------------------------------------------------------------------ 12. no refusal for critical topics
@pytest.mark.parametrize("text", [
    "Он угрожает убить меня в реальной жизни.", "Угрожает найти меня в реале", "Arman_Petrov взломал мой аккаунт", "Я оплатил донат, а игрок меня обманул",
])
def test_critical_complaint_without_attachments_is_not_refused(service, text):
    r, _ = run(service, text, kind="complaint", att=0)
    assert r.category != "complaint.invalid" and r.needs_human and r.priority == "p0"
    assert not any(x in r.draft_reply.text for x in ("нельзя принять", "Подайте жалобу заново"))
    if "угрожа" in text.lower():
        assert "экстренн" in r.draft_reply.text


# ------------------------------------------------------------------------------------------------ 13. languages
@pytest.mark.parametrize("text,lang", [
    ("Аккаунтимни бузишди", "uz"), ("Менга ёрдам беринг", "uz"), ("Меня оскорбил игрок Қайрат, скриншоты приложены", "ru"),
    ("Игрок Сәкен меня убил, скрины есть", "ru"), ("Он сказал «көмектесіңіз» и убил меня", "ru"), ("Менің аккаунтыма кіре алмаймын", "kk"),
    ("Мені қорлады, скриншот бар", "kk"), ("Помогите, застрял в текстурах", "ru"),
])
def test_language_detection_ignores_names_and_handles_uzbek_cyrillic(text, lang):
    assert detect_lang(text) == lang


def test_game_symbols_are_still_normalised_with_the_data_derived_table():
    from app.textutil import normalize_game_chars

    assert normalize_game_chars("Сјлеметсіз бе, кґмек беріңіз, шаєым, ѕ ќ ў ї")[0].startswith("Сәлеметсіз бе, көмек беріңіз, шағым")
    assert detect_lang("Сјлеметсіз, шаєымым бар") == "kk"
    assert detect_lang("Менга ўйинда муаммо бор, ёрдам беринг") == "uz"  # ў here is a real Uzbek letter, not the game's look-alike of ұ


# ------------------------------------------------------------------------------------------------ 14/15. dedup
def _dsvc(rulebook, path=None):
    return TriageService(load_settings(), rules=rulebook, dedup=DedupIndex(path))


def _send(svc, text, ts, rep="r1", nick="Max_Orlov", att=1):
    return svc.triage(make_req(text, reporter_id=rep, created_at=ts, reported_nick=nick, att=att))[0]


def test_different_places_are_different_events(rulebook):
    svc = _dsvc(rulebook)
    assert not _send(svc, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:00:00").is_spam_or_duplicate
    r = _send(svc, "Max_Orlov оскорбил меня в больнице", "2026-10-30T10:20:00")
    assert not r.is_spam_or_duplicate and not any("Дубликат" in x for x in r.reasons)


def test_five_paraphrases_of_one_event_are_recognised(rulebook):
    svc = _dsvc(rulebook)
    texts = ["Max_Orlov оскорбил меня у банка", "У банка Max_Orlov меня оскорблял", "Оскорбление у банка от Max_Orlov",
             "Max_Orlov у банка обзывал и оскорбил меня", "Меня у банка оскорбил игрок Max_Orlov"]
    results = [_send(svc, t, f"2026-10-30T10:{i * 5:02d}:00") for i, t in enumerate(texts)]
    assert results[0].is_spam_or_duplicate is False
    assert sum(r.is_spam_or_duplicate for r in results[1:]) >= 3
    assert all(any("овтор" in x or "Дубликат" in x for x in r.reasons) for r in results[1:])


def test_several_different_events_in_one_hour_are_not_merged(rulebook):
    svc = _dsvc(rulebook)
    rs = [_send(svc, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:00:00"),
          _send(svc, "Max_Orlov оскорбил меня в больнице в 11:30", "2026-10-30T10:10:00"),
          _send(svc, "Max_Orlov оскорбил меня на рынке", "2026-10-30T10:20:00"),
          _send(svc, "Max_Orlov использует чит и летает", "2026-10-30T10:30:00")]
    assert not any(r.is_spam_or_duplicate for r in rs)


def test_other_players_and_authors_are_never_merged(rulebook):
    svc = _dsvc(rulebook)
    _send(svc, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:00:00")
    assert not _send(svc, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:01:00", nick="Dan_Fox").is_spam_or_duplicate
    assert not _send(svc, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:02:00", rep="r2").is_spam_or_duplicate


def test_dedup_history_survives_a_restart(rulebook, tmp_path):
    db = tmp_path / "dedup.db"
    first = _dsvc(rulebook, db)
    assert not _send(first, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:00:00").is_spam_or_duplicate
    assert _send(first, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:05:00").is_spam_or_duplicate
    second = _dsvc(rulebook, db)  # a new process with the same database
    assert second.dedup.size() == 2
    assert _send(second, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:10:00").is_spam_or_duplicate
    assert not _dsvc(rulebook)  .dedup.size()  # without a path nothing is persisted


def test_dedup_window_memory_and_order_independence(rulebook, tmp_path):
    from app import dedup as d

    svc = _dsvc(rulebook, tmp_path / "w.db")
    _send(svc, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:00:00")
    assert not _send(svc, "Max_Orlov оскорбил меня у банка", "2026-10-30T12:00:00").is_spam_or_duplicate  # outside the 60-minute window
    # bounded memory: far more entries than the per-author cap
    for i in range(d.MAX_PER_REPORTER + 20):
        _send(svc, f"Max_Orlov оскорбил меня номер {i} у банка", f"2026-10-31T10:{i % 60:02d}:00", rep="busy")
    assert len(svc.dedup._seen["busy"]) <= d.MAX_PER_REPORTER
    # an earlier event that ARRIVES later is still compared by absolute time difference
    s2 = _dsvc(rulebook)
    _send(s2, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:30:00")
    assert _send(s2, "Max_Orlov оскорбил меня у банка", "2026-10-30T10:10:00").is_spam_or_duplicate


def test_batch_result_does_not_depend_on_file_order(rulebook, tmp_path):
    from app.cli import predict

    recs = [{"ticket_id": f"t{i}", "kind": "complaint", "created_at": f"2026-10-30T10:{i * 4:02d}:00", "reporter_id": "r1", "reported_nick": "Max_Orlov",
             "messages": [{"role": "player", "text": "Max_Orlov оскорбил меня у банка", "attachments": 1}]} for i in range(4)]

    def run_file(order):
        p = tmp_path / f"in{''.join(map(str, order))}.jsonl"
        p.write_text("\n".join(json.dumps(recs[i]) for i in order), encoding="utf-8")
        o = tmp_path / f"out{''.join(map(str, order))}.jsonl"
        predict(str(p), str(o), service=TriageService(load_settings(), rules=rulebook, dedup=DedupIndex()))
        return {json.loads(l)["ticket_id"]: json.loads(l)["is_spam_or_duplicate"] for l in o.read_text(encoding="utf-8").splitlines()}

    assert run_file([0, 1, 2, 3]) == run_file([3, 1, 0, 2])


# ------------------------------------------------------------------------------------------------ 16. ads
@pytest.mark.parametrize("text,spam", [
    ("Здесь рекламируют казино, проверьте", False), ("Игрок пишет в чате: заходи на t.me/xyz за донатом", False), ("Можно ли рекламировать казино на сервере?", False),
    ("В чате постоянно реклама казино, примите меры", False),
    ("Заходи на t.me/casinoplus бесплатный донат раздача", True), ("Подписывайся на мой канал, промокод BONUS", True),
])
def test_complaints_and_questions_about_ads_are_not_spam(service, text, spam):
    r, _ = run(service, text)
    assert r.is_spam_or_duplicate is spam, (text, r.reasons)


# ------------------------------------------------------------------------------------------------ 20. input boundaries
def test_accused_players_comment_is_not_the_complainants_words(service):
    from app.schemas import Message, TriageRequest

    req = TriageRequest(kind="complaint", messages=[
        Message(role="player", text="Max_Orlov оскорбил меня в чате", attachments=1),
        Message(role="reported_player", text="не согласен с жалобой, я оплатил донат и меня взломали", attachments=0),
        Message(role="admin", text="бан выдан, возврат невозможен", attachments=0),
    ])
    r, _ = service.triage(req)
    assert r.category == "complaint.insult" and r.priority == "p2"      # nothing from the accused or the admin leaks into the decision
    assert "Дополнение" not in r.draft_reply.text and not any("апелляц" in x.lower() for x in r.reasons)


def test_player_typed_rule_numbers_are_claims_not_facts(service):
    r, _ = run(service, "Max_Orlov оскорбил меня, нарушен пункт 5.2 и 99.99", kind="complaint", att=1)
    assert "5.2" not in r.draft_reply.cited and "99.99" not in r.draft_reply.text
    assert "99.99" in r.extracted.rule_refs_unverified


# ------------------------------------------------------------------------------------------------ 17. LLM budget
def _calls_handler(log, status=200, body=None):
    def handler(req):
        log.append(json.loads(req.content))
        if status != 200:
            return httpx.Response(status, json=body or {})
        return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps({"text": "Здравствуйте! Сообщение получено. Опишите, когда это произошло?"})}],
                                         "usage": {"input_tokens": 200, "output_tokens": 30}})

    return handler


def _client(handler, **cfg):
    s = dataclasses.replace(load_settings(), llm_api_key="k", **cfg)
    return s, LLMClient(s, transport=httpx.MockTransport(handler))


def _analysis():
    from app.classify import analyze

    return analyze(make_req("Max_Orlov оск родни"))


def test_retry_only_for_a_parameter_compatibility_400():
    log: list = []
    s, c = _client(_calls_handler(log, 400, {"error": {"message": "invalid_request_error: messages.0.content: too long"}}))
    res = c.draft(_analysis(), [], "ru", True, "t")
    assert len(log) == 1 and res.error == "http 400"
    log.clear()
    s, c = _client(_calls_handler(log, 400, {"error": {"message": "thinking.type: not supported"}}))
    c.draft(_analysis(), [], "ru", True, "t")
    assert len(log) == 2 and "thinking" not in log[1] and log[1]["max_tokens"] <= log[0]["max_tokens"]


def test_5xx_and_429_are_not_retried():
    for st in (429, 500, 529):
        log: list = []
        _, c = _client(_calls_handler(log, st))
        assert c.draft(_analysis(), [], "ru", True, "t").error == f"http {st}" and len(log) == 1


def test_no_retry_when_the_time_budget_is_gone():
    log: list = []
    _, c = _client(_calls_handler(log, 400, {"error": {"message": "thinking not supported"}}), llm_total_budget_s=0.5)
    res = c.draft(_analysis(), [], "ru", True, "t", time_budget_s=0.5)
    assert len(log) <= 1 and res.text is None
    log.clear()
    _, c = _client(_calls_handler(log))
    res = c.draft(_analysis(), [], "ru", True, "t", time_budget_s=0.05)
    assert log == [] and "time budget" in (res.error or "")


def test_per_ticket_cost_ceiling_limits_max_tokens_or_blocks_the_call():
    log: list = []
    # the ceiling arithmetic itself: 1000 input tokens at $1/M cost $0.001, the rest of the $0.005 buys 400 output tokens at $10/M
    s, c = _client(_calls_handler(log), llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=10.0, llm_budget_per_ticket_usd=0.005, llm_max_output_tokens=450)
    assert c._max_tokens_within_budget(1000) == 400
    assert c._max_tokens_within_budget(100) == 450          # plenty of room: the configured maximum
    assert c._max_tokens_within_budget(4900) is None        # not even a minimal answer fits
    s, c = _client(_calls_handler(log), llm_price_in_per_mtok=0.1, llm_price_out_per_mtok=2.0, llm_budget_per_ticket_usd=0.0005)
    c.draft(_analysis(), [], "ru", True, "t")
    assert log and log[0]["max_tokens"] < s.llm_max_output_tokens
    log.clear()
    s, c = _client(_calls_handler(log), llm_price_in_per_mtok=1000.0, llm_price_out_per_mtok=1000.0, llm_budget_per_ticket_usd=0.00001)
    res = c.draft(_analysis(), [], "ru", True, "t")
    assert log == [] and "budget" in res.error and c.cost_usd(res) == 0.0  # nothing was sent, nothing was spent


def test_cost_is_unknown_not_zero_without_prices():
    log: list = []
    _, c = _client(_calls_handler(log), llm_price_in_per_mtok=0.0, llm_price_out_per_mtok=0.0)
    res = c.draft(_analysis(), [], "ru", True, "t")
    assert res.calls == 1 and c.cost_usd(res) is None


def test_settings_are_validated(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_IN_PER_MTOK", "-3")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "999999")
    monkeypatch.setenv("LLM_TIMEOUT_S", "900")
    monkeypatch.setenv("LLM_EFFORT", "extreme")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    s = load_settings()
    assert s.llm_price_in_per_mtok == 0.0 and s.llm_max_output_tokens <= 2000 and s.llm_timeout_s <= 20 and s.llm_effort == "low"
    assert len(s.config_warnings) >= 4 and any("UNKNOWN" in w for w in s.config_warnings)
