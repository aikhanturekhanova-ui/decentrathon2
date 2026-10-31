"""Regression tests for the fourth review: account-takeover wordings, the order of events around a Green Zone, one rule = one action,
the official zone list, the LLM whitelist and the hard deadline, payments vs. in-game fines, the accused player, the appeal draft.
Every group has positive AND negative examples and permutations, not only the sentences that exposed the problem."""
from __future__ import annotations

import dataclasses
import http.server
import json
import threading
import time

import httpx
import pytest

from app.classify import analyze
from app.config import load_settings
from app.dedup import DedupIndex
from app.guard import check_free_text, check_llm_sentences
from app.llm import LLMClient
from app.pipeline import TriageService
from tests.conftest import make_req


def run(service, text, kind="report", att=0, **extra):
    return service.triage(make_req(text, kind=kind, att=att, **extra))


# --------------------------------------------------------------------------- independent expectation set (also used by coverage_report)
EXPECT = [json.loads(line) for line in open("tests/fixtures/sensitive_expectations.jsonl", encoding="utf-8") if line.strip()]


def _detected(flags: set) -> str:
    if flags & {"hack", "access"}:
        return "account"
    if "payment" in flags:
        return "payment"
    if "threat_outside_game" in flags:
        return "threat"
    if "appeal" in flags:
        return "appeal"
    return "none"


@pytest.mark.parametrize("row", EXPECT, ids=[r["text"][:40] for r in EXPECT])
def test_detection_against_the_expectation_set(row):
    flags = analyze(make_req(row["text"], kind=row["kind"], att=1)).flags
    assert _detected(flags) == row["expect"] or (row["expect"] != "none" and {"account": flags & {"hack", "access"}, "payment": "payment" in flags,
                                                                               "threat": "threat_outside_game" in flags, "appeal": "appeal" in flags}[row["expect"]]), (row, sorted(flags))


@pytest.mark.parametrize("row", [r for r in EXPECT if r["expect"] in {"account", "payment", "threat"}], ids=lambda r: r["text"][:40])
def test_takeover_payment_threat_end_to_end_report_account_p0_human(service, row):
    r, _ = run(service, row["text"], kind=row["kind"], att=1)
    assert r.needs_human and r.priority == "p0", (row["text"], r.category, r.priority)
    if row["kind"] == "report" and row["expect"] == "account":
        assert r.category == "report.account"
    if row["kind"] == "report" and row["expect"] == "payment":
        assert r.category == "report.payment"


@pytest.mark.parametrize("row", [r for r in EXPECT if r["expect"] == "none"], ids=lambda r: r["text"][:40])
def test_false_critical_signals_are_absent(service, row):
    flags = analyze(make_req(row["text"], kind=row["kind"], att=1)).flags
    assert not (flags & {"hack", "access", "payment", "threat_outside_game", "appeal", "mass_bug"}), (row["text"], sorted(flags))


def test_safe_question_after_a_takeover_does_not_cancel_the_escalation(service):
    for text in ("Что такое DM? Мою учётку угнали", "Мою учётку угнали. Что такое DM?", "DM деген не? Аккаунтымды біреу бұзды"):
        r, _ = run(service, text)
        assert r.needs_human and r.priority == "p0" and r.category == "report.account", text
    r, _ = run(service, "Что такое DM?")
    assert r.needs_human is False and r.draft_reply.terms == ["DM"]


# --------------------------------------------------------------------------- Green Zone: order of events, one rule = one action, official list
ARM = "Arman_Petrov"


@pytest.mark.parametrize("text", [
    "Мы вышли из Зелёной зоны, после чего он меня похитил", "Мы вышли из ЗЗ, после чего он меня похитил", "Вышел из ЗЗ и меня похитили",
    "Мы покинули зону и он связал меня верёвкой", "Мы уехали из ЗЗ, потом {a} меня связал", "Сначала были в ЗЗ, потом {a} меня похитил",
    "{a} связал меня вне ЗЗ", "{a} связал меня не в ЗЗ", "После выхода из ЗЗ {a} связал меня",
])
def test_zone_named_but_left_before_the_event_is_not_cited(service, text):
    r, _ = run(service, text.format(a=ARM), kind="complaint", att=1)
    assert not any(c.startswith("GZ") for c in r.draft_reply.cited), (text, r.draft_reply.cited)


@pytest.mark.parametrize("text", ["{a} связал меня в ЗЗ", "{a} похитил меня в Зелёной Зоне", "В ЗЗ {a} связал меня верёвкой", "{a} ограбил меня в зелёной зоне"])
def test_zone_and_action_together_cite_the_criminal_point_only(service, text):
    r, _ = run(service, text.format(a=ARM), kind="complaint", att=1)
    assert r.draft_reply.cited == ["GZ 1.2"], (text, r.draft_reply.cited)


def test_each_point_needs_its_own_action_and_a_denied_action_is_not_an_action(service):
    c = lambda t: run(service, t, kind="complaint", att=1)[0].draft_reply.cited  # noqa: E731
    assert c(f"{ARM} связал меня в ЗЗ, убийства не было") == ["GZ 1.2"]
    assert c(f"{ARM} убил меня в ЗЗ, связывания не было") == ["GZ 1.1"]
    assert c(f"{ARM} связал и убил меня в ЗЗ") == ["GZ 1.2", "GZ 1.1"]
    assert c(f"{ARM} связал меня в ЗЗ, никто не убил") == ["GZ 1.2"]


@pytest.mark.parametrize("text,cited", [
    ("Он связал меня у больницы г. Арзамас", ["GZ 1.2"]), ("Он связал меня у больницы г. Москва", []), ("Он связал меня у больницы", []),
    ("Он убил меня у полиции г. Южный", ["GZ 1.1"]), ("Он убил меня у полиции", []), ("Он связал меня у банка", ["GZ 1.2"]), ("Он связал меня на автобусной остановке", ["GZ 1.2"]), ("Он связал меня на остановке", []),
    ("Он связал меня в лесу", []), ("Он связал меня на улице", []),
])
def test_objects_of_the_official_list_count_as_green_zones_only_with_their_city(service, text, cited):
    assert run(service, text, kind="complaint", att=1)[0].draft_reply.cited == cited, text


def test_unknown_city_gets_a_targeted_question_not_the_generic_one(service):
    d = run(service, "Он связал меня у больницы", kind="complaint", att=1)[0].draft_reply.text
    assert "город и объект" in d and "Уточните, где это произошло" not in d
    forest = run(service, f"{ARM} похитил меня в лесу и связал верёвкой", kind="complaint", att=1)[0].draft_reply.text
    assert "Уточните, где это произошло" not in forest and "город и объект" not in forest   # the place is stated and it is not a listed object: nothing to ask
    nowhere = run(service, f"{ARM} связал меня верёвкой", kind="complaint", att=1)[0].draft_reply.text
    assert "Уточните, где это произошло" in nowhere


@pytest.mark.skipif(not __import__("pathlib").Path("data/handout/rules.json").is_file(), reason="organizers' rules.json is not in the project")
def test_the_real_zone_list_is_parsed_from_rules_json():
    from app.knowledge import RuleBook
    from app.rag import zone_facts, zone_places

    rb = RuleBook.load("data/handout/rules.json")
    names = [p["name"] for p in zone_places(rb)]
    assert "Больница г. Арзамас" in names and len(names) > 30
    assert zone_facts("он связал меня у больницы г. арзамас", rb).established
    assert not zone_facts("он связал меня у больницы г. москва", rb).established


# --------------------------------------------------------------------------- LLM: whitelist of sentence types, question marks, real deadline
def svc_llm(rulebook, answer, **cfg):
    def handler(req):
        return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps({"text": answer, "cited": []})}], "usage": {"input_tokens": 100, "output_tokens": 40}})

    s = dataclasses.replace(load_settings(), llm_api_key="k", llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=5.0, **cfg)
    return TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(handler)), dedup=DedupIndex())


@pytest.mark.parametrize("answer", [
    "Здравствуйте! Рассмотрение займёт пару часов. Уточните сервер.",
    "Здравствуйте! Игрок потеряет доступ к игре навсегда. Уточните сервер.",
    "Здравствуйте! Решение уже принято в вашу пользу. Уточните сервер.",
    "Здравствуйте! Вам обеспечено возмещение?", "Здравствуйте! Вам всё вернут, не переживайте?", "Здравствуйте! Вашу проблему уже решили.",
    "Hello! The review will take a couple of hours. Please tell us the server.", "Hello! The player will lose access forever. Please tell us the server.",
    "Hello! The decision has already been made in your favour.", "Hola! La decisión ya fue tomada a su favor.", "Olá! O jogador perderá o acesso para sempre.",
    "Сәлеметсіз бе! Шешім сіздің пайдаңызға қабылданды.", "Assalomu alaykum! Qaror allaqachon sizning foydangizga qabul qilindi.",
])
def test_paraphrased_promises_and_fake_outcomes_are_rejected_through_the_whole_service(rulebook, answer):
    r, dbg = svc_llm(rulebook, answer).triage(make_req("Arman_Petrov оскорбил меня", att=1))
    assert dbg.mode == "template" and dbg.llm_error, (answer, dbg.llm_error)
    assert "2.1" in r.draft_reply.cited and any("Черновик модели отклонён" in x for x in r.reasons)


@pytest.mark.parametrize("lang,answer", [
    ("ru", "Здравствуйте! Мы получили ваше обращение. Укажите, пожалуйста, сервер и время. Администратор рассмотрит жалобу. Мы не можем обещать результат."),
    ("en", "Hello! We received your report. Please tell us the server and the time. An administrator will review it. We cannot promise the outcome."),
    ("es", "¡Hola! Hemos recibido su mensaje. Indique el servidor y la hora. Un administrador revisará la queja. No podemos prometer el resultado."),
    ("pt", "Olá! Recebemos sua mensagem. Informe o servidor e o horário. Um administrador analisará a reclamação. Não podemos prometer o resultado."),
    ("uz", "Assalomu alaykum! Murojaatingizni oldik. Server va vaqtni yozing. Administrator shikoyatni ko'rib chiqadi. Natijani oldindan va'da qila olmaymiz."),
    ("kk", "Сәлеметсіз бе! Өтінішіңізді алдық. Сервер мен уақытты көрсетіңіз. Әкімші шағымды қарайды. Нәтижені алдын ала уәде ете алмаймыз."),
])
def test_harmless_sentence_types_are_accepted_in_every_language(lang, answer):
    # the closed vocabulary is per draft language (guard.check_llm_sentences), so the language is part of the check
    assert check_llm_sentences(answer, lang) == [], answer
    assert check_free_text(answer, {"1"}) == [], answer


@pytest.mark.parametrize("text", ["Вам обеспечено возмещение?", "Будет ли вам обеспечено возмещение?", "Do you want your refund guaranteed?"])
def test_a_question_mark_does_not_exempt_a_sentence_from_the_outcome_check(text):
    assert check_free_text(text), text


class _Drip(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", "100000")
        self.end_headers()
        try:
            for _ in range(40):  # a byte every 0.25 s: every single wait is short, the whole body would take 10 s
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.25)
        except OSError:
            pass

    def log_message(self, *a):  # noqa: D401
        pass


def test_a_server_that_drips_bytes_cannot_exceed_the_total_budget(rulebook):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Drip)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        s = dataclasses.replace(load_settings(), llm_api_key="k", llm_base_url=f"http://127.0.0.1:{server.server_address[1]}",
                                llm_total_budget_s=1.5, llm_timeout_s=3.5)
        svc = TriageService(s, rules=rulebook, llm=LLMClient(s), dedup=DedupIndex())
        t = time.perf_counter()
        r, dbg = svc.triage(make_req("Arman_Petrov оскорбил меня", att=1))
        took = time.perf_counter() - t
        assert dbg.mode == "template" and dbg.llm_error in {"deadline exceeded", "timeout"}, dbg.llm_error
        assert took < 2.3, f"took {took:.2f}s for a 1.5 s budget"
        assert r.draft_reply.cited == ["2.1"]
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- payments vs in-game payments, negated requests
@pytest.mark.parametrize("text", ["Как оплатить штраф?", "Как оплатить штрафы?", "Я оплатил штраф за нарушение ПДД", "Оплатил налог на дом, как проверить?", "Как оплатить аренду склада?"])
def test_in_game_fines_taxes_and_rent_are_not_payments(service, text):
    r, _ = run(service, text)
    assert r.category != "report.payment" and r.priority != "p0", (text, r.category)


@pytest.mark.parametrize("text", ["Я не прошу вернуть деньги, просто скажите, можно ли использовать читы?", "Не требую возврат, скажите можно ли оскорблять в чате",
                                  "I am not asking for a refund, just tell me if cheats are allowed"])
def test_a_denied_refund_request_is_not_a_payment_ticket(service, text):
    r, _ = run(service, text)
    assert r.category != "report.payment" and r.priority != "p0", (text, r.category)


def test_a_real_refund_request_still_is_one_even_after_a_negated_one(service):
    r, _ = run(service, "Я не прошу компенсацию, но верните 500 рублей за покупку")
    assert r.category == "report.payment" and r.priority == "p0"


# --------------------------------------------------------------------------- the accused player
@pytest.mark.parametrize("text,nick", [
    ("Я, Dias_Volkov, жалуюсь на Arman_Petrov: он оскорбил меня", "Arman_Petrov"),
    ("Жалуюсь на Arman_Petrov. Я Dias_Volkov", "Arman_Petrov"),
    ("Мой ник Dias_Volkov, Arman_Petrov оскорбил меня", "Arman_Petrov"),
    ("Жалоба от Dias_Volkov на игрока Arman_Petrov, админ Nina_Lebedev выдал мне мут", "Arman_Petrov"),
    ("Arman_Petrov оскорбил меня", "Arman_Petrov"),
    ("Игрок Arman_Petrov убил меня", "Arman_Petrov"),
    ("Dias_Volkov и Arman_Petrov оба там были", None),
    ("Я Dias_Volkov", None),
])
def test_the_accused_is_extracted_not_the_complainant_or_an_admin(service, text, nick):
    assert run(service, text, kind="complaint", att=1)[0].extracted.nickname == nick, text


def test_the_game_field_with_the_accused_nick_wins(service):
    r, _ = run(service, "Я Dias_Volkov, меня оскорбили", kind="complaint", att=1, reported_nick="Arman_Petrov")
    assert r.extracted.nickname == "Arman_Petrov"


# --------------------------------------------------------------------------- Kazakh meaning, appeal draft, clearly formulated questions
@pytest.mark.parametrize("text", ["Ойыншы менің ақшамды алдап алды", "Ойыншы мені алдады"])
def test_kazakh_scam_report_is_understood(service, text):
    r, _ = run(service, text)
    assert r.lang == "kk" and r.category == "report.complaint" and r.needs_human and "4.1" in r.draft_reply.cited


def test_an_appeal_that_arrives_as_a_report_gets_an_appeal_draft_and_p2(service):
    r, _ = run(service, "Аккаунт заблокирован, хочу подать апелляцию")
    assert r.needs_human and r.priority == "p2"
    assert "готового ответа" not in r.draft_reply.text and "что именно вас интересует" not in r.draft_reply.text
    assert "ник администратора" in r.draft_reply.text and "скриншот наказания" in r.draft_reply.text


def test_a_clearly_formulated_question_is_not_asked_what_it_means(service):
    r, _ = run(service, "Как купить машину в автосалоне?")
    assert r.needs_human and "что именно вас интересует" not in r.draft_reply.text and "сервер" in r.draft_reply.text
    short, _ = run(service, "машина")
    assert "что именно" in short.draft_reply.text or short.category == "report.offtopic"
