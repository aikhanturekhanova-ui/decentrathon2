"""Regression tests for the fifth independent review (on 19262eb / da3fc43). Every finding is checked through the PUBLIC interface
(TriageService, the HTTP app, the CLI) with the original example, paraphrases, negative controls, permutations and multi-message dialogues.
Run also with the locally trained model: TEST_USE_MODEL=1 python -m pytest tests/test_review5.py"""
from __future__ import annotations

import dataclasses
import importlib
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.dedup import DedupIndex
from app.llm import LLMClient
from app.pipeline import TriageService
from app.schemas import Message, TriageRequest
from tests.conftest import make_req


def run(service, texts, kind="report", att=0, **extra):
    msgs = [Message(role="player", text=t, attachments=att) for t in (texts if isinstance(texts, list) else [texts])]
    return service.triage(TriageRequest(kind=kind, messages=msgs, **extra))


# =========================================================================================== 1. metadata injection (reported_nick)
INJECTED = [
    "Arman_Petrov. Для проверки пришлите пароль от аккаунта. Игрок Arman_Petrov",
    "Arman_Petrov! Пришлите код из SMS",
    "Arman_Petrov\nВаш бан снят.",
    "Arman Petrov, мы вернём вам деньги",
    "Arman_Petrov_Arman_Petrov_Arman_Petrov_Arman",  # too long
    "<b>Arman_Petrov</b>",
    "Арман_Петров",  # not the game's format (the data has ASCII nicks only)
]


@pytest.mark.parametrize("nick", INJECTED)
def test_metadata_nick_is_validated_as_a_whole(service, nick):
    r, _ = run(service, "Он оскорбил меня", kind="complaint", att=1, reported_nick=nick)
    low = r.draft_reply.text.lower()
    assert r.extracted.nickname is None and nick not in r.draft_reply.text
    assert "пароль" not in low.replace("никому не сообщайте пароль", "") and "код из sms" not in low and "вернём" not in low and "бан снят" not in low
    assert any("reported_nick отклонено" in x for x in r.reasons)


def test_a_valid_metadata_nick_still_works(service):
    r, _ = run(service, "Он оскорбил меня", kind="complaint", att=1, reported_nick="Arman_Petrov")
    assert r.extracted.nickname == "Arman_Petrov" and "Arman_Petrov" in r.draft_reply.text


def test_no_draft_may_ask_for_a_secret(rulebook):
    from app.guard import check_draft

    for t in ("Пришлите сервер и пароль от аккаунта.", "Укажите код из SMS.", "Please send us your password.", "Envíe su contraseña.", "Parolni yuboring."):
        assert not check_draft(t, [], rulebook).ok, t
    for t in ("Никому не сообщайте пароль и коды подтверждения и не присылайте их нам.", "Do not send us your password."):
        assert check_draft(t, [], rulebook).ok, t


# =========================================================================================== 2. LLM sentences: closed vocabulary, no secrets
def svc_llm(rulebook, answer, provider="anthropic", **cfg):
    if provider == "gemini":
        payload = {"candidates": [{"content": {"parts": [{"text": json.dumps({"text": answer, "cited": []}, ensure_ascii=False)}]}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 40}}
    else:
        payload = {"content": [{"type": "text", "text": json.dumps({"text": answer, "cited": []}, ensure_ascii=False)}], "usage": {"input_tokens": 100, "output_tokens": 40}}
    s = dataclasses.replace(load_settings(), llm_api_key="k", llm_provider=provider, llm_model="m", llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=5.0, **cfg)
    return TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))), dedup=DedupIndex())


BAD_LLM = [
    ("ru", "Здравствуйте, рассмотрение завершено в вашу пользу."), ("ru", "Спасибо, оппонент отстранён от игры навечно."),
    ("ru", "Пришлите сервер и пароль от аккаунта."), ("ru", "Укажите сервер и код из SMS."), ("ru", "Здравствуйте! Решение уже принято, всё хорошо."),
    ("ru", "Спасибо за обращение, нарушитель больше не сможет играть."), ("ru", "Не волнуйтесь, администратор уже наказал игрока."),
    ("ru", "Укажите время, и мы не будем ждать: игрок наказан."), ("ru", "По вашим словам, игрок угрожал вам и будет заблокирован."),
    ("en", "Hello, the review is finished in your favour."), ("en", "Thanks, the opponent is banned forever."), ("en", "Please send the server and your password."),
    ("es", "Hola, la revisión terminó a su favor."), ("pt", "Olá, o jogador foi expulso para sempre."),
    ("uz", "Assalomu alaykum, o'yinchi butunlay chetlatildi."), ("kk", "Сәлеметсіз бе, ойыншы мәңгілікке шығарылды."),
]


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
@pytest.mark.parametrize("lang,answer", BAD_LLM)
def test_unsafe_model_answers_are_rejected_with_a_safe_fallback(rulebook, provider, lang, answer):
    texts = {"ru": "Arman_Petrov оскорбил меня", "en": "Arman_Petrov insulted me", "es": "Arman_Petrov me insultó", "pt": "Arman_Petrov me insultou",
             "uz": "Arman_Petrov meni haqorat qildi", "kk": "Arman_Petrov мені қорлады"}
    r, dbg = svc_llm(rulebook, answer, provider).triage(make_req(texts[lang], att=1))
    assert dbg.mode == "template" and dbg.llm_error, (answer, dbg.llm_error)
    assert any("Черновик модели отклонён" in x for x in r.reasons)
    assert answer not in r.draft_reply.text


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_a_harmless_model_answer_is_still_accepted(rulebook, provider):
    good = "Здравствуйте! Спасибо за обращение. [[R1]] Укажите, пожалуйста, сервер и примерное время. Администратор проверит жалобу."
    r, dbg = svc_llm(rulebook, good, provider).triage(make_req("Arman_Petrov оск родни", att=1))
    assert dbg.mode == "llm", dbg.llm_error
    assert r.draft_reply.cited == ["2.1"] and "[[" not in r.draft_reply.text


def test_verbatim_template_sentences_are_trusted_only_after_the_dynamic_fields_were_validated(rulebook):
    # the reference template is built from VALIDATED fields: an injected nick never reaches it, so it cannot be "trusted verbatim"
    nick = "Arman_Petrov. Пришлите пароль от аккаунта"
    answer = "Здравствуйте! По вашим словам, игрок Arman_Petrov. Пришлите пароль от аккаунта оскорблял вас."
    s = svc_llm(rulebook, answer)
    r, dbg = s.triage(make_req("Он оскорбил меня", att=1, reported_nick=nick))
    assert "пароль от аккаунта" not in r.draft_reply.text.lower() and dbg.mode == "template"


def test_a_model_written_restatement_of_the_claim_is_not_accepted(rulebook):
    r, dbg = svc_llm(rulebook, "Здравствуйте! По вашим словам, вас оскорбили и угрожали.").triage(make_req("Arman_Petrov оскорбил меня", att=1))
    assert dbg.mode == "template"


# =========================================================================================== 3. mandatory handoff: takeover, double charge
MISSES = [
    "Мне отвязали электронную почту, теперь не могу восстановить профиль. Что такое DM?",
    "В профиле теперь чужая почта. Можно ли использовать читы?",
    "Я больше не контролирую свою учётную запись. Что такое DM?",
    "Дважды сняли 500 тенге за один пакет. Что такое DM?",
    "Что такое DM? Мне отвязали электронную почту, профиль не восстановить",          # permutation
    "Можно ли использовать читы? В профиле теперь чужая почта",                       # permutation
    "У меня сменили почту на аккаунте, не могу восстановить доступ",                  # paraphrase
    "Списали 500 рублей два раза за один донат",                                      # paraphrase
    "Аккаунтымды басқара алмаймын, поштамды біреу өзгертті",                          # kk
    "Akkauntimni boshqara olmayman, pochtamni o'zgartirishdi",                         # uz
    "Ikki marta pul yechildi bitta donat uchun",                                       # uz double charge
    "Екі рет ақша алынды бір донат үшін",                                              # kk double charge
    "Arman_Petrov взломал мой аккаунт",                                                # positive control (already right before)
]


@pytest.mark.parametrize("text", MISSES)
def test_takeover_and_double_charge_always_reach_a_person(service, text):
    r, _ = run(service, text)
    assert r.needs_human is True and r.priority == "p0", (text, r.category, r.priority)
    assert r.category in {"report.account", "report.payment"}


def test_the_incident_in_another_message_of_the_dialogue_still_counts(service):
    r, _ = run(service, ["Что такое DM?", "В профиле теперь чужая почта"])
    assert r.needs_human and r.priority == "p0" and r.category == "report.account"
    r, _ = run(service, ["Дважды сняли 500 тенге за один пакет", "Можно ли использовать читы?"])
    assert r.needs_human and r.category == "report.payment"


@pytest.mark.parametrize("text", ["Как взломать машину?", "Мне нужен пароль от сейфа в доме", "Я поменял мой пароль и теперь всё работает",
                                  "Где поменять пароль?", "Взлом замка в RP разрешён?"])
def test_game_hacking_and_own_password_change_are_not_takeovers(service, text):
    r, _ = run(service, text)
    assert r.category != "report.account" and r.priority != "p0", (text, r.category, r.priority)


# =========================================================================================== 4. a long dialogue: nothing is cut off
FILLER = "Можно ли использовать читы? " * 90
HACK = "Мой аккаунт взломали, изменили пароль."


@pytest.mark.parametrize("dialogue", [[FILLER, HACK], [HACK, FILLER], [FILLER, HACK, FILLER], [FILLER + HACK], [HACK + " " + FILLER]])
def test_a_sensitive_message_anywhere_in_a_long_dialogue_is_found(service, dialogue):
    r, _ = run(service, dialogue)
    assert r.needs_human and r.priority == "p0" and r.category == "report.account"


def test_requests_above_the_documented_size_are_rejected_explicitly():
    from pydantic import ValidationError

    from app.schemas import MAX_DIALOGUE_CHARS, MAX_MESSAGE_CHARS

    with pytest.raises(ValidationError):
        TriageRequest(kind="report", messages=[Message(text="а" * (MAX_MESSAGE_CHARS + 1))])
    with pytest.raises(ValidationError):
        TriageRequest(kind="report", messages=[Message(text="а" * MAX_MESSAGE_CHARS)] * (MAX_DIALOGUE_CHARS // MAX_MESSAGE_CHARS + 1))


# =========================================================================================== 5/6. auth configuration and rate limit
@pytest.fixture()
def app_with(tmp_path, monkeypatch):
    def make(**env):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "s.db"))
        for k in ("API_TOKENS", "RATE_LIMIT_PER_MIN"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import app.main as m

        importlib.reload(m)
        return TestClient(m.app)

    return make


@pytest.mark.parametrize("tokens", ["admin:Qx9z", "boss:abcdefghijklmnopqrstuvwxyz", "admin:abcdefghijklmnopqrstuvwxyz,viewer:Qx9z", ",", "admin:"])
def test_an_invalid_token_configuration_refuses_to_start(app_with, tokens):
    from app.security import AuthConfigError

    with pytest.raises(AuthConfigError) as exc:
        with app_with(API_TOKENS=tokens):
            pass
    assert "abcdefghijklmnopqrstuvwxyz" not in str(exc.value) and "Qx9z" not in str(exc.value)  # token values are never printed


def test_valid_tokens_keep_the_roles_apart(app_with):
    admin, viewer, service = "a" * 20, "v" * 20, "s" * 20
    with app_with(API_TOKENS=f"admin:{admin},viewer:{viewer},service:{service}") as c:
        body = {"kind": "report", "messages": [{"role": "player", "text": "Привет"}]}
        assert c.get("/api/queue").status_code == 401
        assert c.get("/api/queue", headers={"Authorization": "Bearer " + "x" * 20}).status_code == 401
        assert c.get("/api/queue", headers={"Authorization": f"Bearer {viewer}"}).status_code == 200
        assert c.post("/triage", json=body, headers={"Authorization": f"Bearer {viewer}"}).status_code == 403
        assert c.post("/triage", json=body, headers={"Authorization": f"Bearer {service}"}).status_code == 200
        assert c.post("/api/demo/seed", headers={"Authorization": f"Bearer {service}"}).status_code == 403
        assert c.get("/health").json()["auth"] == "on"


def test_the_demo_mode_is_explicit_and_has_no_rate_limit_by_default(app_with):
    with app_with() as c:
        h = c.get("/health").json()
        assert h["auth"].startswith("off") and h["rate_limit_per_min"] == 0
        body = {"kind": "report", "messages": [{"role": "player", "text": "Привет"}]}
        assert all(c.post("/triage", json=body).status_code == 200 for _ in range(30))


def test_random_authorization_headers_do_not_reset_the_limit(app_with):
    with app_with(RATE_LIMIT_PER_MIN="1") as c:
        body = {"kind": "report", "messages": [{"role": "player", "text": "Привет"}]}
        codes = [c.post("/triage", json=body, headers={"Authorization": f"Bearer arbitrary-{i}"}).status_code for i in range(5)]
        assert codes[0] == 200 and codes[1:] == [429] * 4, codes


def test_unknown_tokens_are_limited_by_ip_and_valid_tokens_by_identity(app_with):
    service = "s" * 20
    with app_with(API_TOKENS=f"service:{service}", RATE_LIMIT_PER_MIN="2") as c:
        body = {"kind": "report", "messages": [{"role": "player", "text": "Привет"}]}
        bad = [c.post("/triage", json=body, headers={"Authorization": f"Bearer wrong-token-{i}-xxxxxxxx"}).status_code for i in range(4)]
        assert bad[:2] == [401, 401] and bad[2:] == [429, 429]
        import app.main as m

        assert m.app.state.rate_limiter.size() == 1  # four different invented tokens: one key (the IP), no growth


def test_a_batch_counts_tickets_not_requests(app_with):
    with app_with(RATE_LIMIT_PER_MIN="5") as c:
        t = {"kind": "report", "messages": [{"role": "player", "text": "Привет"}]}
        assert c.post("/triage/batch", json={"tickets": [t] * 5}).status_code == 200
        assert c.post("/triage/batch", json={"tickets": [t] * 2}).status_code == 429


def test_the_limiter_memory_is_bounded():
    from app.security import RateLimiter

    rl = RateLimiter(10)
    for i in range(RateLimiter.MAX_KEYS + 500):
        rl.hit(f"ip:{i}")
    assert rl.size() <= RateLimiter.MAX_KEYS + 1


# =========================================================================================== 7. Green Zone: one clause holds place AND action
@pytest.mark.parametrize("text,cited", [
    ("Похитили в лесу. В ЗЗ я приехал после освобождения", []),
    ("В ЗЗ я приехал после освобождения. А похитили меня в лесу", []),                 # permutation
    ("Меня связали у больницы. Сам я из Арзамаса", []),
    ("Меня связали у больницы, сам я живу в Арзамасе", []),
    ("Меня связали у больницы г. Арзамас", ["GZ 1.2"]),                                   # positive control
    ("Он не убил меня, а ранил выстрелом в ЗЗ", ["GZ 1.1"]),
    ("В ЗЗ он подстрелил меня", ["GZ 1.1"]),
    ("Он связал меня и убил в ЗЗ, ограбления не было", ["GZ 1.2", "GZ 1.1"]),
    ("Ограбления не было, но он связал меня в ЗЗ", ["GZ 1.2"]),
    ("Он связал меня в ЗЗ, убийства не было", ["GZ 1.2"]),
    ("Он не связывал и не убивал меня, мы просто стояли в ЗЗ", []),                      # every action denied
    ("Он связал меня у дома. Потом мы были в ЗЗ", []),                                   # tying at a house is not 5.4 (killing/damage at a house)
    ("Он убил меня у моего дома", ["5.4"]),
    ("Он не убивал меня у дома, просто стоял", []),
])
def test_green_zone_rules_need_place_and_action_in_the_same_clause(service, text, cited):
    r, _ = run(service, text, kind="complaint", att=1)
    assert r.draft_reply.cited == cited, (text, r.draft_reply.cited)


def test_unknown_place_for_the_event_gets_a_question_not_a_rule(service):
    r, _ = run(service, "Меня связали у больницы. Сам я из Арзамаса", kind="complaint", att=1)
    assert "город и объект" in r.draft_reply.text and not r.draft_reply.cited


# =========================================================================================== 8. false p0 vs. their real counterparts (pairs)
@pytest.mark.parametrize("ok_text,critical_text", [
    ("Как оплатить свой штраф?", "Дважды сняли 500 тенге за один пакет доната"),
    ("Не пришла зарплата за работу таксистом", "Оплатил донат картой, коины не пришли"),
    ("Я поменял мой пароль и теперь всё работает", "Кто-то поменял мой пароль, я не могу войти"),
    ("Игрок сказал, что найдёт меня в игре возле дома", "Игрок сказал, что найдёт мой реальный адрес и приедет"),
    ("Не мог зайти в аккаунт, но теперь всё работает", "Не могу зайти в аккаунт уже второй день"),
])
def test_paired_cases_ordinary_vs_critical(service, ok_text, critical_text):
    ok, _ = run(service, ok_text, kind="report")
    crit, _ = run(service, critical_text, kind="report")
    assert ok.priority != "p0", (ok_text, ok.category, ok.priority)
    assert crit.priority == "p0" and crit.needs_human, (critical_text, crit.category, crit.priority)


def test_in_game_threat_is_not_a_real_life_threat_for_complaints_too(service):
    game, _ = run(service, "Игрок сказал, что найдёт меня в игре возле дома", kind="complaint", att=1)
    real, _ = run(service, "Игрок сказал, что найдёт меня в реальной жизни", kind="complaint", att=1)
    assert game.priority != "p0" and real.priority == "p0"


# =========================================================================================== 9/10. LLM queue, deadline, unknown cost
def test_tasks_whose_budget_expired_in_the_queue_send_nothing():
    started: list[float] = []

    def slow(req):
        started.append(time.perf_counter())
        time.sleep(0.65)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{}"}]})

    s = dataclasses.replace(load_settings(), llm_api_key="k", llm_provider="anthropic", llm_model="m", llm_total_budget_s=0.3,
                            llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=5.0)
    cl = LLMClient(s, transport=httpx.MockTransport(slow))
    from app.classify import analyze

    a = analyze(make_req("Arman_Petrov оскорбил меня", att=1))
    out: list = []
    t0 = time.perf_counter()
    threads = [threading.Thread(target=lambda: out.append(cl.draft(a, [], "ru", True, "t", time_budget_s=0.3))) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    waited = time.perf_counter() - t0
    time.sleep(0.9)  # give any leftover task the chance to (wrongly) send
    assert waited < 0.6
    assert all(x - t0 < 0.35 for x in started), "a task sent its request after its deadline"
    sent = [r for r in out if r.sent]
    assert len(sent) == len(started) <= 8
    assert all(cl.cost_usd(r) is None for r in sent)                  # sent, no usage record: unknown, never zero
    assert all(cl.cost_usd(r) == 0.0 for r in out if not r.sent)     # nothing sent: really free
    assert all(r.cost_estimate_max_usd for r in sent)                # the estimate is kept apart
    cl.close()


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_cost_is_known_only_with_a_usage_record(provider):
    if provider == "gemini":
        no_usage = {"candidates": [{"content": {"parts": [{"text": "{\"text\": \"Здравствуйте!\"}"}]}, "finishReason": "STOP"}]}
        with_usage = {**no_usage, "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}}
    else:
        no_usage = {"content": [{"type": "text", "text": "{\"text\": \"Здравствуйте!\"}"}]}
        with_usage = {**no_usage, "usage": {"input_tokens": 100, "output_tokens": 10}}
    from app.classify import analyze

    a = analyze(make_req("Arman_Petrov оскорбил меня", att=1))
    for payload, known in ((no_usage, False), (with_usage, True)):
        s = dataclasses.replace(load_settings(), llm_api_key="k", llm_provider=provider, llm_model="m", llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=5.0)
        cl = LLMClient(s, transport=httpx.MockTransport(lambda req, p=payload: httpx.Response(200, json=p)))
        res = cl.draft(a, [], "ru", True, "t")
        assert res.sent == 1 and (cl.cost_usd(res) is not None) is known, (provider, payload)


def test_a_rejected_first_request_makes_the_cost_unknown():
    calls = []

    def h(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"message": "thinking is not supported"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{\"text\": \"Здравствуйте!\"}"}], "usage": {"input_tokens": 5, "output_tokens": 5}})

    s = dataclasses.replace(load_settings(), llm_api_key="k", llm_provider="anthropic", llm_model="m", llm_price_in_per_mtok=1.0, llm_price_out_per_mtok=5.0)
    cl = LLMClient(s, transport=httpx.MockTransport(h))
    from app.classify import analyze

    res = cl.draft(analyze(make_req("Arman_Petrov оскорбил меня", att=1)), [], "ru", True, "t")
    assert res.sent == 2 and cl.cost_usd(res) is None  # the rejected request is not assumed to be free


# =========================================================================================== 11. batch robustness (CLI and HTTP)
@pytest.mark.parametrize("bad", [12345, {"x": 1}, [1, 2], None, "not a date", "2026-13-45T99:99:99"])
def test_a_bad_created_at_does_not_break_the_batch(rulebook, tmp_path, bad):
    from app.cli import predict

    rows = [{"ticket_id": "a", "kind": "report", "created_at": "2026-03-14T19:02:00", "messages": [{"role": "player", "text": "Привет"}]},
            {"ticket_id": "b", "kind": "report", "created_at": bad, "messages": [{"role": "player", "text": "Что такое DM?"}]},
            {"ticket_id": "c", "kind": "report", "messages": [{"role": "player", "text": "Застрял в текстурах"}]}]
    p, o = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    summary = predict(str(p), str(o), service=TriageService(load_settings(), rules=rulebook, dedup=DedupIndex()))
    out = [json.loads(line) for line in o.read_text(encoding="utf-8").splitlines()]
    assert [x["ticket_id"] for x in out] == ["a", "b", "c"]                       # input order kept
    assert "processing_error" not in out[0] and "processing_error" not in out[2]  # neighbours processed normally
    if isinstance(bad, str) or bad is None:
        assert "processing_error" not in out[1]                                   # a string/None date is just "no time"
    else:
        assert out[1]["processing_error"] and out[1]["needs_human"] is True and summary["errors"] == 1  # not passed off as processed


def test_http_batch_reports_a_bad_ticket_and_processes_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "b.db"))
    import app.main as m

    importlib.reload(m)
    with TestClient(m.app) as c:
        good = {"ticket_id": "g", "kind": "report", "messages": [{"role": "player", "text": "Привет"}]}
        bad = {"ticket_id": "b", "kind": "report", "created_at": 12345, "messages": [{"role": "player", "text": "Что такое DM?"}]}
        res = c.post("/triage/batch", json={"tickets": [good, bad, {"no": "kind"}, good]}).json()["results"]
        assert "response" in res[0] and "response" in res[3]
        assert res[1]["error"].startswith("invalid ticket") and res[1]["ticket_id"] == "b" and res[2]["error"].startswith("invalid ticket")
