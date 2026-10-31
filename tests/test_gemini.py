"""Gemini provider of the optional LLM drafting, against a mocked HTTP transport (no network, no key)."""
from __future__ import annotations

import json
import os

import httpx
import pytest

from app.config import load_settings
from app.llm import LLMClient
from app.pipeline import TriageService
from tests.conftest import make_req


def _settings(monkeypatch, model="gemini-3.5-flash-lite"):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_MODEL", model)
    return load_settings()


def _answer(text: str, finish: str = "STOP", **usage) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"text": text, "cited": []}, ensure_ascii=False)}]}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": usage.get("pin", 900), "candidatesTokenCount": usage.get("pout", 80), "thoughtsTokenCount": usage.get("think", 0)}}


def test_provider_is_chosen_by_the_key(monkeypatch):
    s = _settings(monkeypatch)
    assert s.llm_provider == "gemini" and s.llm_model == "gemini-3.5-flash-lite" and s.llm_base_url.startswith("https://generativelanguage")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("LLM_MODEL", "")
    assert load_settings().llm_provider == "anthropic"  # Anthropic wins when both keys exist, unless LLM_PROVIDER says otherwise
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert load_settings().llm_provider == "gemini"


def test_request_shape_and_successful_draft(monkeypatch, rulebook):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["key"], seen["body"] = str(request.url), request.headers.get("x-goog-api-key"), json.loads(request.content)
        return httpx.Response(200, json=_answer("Здравствуйте! По вашим словам, вас оскорбили в чате. [[R1]] Администратор рассмотрит жалобу."))

    s = _settings(monkeypatch)
    svc = TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(handler)))
    r, dbg = svc.triage(make_req("Arman_Petrov оскорбил меня в чате, скрины есть", att=2))
    assert seen["url"].endswith("/v1beta/models/gemini-3.5-flash-lite:generateContent") and seen["key"] == "test-key-not-real"
    assert seen["body"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}  # Gemini 3 rejects thinkingBudget 0
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "<player_text>" in seen["body"]["contents"][0]["parts"][0]["text"] and seen["body"]["systemInstruction"]["parts"][0]["text"]
    assert dbg.llm_calls == 1 and dbg.llm_tokens_in == 900 and dbg.llm_tokens_out == 80
    assert dbg.mode in {"llm", "template"}  # the guard decides; either way the result is safe
    assert r.draft_reply.text


def test_gemini_2_uses_a_token_budget(monkeypatch):
    from app.classify import analyze

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_answer("Здравствуйте!"))

    s = _settings(monkeypatch, model="gemini-2.5-flash")
    LLMClient(s, transport=httpx.MockTransport(handler)).draft(analyze(make_req("помогите", kind="report", att=0)), [], "ru", True, "Здравствуйте!")
    assert seen["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_invalid_argument_is_retried_once_without_thinking(monkeypatch, rulebook):
    from app.classify import analyze

    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "thinkingConfig" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}})
        return httpx.Response(200, json=_answer("Здравствуйте! Уточните, пожалуйста, детали."))

    s = _settings(monkeypatch)
    res = LLMClient(s, transport=httpx.MockTransport(handler)).draft(analyze(make_req("помогите", kind="report", att=0)), [], "ru", True, "Здравствуйте!")
    assert len(calls) == 2 and "thinkingConfig" not in calls[1]["generationConfig"] and res.text and res.error is None


@pytest.mark.parametrize("payload,why", [
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "safety filter"),
    (_answer("x", finish="SAFETY"), "safety filter"),
    ({"candidates": []}, "no candidates"),
    ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "no content parts"),
])
def test_blocked_or_empty_answers_fall_back_to_the_template(monkeypatch, rulebook, payload, why):
    s = _settings(monkeypatch)
    svc = TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))))
    r, dbg = svc.triage(make_req("Arman_Petrov оскорбил меня в чате, скрины есть", att=2))
    assert dbg.mode == "template" and why in (dbg.llm_error or "") and r.draft_reply.text


def test_thinking_tokens_are_billed_as_output(monkeypatch):
    from app.classify import analyze

    s = _settings(monkeypatch)
    res = LLMClient(s, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_answer("Здравствуйте!", pout=50, think=30)))).draft(
        analyze(make_req("помогите", kind="report", att=0)), [], "ru", True, "Здравствуйте!")
    assert res.tokens_out == 80


def test_sensitive_tickets_never_reach_gemini(monkeypatch, rulebook):
    called = []
    s = _settings(monkeypatch)
    svc = TriageService(s, rules=rulebook, llm=LLMClient(s, transport=httpx.MockTransport(lambda r: called.append(1) or httpx.Response(500))))
    r, _ = svc.triage(make_req("оплатил донат, коины не пришли, верните деньги", kind="report", att=0))
    assert not called and r.needs_human and r.priority == "p0"
    assert os.environ.get("GEMINI_API_KEY") == "test-key-not-real"  # monkeypatched, never the real key


def test_verbatim_template_sentences_are_trusted_but_new_text_is_checked():
    from app.pipeline import merge_llm_text, template_sentences

    template = ("Здравствуйте! Вложений в обращении: 1; мы их не просматриваем, содержимое проверит администратор. "
                "Заранее сообщить результат или наказание мы не можем.")
    verified = template_sentences(template)
    ok, why = merge_llm_text("Здравствуйте! Вложений в обращении: 1; мы их не просматриваем, содержимое проверит администратор. "
                             "Заранее сообщить результат или наказание мы не можем.", [], "ru", verified=verified)
    assert why is None and ok
    # one changed word turns a verified sentence into the model's own text: checked, and a promise is rejected
    bad, why = merge_llm_text("Здравствуйте! Заранее сообщаем: наказание игрок получит.", [], "ru", verified=verified)
    assert bad is None and why
    # without the verified set the same safe sentence is still judged by the strict checks (the old behaviour)
    assert merge_llm_text("Заранее сообщить результат или наказание мы не можем.", [], "ru")[0] is None


def test_reference_template_has_no_rule_words():
    from app.guard import RULE_TALK
    from app.pipeline import _MARKER, template_with_markers

    class Doc:
        ref = "2.1"

    t = template_with_markers("Здравствуйте! Что говорят правила: п. 2.1 общих правил — оскорбления запрещены. Жалобу рассмотрит администратор.", [Doc()])
    assert "[[R1]]" in t and not RULE_TALK.search(_MARKER.sub(" ", t))


def test_llm_switch_and_real_data_scripts_never_call_a_model(monkeypatch):
    import importlib
    import sys

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_ENABLED", "0")
    assert load_settings().llm_enabled is False
    monkeypatch.setenv("LLM_ENABLED", "1")
    assert load_settings().llm_enabled is True
    sys.path.insert(0, "scripts")
    import _common
    importlib.reload(_common)  # every evaluation script on real tickets imports it first
    assert os.environ["LLM_ENABLED"] == "0" and load_settings().llm_enabled is False
