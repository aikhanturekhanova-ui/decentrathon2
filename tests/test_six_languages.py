"""The game client is localized in ru, kk, uz, en, es, pt: detection, classification, a reply in the
SAME language, citations with a paraphrase in that language, and the safety rules, for every language.

The phrases are written by the authors (not native-reviewed): this proves coverage and safety, not linguistic quality.
"""
from __future__ import annotations

import pytest

from app.guard import check_draft
from app.phrases import GISTS, P
from app.textutil import detect_lang
from tests.conftest import make_req

LANGS = ["ru", "kk", "uz", "en", "es", "pt"]
# (kind, attachments, expected category suffix, text) per language
CASES = {
    "ru": [("complaint", 1, "insult", "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть"), ("complaint", 1, "cheat", "использует читы, летает по воздуху"),
           ("complaint", 1, "dm", "убил меня без причины в зелёной зоне"), ("complaint", 1, "scam", "обманул, не вернул деньги за машину"),
           ("complaint", 1, "rmt_ads", "продаёт аккаунты за реальные деньги в телеграме"), ("complaint", 1, "admin", "не согласен с баном, админ выдал его несправедливо"),
           ("report", 0, "help", "застрял в текстурах, помогите выбраться"), ("report", 0, "account", "взломали мой аккаунт"),
           ("report", 0, "payment", "оплатил донат, монеты не пришли"), ("report", 0, "offtopic", "привет"), ("report", 0, "question", "можно ли оскорблять игроков в чате?")],
    "kk": [("complaint", 1, "insult", "Қайрат мені және отбасымды чатта қорлады, скрин бар"), ("complaint", 1, "cheat", "чит қолданады, ауада ұшады"),
           ("complaint", 1, "dm", "себепсіз өлтірді жасыл аймақта"), ("complaint", 1, "scam", "алдады, көлік ақшасын қайтармады"),
           ("complaint", 1, "rmt_ads", "нақты ақшаға аккаунт сатады"), ("complaint", 1, "admin", "банмен келіспеймін, админ әділетсіз бан берді"),
           ("report", 0, "help", "көмектесіңіз, текстурада тұрып қалдым"), ("report", 0, "account", "аккаунтымды бұзды"),
           ("report", 0, "payment", "донат төледім, ақша келмеді"), ("report", 0, "offtopic", "сәлем"), ("report", 0, "question", "чатта қорлауға бола ма?")],
    "uz": [("complaint", 1, "insult", "Jasur meni va oilamni chatda haqorat qildi, skrin bor"), ("complaint", 1, "cheat", "chit ishlatyapti, havoda uchyapti"),
           ("complaint", 1, "dm", "sababsiz o'ldirdi yashil zonada"), ("complaint", 1, "scam", "aldadi, mashina pulini qaytarmadi"),
           ("complaint", 1, "rmt_ads", "haqiqiy pulga akkaunt sotyapti"), ("complaint", 1, "admin", "banga rozi emasman, admin nohaq ban berdi"),
           ("report", 0, "help", "yordam bering, teksturada tiqilib qoldim"), ("report", 0, "account", "akkauntimni buzishdi"),
           ("report", 0, "payment", "donat to'ladim, tangalar kelmadi"), ("report", 0, "offtopic", "salom"), ("report", 0, "question", "chatda haqorat qilsa bo'ladimi?")],
    "en": [("complaint", 1, "insult", "he insulted me and my family in chat, screenshots attached"), ("complaint", 1, "cheat", "he is cheating, flying with hacks"),
           ("complaint", 1, "dm", "killed me for no reason in the green zone"), ("complaint", 1, "scam", "scammed me and did not return my car money"),
           ("complaint", 1, "rmt_ads", "selling accounts for real money on telegram"), ("complaint", 1, "admin", "I disagree with the ban, admin banned me unfairly"),
           ("report", 0, "help", "help I am stuck under the map"), ("report", 0, "account", "my account got hacked"),
           ("report", 0, "payment", "I paid for donation and coins did not arrive"), ("report", 0, "offtopic", "hello"), ("report", 0, "question", "is it allowed to insult in chat?")],
    "es": [("complaint", 1, "insult", "Juan_Perez me insultó a mí y a mi familia en el chat, tengo capturas"), ("complaint", 1, "cheat", "usa cheats, vuela por el aire con hack"),
           ("complaint", 1, "dm", "me mató sin motivo en la zona verde"), ("complaint", 1, "scam", "me estafó y no devolvió el dinero del coche"),
           ("complaint", 1, "rmt_ads", "vende cuentas por dinero real en telegram"), ("complaint", 1, "admin", "no estoy de acuerdo con el ban, el admin me baneó injustamente"),
           ("report", 0, "help", "ayuda estoy atascado bajo el mapa"), ("report", 0, "account", "me hackearon la cuenta"),
           ("report", 0, "payment", "pagué la donación y no llegaron las monedas"), ("report", 0, "offtopic", "hola"), ("report", 0, "question", "¿se puede insultar en el chat?")],
    "pt": [("complaint", 1, "insult", "Joao_Silva me insultou e à minha família no chat, tenho prints"), ("complaint", 1, "cheat", "usa cheat, voa pelo ar"),
           ("complaint", 1, "dm", "me matou sem motivo na zona verde"), ("complaint", 1, "scam", "me enganou e não devolveu o dinheiro do carro"),
           ("complaint", 1, "rmt_ads", "vende conta por dinheiro real no telegram"), ("complaint", 1, "admin", "não concordo com o ban, o admin me baniu injustamente"),
           ("report", 0, "help", "socorro estou preso embaixo do mapa"), ("report", 0, "account", "hackearam minha conta"),
           ("report", 0, "payment", "paguei a doação e as moedas não chegaram"), ("report", 0, "offtopic", "olá"), ("report", 0, "question", "é permitido insultar no chat?")],
}


def test_every_language_has_every_phrase_and_gist():
    keys = set(P["ru"])
    for lang in LANGS:
        assert set(P[lang]) == keys, (lang, keys ^ set(P[lang]))
    curated = {c for c in GISTS}
    for lang in LANGS:
        assert [c for c in curated if lang not in GISTS[c]] == [], lang


@pytest.mark.parametrize("lang", LANGS)
def test_language_detection_per_language(lang):
    texts = [c[3] for c in CASES[lang] if c[2] not in {"offtopic"}]
    hits = sum(detect_lang(t) == lang for t in texts)
    assert hits / len(texts) >= 0.9, [(t, detect_lang(t)) for t in texts if detect_lang(t) != lang]


@pytest.mark.parametrize("lang", LANGS)
def test_classification_reply_language_and_citations_per_language(service, rulebook, lang):
    wrong = []
    for kind, att, expect, text in CASES[lang]:
        r, _ = service.triage(make_req(text, kind, att))
        if not r.category.endswith("." + expect):
            wrong.append((text, expect, r.category))
        if expect != "offtopic":
            assert r.lang == lang, (text, r.lang)
        assert r.draft_reply.lang == lang, (text, r.draft_reply.lang)  # the draft is written in the player's language
        assert check_draft(r.draft_reply.text, r.draft_reply.cited, rulebook, r.needs_human).ok, text
        if expect in {"payment", "account", "admin"} or kind == "complaint" and expect != "invalid":
            assert r.needs_human, text
        for c in r.draft_reply.cited:  # the cited rule is paraphrased in this language, not left untranslated
            assert GISTS[c][lang] in r.draft_reply.text, (lang, c)
    assert not wrong, wrong


@pytest.mark.parametrize("lang", LANGS)
def test_rule_question_is_answered_in_the_players_language(service, lang):
    kind, att, _, text = CASES[lang][-1]
    r, _ = service.triage(make_req(text, kind, att))
    assert r.category == "report.question" and r.draft_reply.cited and r.draft_reply.lang == lang and r.needs_human is False
