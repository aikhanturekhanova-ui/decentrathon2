"""Segment-level language recognition for mixed messages («Сәлем бро, help me, застрял в текстурах»).

`detect_lang` (textutil) gives one label per message: the contract's `lang` and the reply language. This module explains WHICH parts
of a message are in which language: every word is tagged from strong evidence only (Kazakh/Uzbek letters, marker words, Uzbek suffixes,
Russian function words, Latin transliteration markers); words without evidence take the language of their neighbours; consecutive
words of one language form a segment with character offsets in the original text. Game jargon shared by all languages («бан», «админ»,
«dm», «бро») is never evidence of a second language, so a Russian message with «dm» is not «mixed».
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .textutil import (EN_MARKERS, ES_ONLY, KK_LAT_MARKERS, KK_WORDS, PT_ONLY, RU_STOP, TRANSLIT_MARKERS, UZ_MARKERS, _UZ_SUFFIX, detect_lang,
                       normalize_game_chars)

_WORDRX = re.compile(r"<\w+>|[^\W\d_]+_[^\W\d_]\w*|[^\W\d_]+(?:['’ʻʼ`][^\W\d_]+)*", re.UNICODE)
_NICK = re.compile(r"^[^\W\d_]+_[^\W\d_]\w*$")  # synthetic nicknames (Arman_Petrov) belong to no language
# Russian inflection endings (checked only on Cyrillic words without Kazakh letters and outside the Kazakh word list)
_RU_ENDING = re.compile(r"(?:ться|тся|ешь|ишь|ете|ите|ют|ят|ил|ила|ило|или|ал|ала|али|ял|яла|ах|ях|ами|ями|ого|его|ому|ему|ый|ий|ая|яя|ое|ые|ие|ую|ость|ение|ения|ний)$")
# Kazakh words and endings without Kazakh-only letters (the quotation particle «деп», «жазды» …); endings that Russian also has
# («-ды» in «воды», «-лар» in «доллар») are deliberately not used
KK_SHORT = {"деп", "деді", "жазды", "айтты", "біз", "сіз", "мына", "анау", "қазір", "ойын", "ойыншы", "сала", "бере", "алды", "барды", "кетті"}
_KK_ENDING = re.compile(r"(?:ған|ген|мын|сыз|ның|дың|тың)$")
RU_SHORT = {"в", "во", "с", "со", "к", "у", "и", "а", "о", "на", "не", "но", "за", "по", "от", "до", "из", "что", "как", "это", "я", "он", "она",
            "мы", "вы", "меня", "мне", "его", "её", "ее", "их", "уже", "еще", "ещё", "был", "была", "есть", "нет", "да", "так", "тут", "там",
            "потом", "тоже", "очень", "если", "только", "когда", "просто", "сейчас", "здесь", "чмо", "чё", "че", "ну", "вот"}
# Russian written in Latin letters, incl. the swear words players type that way (evidence of «lat», not of English)
RU_LATIN = {"blyad", "blyat", "blya", "suka", "pizdec", "pizdets", "nahui", "nahuy", "ty", "che", "chto", "menya", "mne", "tebya", "eto", "kak", "pochemu"}
JARGON = {"admin", "админ", "админы", "ban", "бан", "dm", "дм", "db", "дб", "rp", "рп", "nonrp", "gz", "зз", "гз", "chat", "чат", "ok", "ок",
          "бро", "bro", "akkaunt", "аккаунт", "donat", "донат", "skrin", "скрин", "скрины", "chit", "чит", "soft", "софт", "lol", "лол"}
# Latin words shared by Kazakh and Uzbek (and frequent in both): not evidence for either
SHARED_LATIN = {"men", "meni", "sen", "seni", "bar", "ber", "bu", "ol", "va", "ham", "sizga", "biz"}
KK_LETTERS = set("әғқңөұүһі")
UZ_CYR_LETTERS = set("ҳў")
LANG_RU = {"ru": "русский", "kk": "казахский", "uz": "узбекский", "en": "английский", "es": "испанский", "pt": "португальский",
           "lat": "русский латиницей", "und": "не определён"}


@dataclass
class Segment:
    start: int
    end: int
    text: str
    lang: str
    evidence: int  # words with direct evidence in this segment (the rest inherited the language from neighbours)


@dataclass
class LangProfile:
    primary: str
    mixed: bool
    languages: dict[str, float] = field(default_factory=dict)  # share of words
    segments: list[Segment] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _tag(word: str) -> str | None:
    """Language of one word from direct evidence, or None."""
    w = word.lower()
    if w in JARGON or w in SHARED_LATIN or w.startswith("<"):
        return None
    cyr = any("Ѐ" <= c <= "ӿ" for c in w)
    if cyr:
        if set(w) & UZ_CYR_LETTERS:
            return "uz"
        if set(w) & KK_LETTERS or w in KK_WORDS or w in KK_SHORT or (len(w) >= 5 and _KK_ENDING.search(w)):
            return "kk"
        if w in RU_STOP or w in RU_SHORT or (len(w) >= 4 and _RU_ENDING.search(w)):
            return "ru"
        return None
    hits = []
    if w in EN_MARKERS:
        hits.append("en")
    if w in UZ_MARKERS or _UZ_SUFFIX.search(w) or re.search(r"[og]['’ʻʼ`]", w):
        hits.append("uz")
    if w in ES_ONLY or re.search(r"[ñ¿¡]", w):
        hits.append("es")
    if w in PT_ONLY or re.search(r"[ãõç]", w):
        hits.append("pt")
    if w in TRANSLIT_MARKERS or w in RU_LATIN:
        hits.append("lat")
    if w in KK_LAT_MARKERS:
        hits.append("kk")
    return hits[0] if len(hits) == 1 else None  # a word shared by two languages is not evidence


def profile(text: str, server_lang: str | None = None) -> LangProfile:
    primary = detect_lang(text, server_lang)
    norm, _ = normalize_game_chars(text, server_lang)  # same length as `text`
    toks = [m for m in _WORDRX.finditer(norm) if not _NICK.match(m.group(0))]
    if not toks:
        return LangProfile(primary, False, {primary: 1.0} if primary != "und" else {}, [])
    tags = [_tag(m.group(0)) for m in toks]
    direct = [t is not None for t in tags]
    # words without evidence: the nearest tagged word of the same script, else the message language
    script = ["c" if any("Ѐ" <= c <= "ӿ" for c in m.group(0)) else "l" for m in toks]
    resolved = list(tags)
    jargon = [m.group(0).lower() in JARGON for m in toks]
    for i, t in enumerate(tags):
        if t is not None:
            continue
        for d in range(1, len(toks)):
            near = [j for j in (i - d, i + d) if 0 <= j < len(toks) and tags[j] is not None]
            # a neighbour of the same script first; jargon («dm», «бан») may also take the language of the other script
            pick = next((j for j in near if script[j] == script[i]), None)
            if pick is None and jargon[i] and near:
                pick = near[0]
            if pick is not None:
                resolved[i] = tags[pick]
                break
        if resolved[i] is None:
            resolved[i] = ("ru" if primary not in {"kk", "uz"} else primary) if script[i] == "c" else (primary if primary not in {"ru", "kk"} else "und")
    segs: list[Segment] = []
    for m, lang, ev in zip(toks, resolved, direct):
        if segs and segs[-1].lang == lang:
            segs[-1].end, segs[-1].evidence = m.end(), segs[-1].evidence + int(ev)
        else:
            segs.append(Segment(m.start(), m.end(), "", lang, int(ev)))
    for s in segs:
        s.text = text[s.start:s.end]
    counts: dict[str, int] = {}
    for lang in resolved:
        counts[lang] = counts.get(lang, 0) + 1
    shares = {k: round(v / len(resolved), 2) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
    evidenced = {lang for lang, ev in zip(resolved, direct) if ev and lang != "und"}
    return LangProfile(primary, len(evidenced) >= 2, shares, segs)
