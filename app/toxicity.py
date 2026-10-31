"""Contextual abusive-language analysis for moderation (ru / kk / uz / en / es / pt, transliteration, mixed messages).

It supports the triage, it does not replace it: the result is shown to the administrator (moderation panel, `reasons`) and adds a
signal to the insult/threat decisions. It never decides a punishment.

Layers:
  1. length-preserving normalization: NFKC per character, lower case, the game's Kazakh look-alikes, Latin/Cyrillic look-alikes and
     digits-for-letters chosen by the script of each token, zero-width and combining characters turned into separators, spaced-out
     letters («с у к а») joined. Every character keeps its position, so evidence spans point into the ORIGINAL text;
  2. a lexicon of stems by language and type, compiled into patterns that tolerate repeated letters and separators inside a word;
  3. context: who says it (quoted / reported by the author, addressed by the author to someone, or the author's own exclamation),
     whether it is a joke or friendly banter, and whether a threat refers to the game or to real life;
  4. a confidence per finding and an overall decision with a human-readable explanation.

A player who REPORTS an insult is never treated as the offender: words inside quotes or after «написал / назвал / деді / dedi / said»
are attributed to the accused.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Optional

from .textutil import normalize_game_chars

# ---------------------------------------------------------------------------------------------------------------------
# 1. length-preserving normalization

SEP = "·"  # separator placed where obfuscation characters were
_ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "­"}
_LAT2CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "k": "к", "m": "м", "t": "т", "h": "н"}
_CYR2LAT = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "к": "k", "м": "m", "т": "t"}
_DIGIT_CYR = {"0": "о", "3": "з", "4": "ч", "6": "б", "@": "а"}
_DIGIT_LAT = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}
_SPACED = re.compile(r"(?<![^\W\d_])(?:[^\W\d_] ){2,}[^\W\d_](?![^\W\d_])")


def _is_cyr(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"


def _is_lat(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def normalize(text: str) -> str:
    """Same length as `text`; see the module docstring."""
    text, _ = normalize_game_chars(text)
    chars: list[str] = []
    for ch in text:
        if ch in _ZERO_WIDTH or unicodedata.combining(ch):
            chars.append(SEP)
            continue
        n = unicodedata.normalize("NFKC", ch)
        n = n if len(n) == 1 else ch
        low = n.lower()
        chars.append(low if len(low) == 1 else n)
    s = "".join(chars).replace("ё", "е")
    out = list(s)
    # script-aware look-alikes and digits, decided per whitespace-separated token
    for m in re.finditer(r"\S+", s):
        tok = m.group(0)
        cyr = sum(_is_cyr(c) for c in tok)
        lat = sum(_is_lat(c) for c in tok)
        if not cyr and not lat:
            continue  # "1488" or a number stays as it is
        to_cyr = cyr > lat or (cyr == lat and cyr > 0)
        table = _LAT2CYR if to_cyr else _CYR2LAT
        digits = _DIGIT_CYR if to_cyr else _DIGIT_LAT
        for i, c in enumerate(tok):
            j = m.start() + i
            if c in table:
                out[j] = table[c]
            elif c in digits and (cyr + lat) >= 2:
                out[j] = digits[c]
    s = "".join(out)
    # «с у к а» -> «с·у·к·а» (only runs of 3+ single letters, so ordinary words are never glued together)
    return _SPACED.sub(lambda m: m.group(0).replace(" ", SEP), s)


# ---------------------------------------------------------------------------------------------------------------------
# 2. lexicon

PROFANITY, INSULT, FAMILY, THREAT, HATE, SEXUAL = "profanity", "insult", "family_insult", "threat", "hate_speech", "sexual_harassment"
SEVERITY = {PROFANITY: 1, INSULT: 2, FAMILY: 3, THREAT: 3, HATE: 3, SEXUAL: 3}
TYPE_RU = {PROFANITY: "нецензурная брань", INSULT: "прямое оскорбление", FAMILY: "оскорбление родных", THREAT: "угроза",
           HATE: "ненависть по национальному/иному признаку", SEXUAL: "сексуальные домогательства"}


@dataclass(frozen=True)
class Entry:
    stems: tuple[str, ...]
    type: str
    lang: str
    whole: bool = False  # True: the stem must be the whole word; False: the stem starts a word (suffixes allowed)
    prefixes: tuple[str, ...] = ()  # verbal prefixes allowed before the stem (ru «за-/вы-/на-…»)
    ambiguous: bool = False  # has an innocent reading: lower confidence


_RU_VERB_PREFIXES = ("за", "вы", "у", "на", "от", "отъ", "по", "при", "раз", "разъ", "съ", "до", "долбо", "об", "пере", "под", "про", "въ")

LEXICON: tuple[Entry, ...] = (
    # --- Russian (Cyrillic) -------------------------------------------------------------------------------------
    Entry(("хуй", "хуе", "хуя", "хуи", "нахуй", "нахер", "похуй"), PROFANITY, "ru"),
    Entry(("пизд",), PROFANITY, "ru", prefixes=("рас", "с", "от", "за", "вы", "у", "на", "под", "при")),
    Entry(("еба", "ебл", "ебн", "ебу", "ебись", "ебёт", "ебет", "ебош"), PROFANITY, "ru", prefixes=_RU_VERB_PREFIXES),
    Entry(("бля",), PROFANITY, "ru"),
    Entry(("сука", "суки", "суку", "сукой", "суками", "сукам"), INSULT, "ru", whole=True),  # whole words: not the surname «Сукачев»
    Entry(("сучк", "сучар", "сучь"), INSULT, "ru"),
    Entry(("мудак", "мудил", "мудозв", "мудач"), INSULT, "ru"),
    Entry(("гандон", "шлюх", "шалав", "мраз", "ублюд", "дебил", "долбо", "кретин", "гнид", "падл", "скотин", "уебок", "уебан", "уебищ"), INSULT, "ru"),
    Entry(("чмо", "чмошник", "тварь", "твари", "лох", "лошара", "лошок"), INSULT, "ru", whole=True),
    Entry(("даун", "дауны", "дауна", "урод", "уроды", "уродина", "дура", "дурак", "идиот", "идиоты"), INSULT, "ru", whole=True, ambiguous=True),
    Entry(("пидор", "пидар", "пидр", "педик"), HATE, "ru"),
    Entry(("чурк", "хачей", "хачик", "жидов", "жидяр", "ниггер", "негрил", "черножоп", "кацап"), HATE, "ru"),
    Entry(("хач", "хачи", "жид", "жиды", "нигер", "нигеры", "хохол", "хохлы", "хохлов"), HATE, "ru", whole=True),  # whole words: not «Нигерия», «хохлома»
    Entry(("отсос", "отсоси", "пососи", "изнасил", "нюдс", "минет", "трахн", "выебу", "выебать"), SEXUAL, "ru"),
    Entry(("соси", "сосать"), SEXUAL, "ru", whole=True),  # not «сосиска»
    Entry(("убью", "урою", "закопаю", "зарежу", "порежу", "прибью", "грохну", "вычислю"), THREAT, "ru", whole=True),
    # --- Russian in Latin letters -------------------------------------------------------------------------------
    Entry(("blya", "blyat", "blyad", "bl9", "pizd", "pzdc", "nahui", "nahuy", "naxui", "naxuy"), PROFANITY, "lat"),
    Entry(("huy", "hui", "xuy", "xui", "huj"), PROFANITY, "lat", whole=True, ambiguous=True),  # whole words: not Uzbek «hujjat»
    Entry(("eban", "ebal", "ebat", "yeban", "uebok", "ueban", "dolboeb", "dolbaeb"), PROFANITY, "lat"),
    Entry(("suka", "cyka", "mudak", "gandon", "debil", "shluha", "shlyuha", "tvar", "mraz", "chmo"), INSULT, "lat", whole=True),
    Entry(("pidor", "pidar", "pidr"), HATE, "lat"),
    # --- Kazakh -------------------------------------------------------------------------------------------------
    Entry(("сікт", "сігей", "сігем", "сігіп", "сігер", "сікем", "сікейін", "сиктир", "siktir", "sikeyin", "sigeyin", "sigem"), SEXUAL, "kk"),
    Entry(("қотақ", "котак", "qotaq", "kotak", "қўтоқ", "qotoq", "qo'toq"), PROFANITY, "kk"),
    Entry(("жалап", "zhalap", "jalap", "jalab"), INSULT, "kk"),
    Entry(("жалаб",), INSULT, "kk", whole=True),  # whole word: «жалабу» is a misspelt Russian «жалобу» (complaint)
    Entry(("шешеңді", "шешеңнің", "шешеңе", "шешенді", "шешеннің", "sheshendi", "sheshenni", "анаңды", "anandy", "anangdy"), FAMILY, "kk"),
    Entry(("амыңды", "амыңа", "амыңнан", "amyndy", "amyna"), SEXUAL, "kk"),
    Entry(("ақымақ", "акымак", "aqymaq", "akymak"), INSULT, "kk", ambiguous=True),
    Entry(("өлтіремін", "өлтірем", "олтирем", "oltiremin", "oltirem", "өлтіріп тастаймын"), THREAT, "kk"),
    # --- Uzbek --------------------------------------------------------------------------------------------------
    Entry(("qanjiq", "қанжиқ", "канжик", "dalbayob", "далбаёб", "dolboyob", "haromi", "ҳароми", "xaromi"), INSULT, "uz"),
    Entry(("onangni", "онангни", "onangdi", "onangga", "онангга"), FAMILY, "uz"),
    Entry(("sikay", "sikaman", "sikdim", "sikib", "сикаман", "сикай"), SEXUAL, "uz"),
    Entry(("ahmoq", "axmoq", "аҳмоқ", "ахмок"), INSULT, "uz", ambiguous=True),
    Entry(("o'ldiraman", "oldiraman", "ўлдираман", "улдираман", "o'ldirib qo'yaman"), THREAT, "uz"),
    # --- English ------------------------------------------------------------------------------------------------
    Entry(("fuck", "fck", "fuk", "shit", "wtf"), PROFANITY, "en"),
    Entry(("bitch", "biatch", "whore", "slut", "cunt", "asshole", "bastard", "retard", "moron", "stfu"), INSULT, "en"),
    Entry(("dick",), INSULT, "en", whole=True, ambiguous=True),
    Entry(("motherf",), FAMILY, "en"),
    Entry(("your mom", "ur mom", "yo mama", "your mother"), FAMILY, "en", whole=True),  # not «your moment»
    Entry(("nigger", "nigga", "faggot"), HATE, "en"),
    Entry(("fag",), HATE, "en", whole=True),
    Entry(("rape", "send nudes", "nudes"), SEXUAL, "en", whole=True),
    Entry(("kill you", "kill yourself", "kys"), THREAT, "en", whole=True),
    # --- Spanish / Portuguese ----------------------------------------------------------------------------------
    Entry(("puta", "puto", "mierda", "joder", "coño", "porra", "caralho", "merda"), PROFANITY, "es/pt"),
    Entry(("cabron", "cabrón", "pendejo", "gilipollas", "otario", "otário", "babaca", "vagabund"), INSULT, "es/pt"),
    Entry(("hijo de puta", "hdp", "filho da puta", "fdp"), FAMILY, "es/pt"),
    Entry(("tu madre", "sua mae", "sua mãe"), FAMILY, "es", whole=True),
    Entry(("maricon", "maricón", "viado"), HATE, "es/pt"),
    Entry(("te voy a matar", "te mato", "vou te matar"), THREAT, "es/pt", whole=True),
    # --- hate symbols -------------------------------------------------------------------------------------------
    Entry(("зиг хайль", "sieg heil", "zig hail", "1488", "14/88"), HATE, "und", whole=True),
)

_SEP_CLASS = f"[{SEP}.\\-_*'`~|,]{{0,2}}"


def _stem_pattern(stem: str) -> str:
    """Letters may repeat («сууука») and be separated by obfuscation characters; spaces in a stem are real word gaps."""
    parts = []
    for word in stem.split(" "):
        letters = [re.escape(ch) + "+" if ch.isalpha() else re.escape(ch) for ch in word]
        parts.append(_SEP_CLASS.join(letters))
    return r"\s+".join(parts)


def _compile(e: Entry) -> re.Pattern:
    stems = "|".join(_stem_pattern(s) for s in sorted(e.stems, key=len, reverse=True))
    pre = "(?:" + "|".join(re.escape(p) for p in e.prefixes) + ")?" if e.prefixes else ""
    tail = r"(?![^\W\d_])" if e.whole else r"[^\W\d_]*"
    return re.compile(rf"(?<![^\W\d_])({pre}(?:{stems}){tail})", re.UNICODE)


_COMPILED = [(e, _compile(e)) for e in LEXICON]

# ---------------------------------------------------------------------------------------------------------------------
# 2b. cross-script phonetic layer
#
# Players write one language with another language's alphabet: Kazakh swearing in Russian letters («сигейин», «котак»), in Latin
# («qotaq», «sigeyin»), Uzbek in Cyrillic («онангни», «аҳмоқ»), English in Cyrillic («фак ю», «мазафака»), Russian in Latin («blyad»), and
# mix them in one sentence. Listing every spelling is impossible, so every word is also reduced to a phonetic SKELETON in plain Latin
# (қ/q/к → k, ғ/г → g, ң/н → n, ә/а → a, ө/ў/о → o, ұ/ү/у → u, і/ы/и/y → i, х/x/kh/ҳ/һ → h, ж/zh/j → j, ш/sh → sh, ч/ch/tch → ch, double
# letters collapsed, separators inside a word dropped). One canonical stem then matches all spellings, with an index map back to the
# original characters for the evidence spans.

_CYR2SKEL = {
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e", "ё": "e", "є": "e", "э": "e", "ж": "j", "з": "z", "и": "i",
    "і": "i", "ї": "i", "й": "i", "ы": "i", "к": "k", "қ": "k", "л": "l", "м": "m", "н": "n", "ң": "n", "о": "o", "ө": "o", "ў": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u", "ү": "u", "ф": "f", "х": "h", "ҳ": "h", "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "sh", "ъ": "", "ь": "", "ю": "iu", "я": "ia",
}
_LAT_DIGRAPHS = (("tch", "ch"), ("sch", "sh"), ("kh", "h"), ("zh", "j"), ("ck", "k"), ("ph", "f"))
_LAT_SINGLE = {"q": "k", "x": "h", "w": "v", "y": "i", "c": "k"}
_INWORD_SEP = set(SEP + ".-_*'`~|’ʻʼ")


def skeleton(norm: str) -> tuple[str, list[int]]:
    """Phonetic skeleton of a normalized text and, for every skeleton character, the index of the original character."""
    out: list[str] = []
    idx: list[int] = []
    n = len(norm)
    i = 0
    while i < n:
        ch = norm[i]
        if ch.isalpha():
            low = ch.lower()
            if low in _CYR2SKEL:
                piece, step = _CYR2SKEL[low], 1
            elif "a" <= low <= "z":
                piece, step = None, 1
                for dg, rep_ in _LAT_DIGRAPHS:
                    if norm[i:i + len(dg)].lower() == dg:
                        piece, step = rep_, len(dg)
                        break
                if piece is None:
                    nxt = norm[i + 1:i + 2].lower()
                    piece = ("s" if nxt in {"e", "i"} else "k") if low == "c" else _LAT_SINGLE.get(low, low)
            else:
                piece, step = "", 1  # other alphabets (accents are rare in this lexicon)
            for pc in piece:
                if not (out and out[-1] == pc and idx and idx[-1] >= i - 2):  # «сууука», «иттің» -> one letter
                    out.append(pc)
                    idx.append(i)
            i += step
            continue
        if ch in _INWORD_SEP and out and out[-1] != " " and i + 1 < n and norm[i + 1].isalpha():
            i += 1  # «с.у.к.а», «o'ldiraman»: a separator inside a word is not a word boundary
            continue
        if not (out and out[-1] == " "):
            out.append(" ")
            idx.append(i)
        i += 1
    return "".join(out), idx


def _fold_stem(stem: str) -> str:
    return skeleton(stem.lower())[0].strip()


# canonical stems: each matches every alphabet the word can be written in (see the module docstring for the languages)
SKELETON_LEXICON: tuple[Entry, ...] = (
    # Kazakh
    Entry(("siktir", "sikt", "sigeiin", "sigei", "sigem", "sigip", "siker", "sikem", "sikeiin"), SEXUAL, "kk"),
    Entry(("kotak", "kotok"), PROFANITY, "kk"),
    Entry(("jalap",), INSULT, "kk"),
    Entry(("sheshendi", "sheshennin", "sheshene", "anandi", "anannin"), FAMILY, "kk"),
    Entry(("amindi", "aminnan"), SEXUAL, "kk", whole=True),
    Entry(("akimak", "akmak"), INSULT, "kk", ambiguous=True),
    Entry(("it balasi", "itting balasi", "itin balasi"), INSULT, "kk", whole=True),
    # Uzbek
    Entry(("kanjik", "dalbaiob", "dalbaeb", "haromi"), INSULT, "uz"),
    Entry(("iflos",), INSULT, "uz", ambiguous=True),  # also just «dirty»
    Entry(("onangni", "onangga", "onangdi"), FAMILY, "uz"),
    Entry(("sikai", "sikaman", "sikdim", "sikib"), SEXUAL, "uz"),
    Entry(("ahmok", "ahmak"), INSULT, "uz", ambiguous=True),
    # Russian written in any alphabet («пiзда», «blyad», «сyка»)
    Entry(("pizd", "bliad", "bliat", "dolboeb", "zaeb", "uebok", "uebish", "nahui", "pohui"), PROFANITY, "ru"),
    Entry(("suka", "suki", "suku"), INSULT, "ru", whole=True),
    Entry(("suchk", "suchar", "mudak", "mudil", "gandon", "debil", "shliuh", "shluh", "ublud"), INSULT, "ru"),
    Entry(("pidor", "pidar", "pidr", "pidaras", "pidoras"), HATE, "ru"),
    # English written in Cyrillic («фак ю», «мазафака», «бич»)
    Entry(("fak", "fakin", "fak iu", "fak of", "fuk", "fukin"), PROFANITY, "en", whole=True),  # not «shit»: «шить» (to sew) folds to it
    Entry(("vatafak", "vot a fak", "vat a fak"), PROFANITY, "en", whole=True),
    Entry(("bich", "bichi"), INSULT, "en", whole=True, ambiguous=True),
    Entry(("mazafak", "madafak", "maza faka"), FAMILY, "en"),
)


def _compile_skeleton(e: Entry) -> re.Pattern:
    stems = "|".join(re.escape(_fold_stem(s)) for s in sorted(e.stems, key=len, reverse=True))
    tail = r"(?![a-z])" if e.whole else r"[a-z]*"
    return re.compile(rf"(?<![a-z])((?:{stems}){tail})")


_SKELETON_COMPILED = [(e, _compile_skeleton(e)) for e in SKELETON_LEXICON]

# the alphabet a language is normally written in: anything else is reported as «written with another alphabet»
_NATIVE = {"ru": "cyr", "kk": "cyr", "en": "lat", "es": "lat", "es/pt": "lat", "lat": "lat"}
LANG_NAME_RU = {"ru": "русский", "lat": "русский", "kk": "казахский", "uz": "узбекский", "en": "английский", "es": "испанский",
                "es/pt": "испанский/португальский", "und": "символы"}
_KK_ONLY = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


def written_form(original: str, lang: str) -> Optional[str]:
    """Which alphabet the word was written in, when it is not the usual one for its language («казахское слово русскими буквами»)."""
    cyr = any("\u0400" <= c <= "\u04ff" for c in original)
    lat = any(c.isascii() and c.isalpha() for c in original)
    if cyr and lat:
        return "смешение алфавитов внутри слова"
    script = "cyr" if cyr else ("lat" if lat else None)
    if script is None:
        return None
    if lang == "lat":
        return "русское слово латиницей"
    if lang == "kk" and script == "cyr" and not (set(original) & _KK_ONLY):
        return "казахское слово русскими буквами (без казахских букв)"
    native = _NATIVE.get(lang)
    if lang == "uz" or native is None or native == script:
        return None  # Uzbek is written in both alphabets
    name = {"ru": "русское", "kk": "казахское", "en": "английское", "es": "испанское", "es/pt": "испанское/португальское"}.get(lang, lang)
    return f"{name} слово {'латиницей' if script == 'lat' else 'кириллицей'}"

# ---------------------------------------------------------------------------------------------------------------------
# 3. context

REPORT_CUE = re.compile(
    r"(?:на)?писал\w*|сказал\w*|назвал\w*|обозвал\w*|оскорб\w*|говорит|пишет|послал\w*|кричал\w*|ругал\w*|матер\w*|обзыва\w*|угрожа\w*|"
    r"деді|жазды|айтты|қорлады|сөкті|боқтады|"
    r"dedi|yozdi|aytdi|so'?kdi|haqorat\w*|"
    r"said|called\s+me|wrote|told\s+me|insulted|typed|"
    r"dijo|me\s+llam[oó]|escribi[oó]|insult[oó]|disse|me\s+chamou|escreveu|xingou|"
    r"(?<![^\W\d_])(?:деп|dep|deb|деб)(?![^\W\d_])", re.I)  # kk «деп», uz «deb»: the quotation particle after the quoted words
# words about relatives: abuse next to them is an insult of the family (taxonomy example «оскорбил меня и мою семью»)
FAMILY_WORD = re.compile(
    r"(?<![^\W\d_])(?:мат(?:ь|ери|ерью)|мам[ауеы]?|мамк\w*|мамаш\w*|семь[юяеи]|родн\w*|сестр\w*|отц\w*|бабк\w*|"
    r"шеше\w*|ана(?:ң|ны|сы)\w*|ona(?:ng|si|ngni|ngga)?\w*|она(?:нг|си)\w*|mom|mother|mama|madre|m[aã]e)(?![^\W\d_])", re.I)
SECOND_PERSON = re.compile(
    r"(?<![^\W\d_])(?:ты|тебя|тебе|тобой|твой|твою|твоя|твоё|твое|твои|вы|вас|вам|ваш|сен|сені|саған|сенің|sen|seni|senga|sening|"
    r"you|your|ur|u|tu|te|vc|voce|você|seu|sua)(?![^\W\d_])", re.I)
LAUGHTER = re.compile(r"(?:а?ха){2,}|(?:ах){2,}|(?:ха){2,}|(?<![^\W\d_])(?:лол|кек|xd|lol|lmao|haha\w*|jaja\w*|kkkk+|rsrs\w*)(?![^\W\d_])|\){2,}|😂|🤣|😆", re.I)
FRIENDLY = re.compile(r"(?<![^\W\d_])(?:бро|братан\w*|братишк\w*|дружище|bro|bratan|dude|mate|brat)(?![^\W\d_])", re.I)
REAL_LIFE = re.compile(r"в\s+реальн\w*|в\s+жизни|в\s+реале|irl|адрес|домой|дом\s+твой|школ\w*|по\s+ip|айпи|вычисл\w*|деанон\w*|ddos|ддос\w*|сват\w*|"
                       r"өміріңде|үйіңе|uyingga|real\s+life|your\s+house|tu\s+casa|sua\s+casa", re.I)
IN_GAME = re.compile(r"в\s+игр\w*|на\s+сервер\w*|в\s+рп|по\s+рп|ingame|in\s+game|в\s+катк\w*|на\s+респ\w*|у\s+спавн\w*|o'yinda|ойында|en\s+el\s+juego|no\s+jogo", re.I)
_QUOTE_SPANS = re.compile(r"«[^»]*»|“[^”]*”|\"[^\"]*\"|'[^']{2,}'")


@dataclass
class Finding:
    message: int  # index in the request's messages
    role: str
    start: int  # character offsets in the ORIGINAL text of that message
    end: int
    text: str
    type: str
    type_ru: str
    lang: str
    severity: int  # 1 mild .. 3 serious (after context adjustments)
    direction: str  # quoted_or_reported | author_to_addressee | author_expressive | accused_reply
    obfuscated: bool
    real_life: Optional[bool] = None  # threats only
    confidence: float = 0.0
    form: Optional[str] = None  # written with another language's alphabet, e.g. «казахское слово латиницей»


@dataclass
class ToxicityReport:
    has_abuse: bool = False
    reported_abuse: bool = False  # the author reports/quotes someone else's abusive words
    author_abusive: bool = False  # the author's own words are abusive towards someone
    accused_abusive: bool = False  # the reported player's own comment contains abuse
    languages: list[str] = field(default_factory=list)  # languages of the abusive words found
    mixed_language_abuse: bool = False  # abusive words of two or more languages in one ticket
    types: list[str] = field(default_factory=list)
    max_severity: int = 0
    tone: str = "neutral"  # neutral | hostile | joking
    real_life_threat: bool = False
    confidence: float = 0.0
    findings: list[Finding] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _clause_bounds(s: str, start: int, end: int) -> tuple[int, int]:
    left = max(s.rfind(ch, 0, start) for ch in ".!?\n;")
    rights = [i for i in (s.find(ch, end) for ch in ".!?\n;") if i != -1]
    return left + 1, (min(rights) if rights else len(s))


def _in_quotes(original: str, start: int, end: int) -> bool:
    return any(q.start() < start and end <= q.end() for q in _QUOTE_SPANS.finditer(original))


def _direction(original: str, norm: str, start: int, end: int, role: str) -> tuple[str, float]:
    if role == "reported_player":
        return "accused_reply", 0.9
    if _in_quotes(original, start, end):
        return "quoted_or_reported", 0.9
    a, b = _clause_bounds(norm, start, end)
    before = norm[max(a, start - 60):start]
    if REPORT_CUE.search(before) or re.search(r":\s*$", before):
        return "quoted_or_reported", 0.8
    # Kazakh/Uzbek put the verb last: «ол маған жалап деді», «onangni deb so'kdi»
    if REPORT_CUE.search(norm[end:min(b, end + 40)]):
        return "quoted_or_reported", 0.75
    if SECOND_PERSON.search(norm[a:b]):
        return "author_to_addressee", 0.7
    return "author_expressive", 0.6


def analyze_text(text: str, message: int = 0, role: str = "player") -> list[Finding]:
    if not text:
        return []
    norm = normalize(text)
    if len(norm) != len(text):  # never expected (normalize is length-preserving); spans must stay valid regardless
        norm = text.lower()
    found: list[Finding] = []
    taken: list[tuple[int, int]] = []
    cands = [(m.start(1), m.end(1), e) for e, rx in _COMPILED for m in rx.finditer(norm)]
    skel, idx = skeleton(norm)
    for e, rx in _SKELETON_COMPILED:
        for m in rx.finditer(skel):
            a, b = m.start(1), m.end(1) - 1
            s0, t0 = idx[a], idx[b] + 1
            while t0 < len(norm) and norm[t0].isalpha():  # the skeleton may end inside a digraph: take the rest of the word
                t0 += 1
            cands.append((s0, t0, e))
    # the longest match wins an overlap: «hijo de puta» is a family insult, not the bare «puta»
    for s, t, e in sorted(cands, key=lambda c: (-(c[1] - c[0]), c[0])):
        if any(s < b and a < t for a, b in taken):
            continue
        taken.append((s, t))
        orig = text[s:t]
        seg = norm[s:t]
        obfuscated = orig.lower().replace("ё", "е") != seg or SEP in seg or bool(re.search(r"[^\W\d_][.\-_*`~|,]+[^\W\d_]", seg))
        direction, dconf = _direction(text, norm, s, t, role)
        kind = e.type
        if kind in {PROFANITY, INSULT, SEXUAL} and FAMILY_WORD.search(norm[max(0, s - 25):min(len(norm), t + 25)]):
            kind = FAMILY  # «твою мать еб*л», «мамку …»: abuse aimed at the relatives
        sev = SEVERITY[kind]
        real = None
        if kind == THREAT:
            a, b = _clause_bounds(norm, s, t)
            ctx = norm[max(0, a - 40):min(len(norm), b + 60)]
            real = bool(REAL_LIFE.search(ctx))
            if not real:  # «убью тебя в игре» is game talk; without any context it is unclear, not «serious»
                sev = 1 if IN_GAME.search(ctx) else 2
        conf = (0.6 if e.ambiguous else 0.9) * (0.9 if obfuscated else 1.0)
        conf = round(min(conf, 0.5 + dconf / 2), 2)
        lang = "ru" if e.lang == "lat" else e.lang
        found.append(Finding(message, role, s, t, orig, kind, TYPE_RU[kind], lang, sev, direction, obfuscated, real, conf,
                             written_form(orig, e.lang)))
    found.sort(key=lambda f: f.start)
    return found


def analyze_messages(messages: list) -> ToxicityReport:
    """`messages`: objects or dicts with `role` and `text` (the admin's messages are ignored)."""
    rep = ToxicityReport()
    texts: dict[int, str] = {}
    for i, m in enumerate(messages):
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "player")
        text = getattr(m, "text", None) if not isinstance(m, dict) else m.get("text")
        if role == "admin" or not text:
            continue
        texts[i] = text
        rep.findings += analyze_text(text, i, role)
    if not rep.findings:
        return rep
    def joking_near(f: Finding) -> bool:  # laughter/«бро» next to THIS word, not anywhere in a long message
        t = texts[f.message]
        win = t[max(0, f.start - 40):min(len(t), f.end + 40)]
        return bool(LAUGHTER.search(win) or FRIENDLY.search(win))

    joke_flags = {id(f): joking_near(f) for f in rep.findings}
    for f in rep.findings:
        if joke_flags[id(f)] and f.direction in {"author_expressive", "author_to_addressee"} and f.type in {PROFANITY, INSULT}:
            f.severity = max(1, f.severity - 1)
            f.confidence = round(f.confidence * 0.8, 2)
    rep.has_abuse = True
    rep.types = sorted({f.type for f in rep.findings}, key=lambda t: -SEVERITY[t])
    rep.max_severity = max(f.severity for f in rep.findings)
    rep.reported_abuse = any(f.direction == "quoted_or_reported" for f in rep.findings)
    rep.author_abusive = any(f.direction == "author_to_addressee" and f.type != PROFANITY for f in rep.findings)
    rep.accused_abusive = any(f.direction == "accused_reply" for f in rep.findings)
    rep.real_life_threat = any(f.type == THREAT and f.real_life for f in rep.findings)
    rep.languages = sorted({f.lang for f in rep.findings if f.lang != "und"})
    rep.mixed_language_abuse = len({LANG_NAME_RU.get(x, x) for x in rep.languages}) >= 2
    own = [f for f in rep.findings if f.direction in {"author_to_addressee", "author_expressive"}]
    joking = bool(own) and all(joke_flags[id(f)] and f.type in {PROFANITY, INSULT} for f in own)
    rep.tone = "joking" if joking else ("hostile" if rep.author_abusive else "neutral")
    rep.confidence = max(f.confidence for f in rep.findings)
    kinds = ", ".join(TYPE_RU[t] for t in rep.types)
    if rep.reported_abuse:
        rep.explanation.append(f"Автор приводит или описывает чужие слова ({kinds}): это сведения о поведении другого игрока, автор не считается нарушителем")
    if rep.accused_abusive:
        rep.explanation.append("В комментарии обвиняемого есть грубые выражения")
    if rep.author_abusive:
        rep.explanation.append("Сам автор обращения использует оскорбления в адрес собеседника")
    elif any(f.direction == "author_expressive" for f in rep.findings):
        rep.explanation.append("Автор использует нецензурные слова как эмоцию, без адресата")
    if rep.tone == "joking":
        rep.explanation.append("Есть признаки шутки или дружеского общения (смех, «бро»): серьёзность снижена, оценку даёт человек")
    if rep.real_life_threat:
        rep.explanation.append("Угроза связана с реальной жизнью: нужен человек")
    elif any(f.type == THREAT for f in rep.findings):
        rep.explanation.append("Угроза похожа на игровую (контекст игры или без признаков реальной жизни)")
    if rep.mixed_language_abuse:
        rep.explanation.append("Грубые слова на нескольких языках: " + ", ".join(dict.fromkeys(LANG_NAME_RU.get(x, x) for x in rep.languages)))
    forms = list(dict.fromkeys(f.form for f in rep.findings if f.form))
    if forms:
        rep.explanation.append("Слова записаны чужим алфавитом: " + "; ".join(forms))
    if any(f.obfuscated for f in rep.findings):
        rep.explanation.append("Часть слов замаскирована (разделители, похожие символы, цифры): найдены после нормализации")
    return rep
