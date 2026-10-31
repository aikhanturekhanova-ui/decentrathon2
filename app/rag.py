"""Choose which rule points may be cited for a ticket.

Two independent checks must pass before a point reaches `draft_reply.cited`:
  1. EXISTENCE  - the point resolves in the loaded rules.json (section-aware, see knowledge.py);
  2. RELEVANCE  - (a) the point is a curated candidate for the detected category/topic AND its
                  real text contains the anchor words that justify the curated mapping, or
                  (b) a player-claimed point is retrieved by BM25 for the ticket text.
A rule number written by the player is never trusted by itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .classify import QUESTION_TOPICS, Analysis
from .knowledge import RuleBook, RuleDoc

# Anchor = regex that MUST occur in the real rule text, otherwise our curated meaning of the
# point is stale (e.g. organizers renumbered the rules) and the point is not cited.
ANCHORS: dict[str, str] = {
    "2.1": r"оскорблен",
    "2.2": r"пропаганд",
    "2.3": r"реклам",
    "2.4": r"расизм|ксенофоб",
    "2.5": r"аудио|микрофон",
    "2.6": r"IC и OOC|OOC",
    "1.4": r"сохранность",
    "1.6": r"никнейм",
    "3.1": r"репорт",
    "3.3": r"мешать работе",
    "3.4": r"администрац",
    "4.1": r"махинац|мошеннич",
    "4.2": r"вымогательств",
    "4.4": r"реальные деньги",
    "5.2": r"стороннего программного",
    "5.3": r"RP-ситуац|логическ",
    "5.4": r"жильц|дом",
    "5.9": r"ивент|мероприят",
    "GZ 1.1": r"Зелён",
    "GZ 1.2": r"Зелён",
    "ЖБ 1.1.2": r"не предназначен",
    "ЖБ 1.2.2": r"Доказательства",
    "ЖБ 1.2.7": r"оскорблен",
    "ЖБ 1.2.9": r"дублирует|спам",
    "ЖБ 1.3.1": r"Жалоба должна содержать",
    "ЖБ 1.3.4": r"общие формулировки|разберитесь",
    "ЖБ 1.2.6": r"48 часов",
    "ЖБ 1.2.1": r"платформ",
}

# category -> ordered list of (cite, trigger regex over the cleaned text, or None = default)
_R = lambda p: re.compile(p, re.I | re.UNICODE)  # noqa: E731

CRIME_TRIGGER = _R(r"верев|связал|связыва|связан|похит|похищ|мешок|мешк|грабеж|ограбил|заложник|kidnap|secuestr|kishan|arqon|\bqop")
KILL_TRIGGER = _R(r"убил|убий|убит|убива|убьют|расстрел|застрел|стрел[яь]|стрельб|урон|ранил|ударил|избил|сбил|сбив|переех|задавил|(?<![\w])(?:дм|дб|тк|ск|dm|db|tk|sk)(?![\w])|kill|o['`ʻ]ldir|өлтір|смерт|(?<![\w])(?:зз|гз|gz|zz)(?![\w])|зелен\w*\s+зон|green\s*zone")

CANDIDATES: dict[str, list[tuple[str, re.Pattern | None]]] = {
    "complaint.insult": [
        ("2.6", _R(r"угро[зж]|доксинг|деанон|личн\w+\s+информ|адрес|swat|сват|tahdid")),
        ("2.4", _R(r"расиз|расист|нацизм|нац\w*\s+рознь|религ|чурк|хач|негр|ксенофоб|racis")),
        ("2.1", None),
    ],
    "complaint.cheat": [("5.2", None)],
    "complaint.dm": [
        ("GZ 1.2", CRIME_TRIGGER),
        ("GZ 1.1", KILL_TRIGGER),
        ("5.3", _R(r"уход\w*\s+от\s+(?:рп|rp|смерт|ситуац|перестрел)|ливнул|вышел\s+из\s+игр|афк|afk")),
        ("5.4", _R(r"у\s+вход\w*|(?:у|возле|около)\s+(?:мо\w+\s+|его\s+|её\s+)?дом\w*|на\s+территори\w*\s+(?:мо\w+\s+)?дом|подъезд|квартир|особняк|коттедж")),
    ],
    "complaint.scam": [("4.2", _R(r"вымог|угрож\w+\s+(?:за|чтоб)")), ("4.1", None)],
    "complaint.rmt_ads": [
        ("4.4", _R(r"за\s+реал|реальн\w+\s+деньги|за\s+(?:руб|тенге|сум|доллар)|донат|продаж\w*\s+(?:акк|аккаунт)|продаю|продает|продаёт|selling|сатады")),
        ("2.3", None),
    ],
    "complaint.admin": [("3.4", None)],
    "complaint.other": [
        ("2.2", _R(r"18\s*\+|порн|наркот|пропаганд")),
        ("2.5", _R(r"музык|микрофон|крич|орет|орёт|шум")),
        ("5.9", _R(r"ивент|мероприят|мешает\s+(?:провод|участ)")),
        ("1.6", _R(r"никнейм|(?<![\w])ник(?![\w])|имя\s+персонаж")),
        ("3.3", _R(r"фарм\s*репорт|флуд\w*\s+(?:в\s+)?репорт|мешает\s+работ")),
    ],
    "complaint.invalid": [],  # chosen by reason (see select_rules)
    # questions about a rule are answered from the matching rule point (topics are defined in classify.py)
    "report.question": list(QUESTION_TOPICS),
    "report.help": [("3.1", _R(r"телепорт|(?<![\w])порт(?![\w])|(?<![\w])тп(?![\w])|дай\w*\s+(?:деньги|валют|уровень|лвл|машин|оружи)|выдайте|верните\s+на"))],
    "report.complaint": [],
    "report.account": [("1.4", None)],
    "report.payment": [],
    "report.offtopic": [("3.1", _R(r"флуд|дай\w*\s+(?:деньги|валют|уровень|лвл)|выдайте"))],
}

# EXISTENCE of a point (+ its anchor words) is one check, APPLICABILITY is another. A place-bound rule needs THREE facts of the story,
# each checked on its own:  (1) the place - a Green Zone named as such or an object of the official list in rules.json (section 32, 2.x),
# in the right moment of the story (not "вышли из ЗЗ, после чего ..." / "вне ЗЗ" / "сначала были в ЗЗ, потом ...");  (2) the ACTION the
# point forbids (GZ 1.1: killing or damage; GZ 1.2: kidnapping, robbery, ropes) and (3) that the action is not denied ("убийства не было").
# An unknown place is never assumed to be a Green Zone: the draft asks for it. A hospital is a zone only for the cities the list names.
GZ_WORDS = [["зз"], ["гз"], ["gz"], ["zz"], ["зелен", "зон"], ["green", "zone"], ["zona", "verde"], ["yashil", "zona"], ["жасыл", "аймақ"], ["жасыл", "зона"]]
GZ_CTX = _R(r"(?<![\w])(?:зз|гз|gz|zz)(?![\w])|зелен\w*\s+зон\w*|green\s*zone|zona\s+verde|yashil\s+zona\w*|жасыл\s+(?:аймақ|зона)\w*")  # (explicit words only)
_LEAVE_STEMS = ("вышл", "вышел", "выход", "покин", "уехал", "ушли", "ушел", "отошл", "выех", "outsi", "left", "fuera", "fora", "выбег", "выгнал")  # within 4 words before
_LEAVE_NEAR = ("вне", "не", "за")  # only as the word right before ("не в ЗЗ", "вне ЗЗ", "за пределами ЗЗ")
_SEQ_AFTER = _R(r"^(?:\w+\s+)?(?:потом|затем|после\s+чего|после\s+этого|после\s+выхода|позже|спустя|then|after)(?![\w])")
KILL_DAMAGE = _R(r"убил\w*|убий\w*|убит\w*|убива\w*|убьют|расстрел\w*|застрел\w*|стрел[яь]\w*|стрельб\w*|выстрел\w*|подстрелил\w*|урон\w*|ранил\w*|ранен\w*|избива\w*|(?<![\w])(?:из)?бил\s+меня|ударил\w*|избил\w*|сбил\w*|сбив\w*|переех\w*|задавил\w*|"
                 r"(?<![\w])(?:дм|дб|тк|ск|dm|db|tk|sk)(?![\w])|kill\w*|shot(?![\w])|o['`ʻ]ldir\w*|өлтір\w*|мертв\w*|смерт\w*")
NEG_KILL = _R(r"(?:не\s+было|без)\s+(?:\w+\s+){0,1}?(?:убийств\w*|урон\w*|стрельб\w*|смерт\w*)|(?:убийств\w*|урон\w*|смерт\w*)\s+не\s+(?:было|нанос\w*)|"
              r"никто\s+не\s+(?:убил\w*|умер\w*|погиб\w*)|не\s+(?:убил\w*|убива\w*|стрел\w*|бил\w*|ударил\w*)|no\s+(?:one\s+)?(?:was\s+)?kill\w*|nobody\s+(?:was\s+)?(?:killed|died)")
CRIME = _R(r"верев\w*|связал\w*|связыва\w*|связан\w*|похит\w*|похищ\w*|мешок|мешк\w*|грабеж\w*|ограбил\w*|заложник\w*|kidnap\w*|secuestr\w*|kishan\w*|arqon\w*|(?<![\w])qop(?![\w])")
NEG_CRIME = _R(r"не\s+(?:связыва\w*|связал\w*|похищ\w*|похит\w*|граб\w*|ограбил\w*)|(?:похищени\w*|ограблени\w*|связыван\w*|связывани\w*|верев\w*|мешк\w*)\s+не\s+(?:было|использ\w*)")
ACTION_FACTS: dict[str, tuple[re.Pattern, re.Pattern]] = {"GZ 1.1": (KILL_DAMAGE, NEG_KILL), "GZ 1.2": (CRIME, NEG_CRIME),
                                                         # 5.4 (real text): "убийство или нанесение урона жильцам … у входа в дом или квартиру"
                                                         "5.4": (KILL_DAMAGE, NEG_KILL)}
HOUSE_PLACE = _R(r"у\s+вход\w*|у\s+(?:мо\w+\s+|его\s+|её\s+)?дом\w*|возле\s+(?:мо\w+\s+)?дом\w*|около\s+(?:мо\w+\s+)?дом\w*|на\s+территори\w*\s+(?:мо\w+\s+)?дом\w*|подъезд\w*|квартир\w*|у\s+(?:мо\w+\s+)?(?:хаты|особняк\w*|коттедж\w*)")
CONTEXT_REQUIRED = ACTION_FACTS  # cites that need facts (kept under the old name for the audit scripts)
_PLACE_GENERIC = _R(r"(?<![\w])(?:улиц\w*|дорог\w*|трасс\w*|лес(?:у|ом|а|е)?|пляж\w*|город\w*|центр\w*|район\w*|двор\w*|подъезд\w*|парк\w*|заправк\w*|гараж\w*|"
                    r"шахт\w*|ферм\w*|порт(?:у|а|е)?|аэропорт\w*|дом(?:а|у|е|ом)?|квартир\w*|горах?|озер\w*|мор[еяю]|стройк\w*|склад\w*|клуб\w*|бар(?:е|а)?|ст[ао]ян\w*)(?![\w])")
_STOP_PLACE = {"г", "работа", "всех", "зданий", "кроме", "во", "время", "их", "проведения", "зоны", "и", "в"}
_REF_TOKEN = _R(r"(?:жб|zhb|jb|gz|гз|зз|zz)\s*\d{1,2}(?:\s*[.,]\s*\d{1,2}){1,3}|\d{1,2}(?:\s*[.,]\s*\d{1,2}){1,3}\s*(?:гз|зз|gz)")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _stem(w: str) -> str:
    """Prefix stem tolerant to the ending of a Russian word: «Банки» matches банк/банка/банке/банкомат, «больница» matches больницы/больнице."""
    return w if len(w) <= 4 else w[: min(5, max(4, len(w) - 2))]


def zone_places(rb: RuleBook) -> list[dict]:
    """The official list of Green Zone objects, parsed from rules.json (section 32, points 2.x: «• Больница г. Арзамас», «• Банки», …).
    Parsed at load time, nothing is hard-coded. A city after «г.» is a REQUIRED part of the match («Больница г. Арзамас»)."""
    cached = getattr(rb, "_zone_places", None)
    if cached is not None:
        return cached
    places: list[dict] = []
    for d in rb.docs:
        if d.section_id != 32 or not d.ref.startswith("2."):
            continue
        # the real file keeps the bullets on one line («Государственные учреждения: • Правительство • Больница г. Арзамас • …»)
        items = d.body.split("•")[1:] if "•" in d.body else [ln.lstrip(" -*") for ln in d.body.split("\n")[1:]]
        for item in items:
            name = item.strip()
            if not name:
                continue
            main = re.sub(r"\([^)]*\)", " ", name)
            cm = re.search(r"г\.\s*([^\s,]+)", main)
            city = _stem(cm.group(1).lower().replace("ё", "е")) if cm else None
            words = [w for w in _WORD.findall(main.lower().replace("ё", "е")) if w not in _STOP_PLACE and not (city and _stem(w) == city)]
            if words:
                places.append({"name": name, "main": [_stem(w) for w in words], "city": city, "explicit": False})
    try:
        rb._zone_places = places  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return places


@dataclass
class ZoneFacts:
    established: bool = False        # the violation's place is a Green Zone, in the right moment of the story
    mentioned: bool = False          # a zone or a listed object is named at all (even if it was left before the event)
    partial: list = field(default_factory=list)  # listed objects named without the required city ("больница")
    place_stated: bool = False       # some place is stated (zone or not)


def _positions(tokens: list[str], stems: list[str]) -> list[int]:
    out = []
    for i, t in enumerate(tokens):
        if t.startswith(stems[0]) and (len(stems[0]) > 3 or t == stems[0]):
            if all(any(abs(j - i) <= 4 and tokens[j].startswith(sx) for j in range(max(0, i - 4), min(len(tokens), i + 5))) for sx in stems[1:]):
                out.append(i)
    return out


def _origin(tokens: list[str], j: int) -> bool:
    """The city at position j describes where the AUTHOR is from or lives, not where the event happened."""
    before = tokens[max(0, j - 3): j]
    return bool(before) and (before[-1] in {"из", "родом", "from"} or "живу" in before or "живем" in before or "проживаю" in before)


def zone_facts(clean: str, rb: RuleBook | None = None) -> ZoneFacts:
    """Place facts of the story. A rule number typed by the player ("GZ 1.2") names the rule, not the place."""
    facts = _REF_TOKEN.sub(" ", clean)
    tokens = _WORD.findall(facts.lower().replace("ё", "е"))
    z = ZoneFacts(place_stated=bool(_PLACE_GENERIC.search(facts)))
    mentions: list[int] = []
    for stems in GZ_WORDS:
        mentions += _positions(tokens, stems)
    for p in (zone_places(rb) if rb is not None else []):
        pos = _positions(tokens, p["main"])
        if not pos:
            continue
        if p["city"]:
            # the city must belong to the OBJECT ("больница г. Арзамас"), not to the author ("сам я из Арзамаса", "живу в Арзамасе")
            pos = [i for i in pos if any(tokens[j].startswith(p["city"]) and not _origin(tokens, j) for j in range(max(0, i - 4), min(len(tokens), i + 6)))]
            if not pos:
                z.partial.append(p["name"])
                z.place_stated = True
                continue
        mentions += pos
    if mentions:
        z.mentioned = z.place_stated = True
    text_after = lambda i: " ".join(tokens[i + 1: i + 4])  # noqa: E731
    for i in mentions:
        before = tokens[max(0, i - 4): i]
        if any(w.startswith(st) for w in before for st in _LEAVE_STEMS) or any(w in _LEAVE_NEAR for w in tokens[max(0, i - 2): i]):
            continue  # "вышли из ЗЗ", "вне ЗЗ", "не в ЗЗ", "за пределами": the event is not inside
        if _SEQ_AFTER.search(text_after(i)):
            continue  # "были в ЗЗ, потом ...": the event came later, the place is not established
        z.established = True
        break
    return z


# A story is split into clauses: sentences, and ", а / но / потом / после / сам …" inside a sentence. The forbidden action and the
# Green Zone must be facts of the SAME clause: «Похитили в лесу. В ЗЗ я приехал после освобождения» is a kidnapping in a forest followed
# by a later visit to the zone, not a kidnapping in the zone. A negation applies to ITS action only: «не убил, а ранил выстрелом в ЗЗ» is
# damage in the zone (GZ 1.1 forbids killing AND damage), «связал и убил в ЗЗ, ограбления не было» is still tying + killing.
_CLAUSE_SPLIT = re.compile(r"(?:(?<!\d)\.|\.(?!\d)|[!?;\n])+|,\s*(?=(?:а|но|однако|потом|затем|после|сам|сама|когда|зато|хотя|but|then|after|later)(?![\w]))", re.I)
_NEG_BEFORE_ACTION = re.compile(r"(?:^|\s)(?:не|ни|без|никто\s+не|никого\s+не|not|never|didn't|did\s+not|no)\s+(?:\w+\s+)?$", re.I)
_NEG_AFTER_NOUN = re.compile(r"^\s*(?:\w+\s+)?не\s+(?:было|случилось|произошло|были)|^\s*(?:\w+\s+)?(?:did\s+not|didn't)\s+happen", re.I)


def clauses(text: str) -> list[str]:
    from .textutil import clean_for_matching

    text = _REF_TOKEN.sub(" ", (text or "").lower())  # a typed rule number ("GZ 1.2") is neither a place nor a sentence boundary
    text = re.sub(r"(?<![\w])(г|гор|п|ул|д|ст|пр|пос|им|т)\.\s*", r"\1 ", text)  # "больница г. Арзамас": an abbreviation, not the end of a sentence
    parts = [p for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
    return [clean_for_matching(p) for p in parts] or [clean_for_matching(text or "")]


def positive_action(clause: str, rx: re.Pattern) -> bool:
    """At least one match of the action that is not negated in place ("не убил", "убийства не было", "без убийства")."""
    for m in rx.finditer(clause):
        if _NEG_BEFORE_ACTION.search(clause[max(0, m.start() - 30): m.start()]) or _NEG_AFTER_NOUN.search(clause[m.end(): m.end() + 30]):
            continue
        return True
    return False


def context_ok(cite: str, clean: str, rb: RuleBook | None = None, text: str | None = None) -> bool:
    """A place-bound point applies only when ONE clause of the story holds the place (a Green Zone, in the right moment) AND the forbidden
    action, not negated. `text` (with punctuation) gives the clauses; without it the whole cleaned text is one clause."""
    spec = ACTION_FACTS.get(cite)
    if spec is None:
        return True
    action, _denied = spec
    for c in (clauses(text) if text else [clean]):
        facts = _REF_TOKEN.sub(" ", c)
        if not positive_action(facts, action):
            continue
        if cite.startswith("GZ") and zone_facts(c, rb).established:
            return True
        if cite == "5.4" and HOUSE_PLACE.search(facts):
            return True
    return False


# jargon -> canonical words so that BM25 can match slang against the formal rule text
JARGON: list[tuple[re.Pattern, str]] = [
    (_R(r"оск\w*|osk\w*|ock|оскор\w*|упом\w*\s+род\w*|род(?:ни|них|ных)|мам[ауыие]|мат(?![\w])|so['`ʻ]?k\w*|haqorat\w*|qorla\w*"), "оскорбление игрока родственников чат"),
    (_R(r"(?<![\w])(?:дм|dm|дб|db|тк|tk|ск|sk)(?![\w])|убил\w*|убий\w*|убива\w*|kill\w*"), "убийство нанесение урона"),
    (_R(r"(?<![\w])(?:зз|гз|gz|zz)(?![\w])|зелен\w*\s+зон\w*"), "Зелёная Зона запрещены убийства нанесение урона"),
    (_R(r"верев\w*|связал\w*|похит\w*|мешк\w*"), "верёвки похищения Зелёная Зона криминальные действия"),
    (_R(r"уход\w*\s+от\s+\w+"), "RP-ситуацию логического завершения уход выход из игры"),
    (_R(r"софт\w*|чит\w*|cheat\w*|аим\w*|hack\w*|flyhack|speed"), "стороннее программное обеспечение читы"),
    (_R(r"скам\w*|scam\w*|обман\w*|кинул\w*|мошенн\w*|aldab\w*|aldadi"), "мошенничество махинации игровая валюта"),
    (_R(r"вымог\w*"), "вымогательство"),
    (_R(r"реклам\w*|пиар\w*|тг\s*канал|телеграм\w*|дискорд\w*"), "реклама сторонние сайты социальные сети"),
    (_R(r"за\s+реал\w*|реальн\w+\s+деньг\w*|донат\w*"), "сделки за реальные деньги имущество аккаунты валюта"),
    (_R(r"админ\w*|admin\w*"), "администрации жалобы на администраторов"),
    (_R(r"флуд\w*|спам\w*"), "флуд репорт"),
    (_R(r"музык\w*|микрофон\w*"), "аудиотрансляции микрофон"),
    (_R(r"порн\w*|18\s*\+"), "порнография запрещённые темы"),
]


# game jargon -> the term defined in the rulebook section «RP термины» (first match wins; the terms have no numbered refs,
# so they are quoted by name and listed in `draft_reply.terms`, never in `cited`)
TERM_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("SK", _R(r"(?<![\w])(?:ск|sk|spawn\s*kill|спавн\s*кил\w*|спавнкил\w*)(?![\w])")),
    ("DB", _R(r"(?<![\w])(?:дб|db|drive\s*by|драйв\s*бай|драйвбай)(?![\w])|таран\w*|сбил\w*\s+(?:на\s+)?машин\w*|сбив\w*\s+машин\w*|переех\w*|задавил\w*|atropell\w*")),
    ("TK", _R(r"(?<![\w])(?:тк|tk|team\s*kill|тим\s*кил\w*)(?![\w])")),
    ("RK", _R(r"(?<![\w])(?:рк|rk|revenge\s*kill|реве[нн]дж\w*)(?![\w])|месть|мстил\w*")),
    ("MG", _R(r"(?<![\w])(?:мг|mg|metagam\w*|метагейм\w*|метагам\w*)(?![\w])")),
    ("PG", _R(r"(?<![\w])(?:пг|pg|powergam\w*|павергейм\w*)(?![\w])")),
    ("NonRP", _R(r"(?<![\w])(?:нон\s*рп|nonrp|non\s*rp|нрп|nrp)(?![\w])|нон-рп")),
    ("DM", _R(r"(?<![\w])(?:дм|dm|дэм|massdm|mass\s*dm|death\s*match|дезматч)(?![\w])|убил\w*\s+без|без\s+причин\w*\s+убил|убийств\w*\s+без|dmq\w*|дмит\w*|дмят")),
]
KILL_WORDS = _R(r"убил\w*|убива\w*|убийств\w*|расстрел\w*|застрел\w*|стрел[яь]\w*|kill\w*|killed|matar\w*|mat[óo](?![\w])|matou|o['`ʻ‘’]?ld[ir]\w*|otib\w*|өлтір\w*")


def select_terms(a: Analysis, rb: RuleBook) -> list:
    """Pick at most one defined game term (DM/DB/SK/TK/RK/MG/PG/NonRP) that matches the player's jargon and exists in the rulebook."""
    from .phrases import TERM_ANCHORS, TERM_GISTS

    from .classify import TERM_ANY

    eff = a.topic if (a.category == "report.complaint" and a.topic) else a.category
    if a.category not in {"complaint.dm", "complaint.other", "report.complaint", "report.question"}:
        return []
    if a.category == "report.question" and not TERM_ANY.search(a.clean):
        return []  # a question gets a definition only when it names a defined term ("что такое DM?")
    chosen = None
    for abbr, rx in TERM_TRIGGERS:
        if rx.search(a.clean):
            chosen = abbr
            break
    if chosen is None and eff == "complaint.dm" and KILL_WORDS.search(a.clean) and not GZ_CTX.search(a.clean):
        chosen = "DM"  # a plain "убил без причины" is exactly what DeathMatch defines
    term = rb.terms.get(chosen) if chosen else None
    if term is None or chosen not in TERM_GISTS or not re.search(TERM_ANCHORS.get(chosen, ""), term.definition, flags=re.I):
        return []
    return [term]


@dataclass
class RagResult:
    terms: list = field(default_factory=list)
    cited: list[RuleDoc] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    claimed_ok: list[str] = field(default_factory=list)
    claimed_irrelevant: list[str] = field(default_factory=list)
    claimed_unknown: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    zone_hint: str = ""  # '' | 'ask_place' (no place stated) | 'ask_city' (a listed object named without its required city)


def build_query(a: Analysis) -> str:
    extra = [words for rx, words in JARGON if rx.search(a.clean)]
    return a.clean + " " + " ".join(extra)


def _anchored(rb: RuleBook, cite: str) -> RuleDoc | None:
    doc = rb.resolve_cite(cite)
    if doc is None:
        return None
    anchor = ANCHORS.get(cite)
    if anchor and not re.search(anchor, doc.body, flags=re.I):
        return None
    return doc


def select_rules(a: Analysis, rb: RuleBook, no_evidence: bool = False, max_cites: int = 2) -> RagResult:
    res = RagResult()
    if not rb.docs:  # rules.json was not found: citing is impossible, say so instead of calling every player claim "unknown"
        res.notes.append("rules_missing")
        return res
    res.terms = select_terms(a, rb)
    query = build_query(a)
    hits = rb.search(query, top_k=5)
    res.retrieved = [{"cite": d.cite if d.citable else None, "section": d.section_title, "ref": d.ref, "score": round(s, 2),
                      "citable": d.citable} for d, s in hits]
    top_keys = {d.key for d, _ in hits[:5]}

    chosen: list[RuleDoc] = []

    def add(doc: RuleDoc | None) -> None:
        if doc is not None and doc.key not in {c.key for c in chosen}:
            chosen.append(doc)

    cat = a.category
    if cat == "complaint.invalid":
        # pick by *why* the ticket is not a proper complaint
        if a.suspected_category or "no_evidence" in a.flags:
            add(_anchored(rb, "ЖБ 1.2.2"))
        elif re.search(r"\?|^(?:как|где|что|когда|почему|можно)", a.clean):
            add(_anchored(rb, "ЖБ 1.1.2"))
        else:
            add(_anchored(rb, "ЖБ 1.3.4"))
        if not chosen:
            add(_anchored(rb, "ЖБ 1.3.1"))
    else:
        # a `report` about a violation is retrieved by the SUBSTANCE of the violation (topic), not by the generic category
        spec_cat = a.topic if (cat == "report.complaint" and a.topic) else cat
        specs = CANDIDATES.get(spec_cat, [])
        if cat == "report.question" and "report.question" not in a.matched:
            specs = []  # a question about game mechanics: do not attach a rule just because a keyword occurs
        # a QUESTION about a rule ('что такое ЗЗ?') describes no event, so the event facts are not required there
        needs_facts = cat != "report.question"
        triggered = [c for c, rx in specs if rx is not None and rx.search(a.clean) and (not needs_facts or context_ok(c, a.clean, rb, a.text))]
        for c in triggered:
            add(_anchored(rb, c))
        if not chosen:
            default = [c for c, rx in specs if rx is None]
            for c in default:
                add(_anchored(rb, c))

    # ---- verify the references the player typed -------------------------------------------
    for ref in a.claimed_refs:
        doc = rb.resolve_cite(ref)
        if doc is None:
            res.claimed_unknown.append(ref)
            continue
        category_map = dict(CANDIDATES.get(spec_cat if cat != "complaint.invalid" else cat, [])) if (cat != "report.question" or "report.question" in a.matched) else {}
        retrieved = doc.key in top_keys
        # a curated candidate with a trigger needs the trigger (or retrieval) to confirm the topic; a place-bound rule also needs the place
        in_category = (
            ref in category_map
            and _anchored(rb, ref) is not None
            and (cat == "report.question" or context_ok(ref, a.clean, rb, a.text))
            and (category_map[ref] is None or bool(category_map[ref].search(a.clean)) or retrieved)
        )
        if in_category:
            res.claimed_ok.append(ref)
            add(doc)
        else:
            # exists in the rules but does not fit the ticket (or is not a curated rule for it):
            # reported to the admin, never quoted. BM25 alone is NOT enough to cite a rule (measured
            # on validation: retrieval-only citations pulled in unrelated points such as the
            # "internal evidence is not shown to players" rule).
            res.claimed_irrelevant.append(ref)

    res.cited = chosen[:max_cites]
    if (a.topic if cat == "report.complaint" else cat) == "complaint.dm" and not any(d.cite.startswith("GZ") for d in res.cited):
        zf = zone_facts(a.clean, rb)
        if zf.partial:
            res.zone_hint = "ask_city"
        elif not zf.place_stated:
            res.zone_hint = "ask_place"
    return res
