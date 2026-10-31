"""Text normalization and language detection (no external dependencies).

Language labels in the dataset are heuristic and noisy (Cyrillic texts labelled 'lat', Uzbek
labelled 'pt'), so this module does its own detection and we never treat dataset labels as truth.
"""
from __future__ import annotations

import re
import unicodedata

# Characters typed on kz servers because the game uses cp1251.
#
# IMPORTANT: this table is derived from the REAL DATA, not copied from the case statement. The PDF/README table says
# ґ=ғ, є=ә, ј=ө, but the words in the data (admin replies and players, e.g. «Сјлеметсіз», «шаєым», «кґмек», «дјлелдеріѕіз»,
# «сўраєыѕызды», «жґндемейміз», «баќылаудамын») only make sense with  ј=ә, є=ғ, ґ=ө  (checked on all kz/kk texts of
# train+valid, see docs/LIMITATIONS.md). The other four letters (ќ=қ, ѕ=ң, ў=ұ, ї=ү) agree in both tables.
GAME_KK_MAP = {
    "ќ": "қ", "Ќ": "Қ",
    "ґ": "ө", "Ґ": "Ө",
    "ѕ": "ң", "Ѕ": "Ң",
    "є": "ғ", "Є": "Ғ",
    "ј": "ә", "Ј": "Ә",
    "ў": "ұ", "Ў": "Ұ",
    "ї": "ү", "Ї": "Ү",
}
KK_TO_GAME = {v: k for k, v in GAME_KK_MAP.items()}


def to_game_chars(text: str) -> str:
    """Encode Kazakh letters the way the game (cp1251) requires, e.g. for replies typed into the game chat."""
    return "".join(KK_TO_GAME.get(ch, ch) for ch in text)
GAME_CHARS = set(GAME_KK_MAP)
KK_LETTERS = set("қғңәөұүһҚҒҢӘӨҰҮҺ")  # unambiguous Kazakh letters (і/І also exists in Ukrainian, so excluded)

_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[^\W\d_]+(?:['’ʻʼ`][^\W\d_]+)*", re.UNICODE)
_TOKEN = re.compile(r"<(?:url|email|handle|digits|nick)>")


def _is_cyr(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"


# --- Uzbek written in Cyrillic ("Аккаунтимни бузишди"): letters ҳ, ў, қ, ғ and typical words/suffixes ------------------------
UZ_CYR_STRONG = {
    "мени", "менга", "ёрдам", "беринг", "йўқ", "йук", "ҳақорат", "сўкди", "сўкишди", "қилди", "қилишди", "учун", "билан", "эмас", "нега", "қандай",
    "қаерда", "тўлов", "тўладим", "берди", "ўйин", "ўйинда", "ўйинчи", "раҳмат", "салом", "бузишди", "бузилди", "аккаунтим", "аккаунтимни",
    "аккаунтимга", "қайтаринг", "шикоят", "ўлдирди", "алдади", "алдаб", "бўлмайди", "йўқолди", "ишламаяпти", "ҳамма", "лекин", "керакми", "борми",
    "пулимни", "пулим", "олди", "кирол", "кира", "олмайман", "олмаяпман",
}
_UZ_CYR_SUFFIX = re.compile(
    r"(?:ишди|ишган|ишибди|япти|япман|япсиз|аяпти|аяпман|олмайман|олмаяпман|имни|ингизни|имизни|лашди|ланди|ганман|ганмиз|майман|маяпти)$")
_UZ_CYR_LETTERS = set("ҳҲ")  # Kazakh has һ, never ҳ
# a word after "игрок / ник / player / ойыншы" is a NAME only when it starts with a capital letter ("игрок Қайрат"), not a pronoun ("ойыншы мені алдады")
_NAME_AFTER = re.compile(
    r"(?<![\w])(?i:игрок\w*|ник\w*|никнейм\w*|нарушител\w*|player|nick(?:name)?|jugador|jogador|o['ʻ’]?yinchi\w*|ойыншы\w*)\s+([A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][^\W\d]*)")
_QUOTED = re.compile(r"«[^»]*»|“[^”]*”|\"[^\"]*\"")
_MID_CAPITAL = re.compile(r"(?<=[^\s.!?…])\s+[A-ZА-ЯЁӘҒҚҢӨҰҮҺІҲЎ][a-zа-яёәғқңөұүһіҳў]+")
RU_STOP = {
    "меня", "мне", "мой", "моя", "мою", "мои", "они", "это", "что", "как", "игрок", "игрока", "игроку", "оскорбил", "убил", "скриншоты", "приложены",
    "приложил", "есть", "нет", "был", "была", "было", "его", "её", "ее", "их", "нас", "вас", "тебя", "чат", "чате", "сервер", "сервере", "помогите",
    "пожалуйста", "пишет", "писал", "сказал", "дали", "взломали", "оскорбляет", "оскорблял", "когда", "потому", "тоже", "очень", "можно", "хочу",
}


def uz_cyr_score(text: str) -> int:
    """How strongly a Cyrillic text looks like Uzbek (0 = not at all). The letter ҳ alone is decisive."""
    t = text.lower()
    score = 3 * sum(1 for c in t if c in _UZ_CYR_LETTERS)
    for w in re.findall(r"[^\W\d_]+", t):
        if w in UZ_CYR_STRONG:
            score += 2
        elif len(w) >= 6 and _UZ_CYR_SUFFIX.search(w):
            score += 2
    return score


def strip_names(text: str) -> str:
    """Remove what must not decide the language: quoted text, nicknames, words after 'игрок/ник/player', capitalized names."""
    t = _QUOTED.sub(" ", text)
    t = _NAME_AFTER.sub(lambda m: m.group(0)[: m.start(1) - m.start(0)] + " ", t)
    t = re.sub(r"\S*_\S*|\S*\d\S*", " ", t)
    t = _MID_CAPITAL.sub(" ", t)
    return t


def normalize_game_chars(text: str, server_lang: str | None = None) -> tuple[str, int]:  # noqa: ARG001
    """Replace cp1251 look-alike letters by Kazakh letters, with context.

    The look-alikes are almost never used in genuine Russian text (the dataset has ~95 such
    tickets in 118k), but є/ї/ґ are normal Ukrainian letters, so a replacement is made only when
    the character sits inside a word that also has other Cyrillic letters. Ukrainian text is not
    supported by the game, and the case statement says the service receives text in this form.
    The caller keeps the original text. Returns (normalized_text, number_of_replacements).
    """
    if not any(ch in GAME_CHARS for ch in text):
        return text, 0
    only_u = {ch for ch in text if ch in GAME_CHARS} <= {"ў", "Ў"}
    if only_u and uz_cyr_score(text) >= 2:
        return text, 0  # ў is a real letter of Uzbek Cyrillic here, not the game's look-alike of kazakh ұ
    out: list[str] = []
    count = 0
    for word in re.split(r"(\s+)", text):
        has_game = any(ch in GAME_CHARS for ch in word)
        cyr_letters = sum(1 for ch in word if _is_cyr(ch) and ch not in GAME_CHARS)
        if has_game and cyr_letters >= 1:
            new = "".join(GAME_KK_MAP.get(ch, ch) for ch in word)
            count += sum(1 for a, b in zip(word, new) if a != b)
            out.append(new)
        else:
            out.append(word)
    return "".join(out), count


def clean_for_matching(text: str) -> str:
    """Lowercase, fold ё/ʻ variants, drop placeholder tokens and punctuation, squeeze repeats."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _TOKEN.sub(" ", t)
    t = t.replace("ё", "е").replace("ʻ", "'").replace("’", "'").replace("ʼ", "'").replace("`", "'")
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)  # "оскорблениееее" -> "оскорбление"
    t = re.sub(r"[^\w'+\-./]+", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


UZ_MARKERS = {
    "salom", "assalomu", "alaykum", "aleykum", "yordam", "yordamga", "bilan", "uchun", "emas", "nega", "bor",
    "yoq", "yo'q", "shikoyat", "shikoyatim", "sokdi", "sokindi", "qil", "qildi", "ber", "bering", "hamma", "lekin",
    "bu", "men", "meni", "mani", "sen", "siz", "admin", "adminlar", "haqorat", "haqoratladi", "kerak", "iltimos",
    "rahmat", "ban", "berdi", "bergan", "oldi", "qoida", "qoidani", "ishlamayapti", "nima", "qanday", "qayerda",
    "menga", "manga", "uni", "ularga", "va", "ham", "yoki", "kerakmi", "bormi", "bo'ldi", "boldi", "otyabdi",
    "yabdi", "moshina", "mashina", "bloklab", "blok", "chora", "tegishli", "odam", "odamni", "kimdir", "sotib",
    "sotyapti", "sotadi", "pul", "oʻzidan", "o'zidan", "ona", "onadan", "xaqorat",
}
EN_MARKERS = {
    "the", "is", "my", "you", "he", "she", "and", "was", "help", "not", "to", "me", "this", "that", "with",
    "player", "please", "killed", "stuck", "hacked", "account", "paid", "bought", "has", "have", "been", "for",
    "in", "on", "of", "cheating", "insulted", "scam", "scammed", "admin", "banned", "ban", "how", "where", "can",
}
ES_MARKERS = {
    "hola", "ayuda", "por", "que", "qué", "el", "la", "los", "me", "mi", "jugador", "gracias", "favor", "no",
    "puedo", "cuenta", "pagué", "compré", "hackearon", "insultó", "mató", "estoy", "atascado", "tengo", "un",
    "una", "es", "con", "para", "cómo", "dónde",
}
PT_MARKERS = {
    "olá", "ola", "ajuda", "não", "nao", "eu", "você", "voce", "jogador", "obrigado", "favor", "preciso",
    "conta", "paguei", "comprei", "hackearam", "insultou", "matou", "estou", "preso", "tenho", "uma", "é",
    "com", "para", "como", "onde", "meu", "minha", "foi",
}
EN_MARKERS -= {"me", "no"}  # shared with Spanish/Portuguese
EN_MARKERS |= {"hello", "hi", "hey", "thanks", "thank", "kill", "stuck", "unfairly", "disagree", "family", "chat", "paid", "donation", "coins", "arrive"}
ES_MARKERS |= {
    "sin", "motivo", "zona", "verde", "mató", "mato", "insultó", "insulto", "estafó", "engañó", "vende", "dinero", "real", "cuentas", "usa", "vuela",
    "por", "aire", "familia", "capturas", "baneó", "baneado", "injustamente", "acuerdo", "estoy", "atascado", "debajo", "hackearon", "pagué", "donación",
    "monedas", "llegaron", "cómo", "hago", "policía", "puede", "insultar", "permitido", "chat", "del", "las", "al", "se", "te", "tu", "su", "muy", "pero",
    "también", "quiero", "necesito", "alguien", "jugadores", "servidor", "coche", "devolvió", "mapa", "ayúdenme", "ayudame", "perdí",
}
PT_MARKERS |= {
    "sem", "motivo", "zona", "verde", "matou", "insultou", "enganou", "vende", "dinheiro", "real", "contas", "usa", "voa", "pelo", "ar", "família",
    "prints", "baniu", "banido", "injustamente", "concordo", "estou", "preso", "embaixo", "hackearam", "paguei", "doação", "moedas", "chegaram",
    "como", "torno", "policial", "permitido", "insultar", "no", "na", "do", "da", "das", "dos", "um", "uma", "seu", "sua", "muito", "mas", "também",
    "quero", "preciso", "alguém", "jogadores", "servidor", "carro", "devolveu", "mapa", "socorro", "perdi", "minha", "meu", "me", "foi", "são", "está",
}
ES_ONLY = {"qué", "cómo", "dónde", "estoy", "puedo", "gracias", "pero", "también", "usted", "mató", "sin", "puede", "tengo", "hola"}
PT_ONLY = {"é", "fui", "quero", "dinheiro", "duas", "vezes", "volta", "meu", "não", "nao", "você", "voce", "obrigado", "estou", "matou", "enganou", "sem", "uma", "preciso", "olá", "ola", "também", "são", "está", "permitido", "doação", "família"}
UZ_MARKERS |= {
    "ishlat", "ishlatyapti", "uchyapti", "havoda", "tiqilib", "qoldim", "toladim", "to'ladim", "tangalar", "kelmadi", "akkaunt", "akkauntimni", "buzishdi",
    "sotyapti", "haqiqiy", "pulga", "rozi", "emasman", "nohaq", "politsiya", "politsiyachi", "bo'ladimi", "boladimi", "kerak", "qilish", "yordam", "bering",
    "salom", "chit", "sababsiz", "o'ldirdi", "oldirdi", "aldadi", "qaytarmadi", "mashina", "pulini", "oilamni", "chatda", "haqorat", "yashil", "zonada",
    "skrin", "bor", "teksturada", "donat", "meni", "ban", "berdi",
}
_UZ_SUFFIX = re.compile(r"(?:yapti|yapman|yapsiz|yotgan|ayotgan|moqda|ibdi|ishdi|ladimi|adimi|ganman|ganmiz|ldim|ladim|tdim|qdim)$")
# Russian written with Latin letters ("osk rodni", "dm bez prichiny")
TRANSLIT_MARKERS = {
    "osk", "ock", "oskorbil", "oskorbleniya", "rodni", "rodnix", "rodnyh", "mamu", "mamy", "bez", "prichiny",
    "pomogite", "pomogi", "pamagite", "zastryal", "ubil", "ubiystvo", "spasibo", "privet", "kak", "gde", "chit",
    "chiter", "soft", "zz", "gz", "dm", "db", "sk", "tk", "nonrp", "nrp", "rp", "frapsa", "fraps",
}
# Kazakh typed in Latin letters: only words that differ from their Uzbek counterparts (salem/salom, kerek/kerak, komek/yordam,
# rakhmet/rahmat …). The data has practically no such tickets, so this is a small, conservative list.
KK_LAT_MARKERS = {
    "salem", "salemetsiz", "kerek", "komek", "kömek", "komektesiniz", "rakhmet", "raqmet", "qaldym", "qaldim", "qorlady", "korlady", "zhok", "joq",
    "oiyn", "oyyn", "oiynshy", "zhane", "magan", "bolady", "bolmaidy", "tusinbedim", "sizder", "adminder", "qazir", "kazir", "ishinde",
}
KK_WORDS = {
    "мен", "бан", "жоқ", "бер", "керек", "үшін", "болды", "және", "емес", "жүр", "істе", "ақша", "ойын", "сен",
    "бұл", "бар", "қой", "көмектесіңіз", "көмектесші", "қалдым", "тұрып", "мені", "маған", "ол", "оны", "неге",
    "қалай", "қайда", "рахмет", "салем", "сәлем", "ереже", "ойыншы", "ақша", "күнге", "берсін", "берек",
    "жасады", "қорлады", "сатып", "жалған", "аккаунт", "кейін", "қамап", "прцнажсіз",
}


def detect_lang(text: str, server_lang: str | None = None) -> str:
    """Return ru | kk | uz | en | es | pt | lat | und."""
    norm, _ = normalize_game_chars(text, server_lang)
    plain = _TOKEN.sub(" ", norm)
    letters = [c for c in plain if c.isalpha()]
    if len(letters) < 2:
        return "und"
    # names, nicknames and quotations must not decide the language ("Меня оскорбил игрок Қайрат" is Russian);
    # if nothing is left after stripping (a one-word message), the unstripped text is used
    stripped = strip_names(plain)
    if sum(1 for c in stripped if c.isalpha()) >= 2:
        plain = stripped
        letters = [c for c in plain if c.isalpha()]
    cyr = sum(1 for c in letters if _is_cyr(c))
    lat = sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ" or c in "ʻʼ")
    ws = words(plain)
    wset = {w for w in ws}

    if cyr >= lat:
        uz_c = uz_cyr_score(plain)
        kk_strong = sum(1 for c in letters if c in "ңәөұүһіҢӘӨҰҮҺІ")  # never Russian, never Uzbek Cyrillic
        kk_shared = sum(1 for c in letters if c in "қғҚҒ")  # Kazakh, but also Uzbek Cyrillic
        kk_hits = sum(1 for w in ws if w in KK_WORDS)
        ru_hits = sum(1 for w in ws if w in RU_STOP)
        if uz_c >= 2 and kk_strong == 0:
            return "uz"
        kk_score = 3 * kk_strong + 2 * kk_shared + kk_hits
        if kk_score >= 2 and kk_score > ru_hits:
            return "kk"
        if (server_lang or "") == "kk" and kk_hits >= 1 and len(ws) <= 3 and ru_hits == 0:
            return "kk"
        if (server_lang or "") == "uz" and uz_c >= 1 and ru_hits == 0:
            return "uz"
        return "ru"

    # Latin script
    es_pt_boost = {"es": len(wset & ES_ONLY) * 2, "pt": len(wset & PT_ONLY) * 2}
    scores = {
        "uz": len(wset & UZ_MARKERS) + sum(1 for w in ws if _UZ_SUFFIX.search(w)) + sum(1 for c in plain if c in "ʻʼ") * 0.5
        + (1 if re.search(r"\b\w*o[\'ʻʼ‘’]\w*", plain) else 0)
        + (1 if re.search(r"\b\w*g[\'ʻʼ‘’]\w*", plain) else 0),
        "en": len(wset & EN_MARKERS),
        "es": len(wset & ES_MARKERS) + es_pt_boost["es"] + (2 if re.search(r"[ñ¿¡]", plain) else 0),
        "pt": len(wset & PT_MARKERS) + es_pt_boost["pt"] + (2 if re.search(r"[ãõç]", plain) else 0),
        "lat": len(wset & TRANSLIT_MARKERS),
        "kk": len(wset & KK_LAT_MARKERS) * 1.5,
    }
    # discount words shared between languages
    if scores["uz"] and (server_lang or "") == "uz":
        scores["uz"] += 1
    best = max(scores, key=lambda k: (scores[k], k == "uz"))
    if scores[best] <= 0:
        if (server_lang or "") == "uz" and len(ws) >= 2:
            return "uz"
        return "und"
    # en vs lat: tokens like 'dm', 'rp' are jargon, not English; require true English markers
    if best == "lat" and scores["en"] >= 2:
        best = "en"
    if best == "en" and scores["lat"] > scores["en"]:
        best = "lat"
    return best
