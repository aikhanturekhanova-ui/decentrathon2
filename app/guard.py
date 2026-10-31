"""Output guard: programmatic checks applied to EVERY draft (template or LLM).

Checks (each one is covered by tests/test_guard.py):
  * every cited point exists in rules.json and passes its anchor (relevance) check;
  * every cited point is actually mentioned in the text, and no uncited "п. X" appears;
  * no affirmative promise of punishment / unban / refund / restoration (all 6 languages);
  * no claim that screenshots or videos were viewed (only their count is known);
  * no URLs except those that occur in the rules, and no invented slash-commands.
A sentence containing a negation ("we cannot promise ...") is not treated as a promise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .knowledge import RuleBook
from .rag import ANCHORS

# Negation only protects a promise if it stands in front of it, in the SAME clause, within three words
# ("не можем обещать, что вернём деньги"). "Не волнуйтесь, мы вернём деньги" and
# "не можем обещать сроки, но вернём деньги" are therefore still promises.
NEG_WORD = (r"(?:не|нет|нельзя|ни|never|cannot|can't|can\s+not|won't|do\s+not|does\s+not|don't|not|no|"
            r"no\s+podemos|nunca|não|nao|nem|jamais)")
NEG_BEFORE = re.compile(r"(?<![\w'])" + NEG_WORD + r"(?![\w'])(?:\s+\S+){0,3}\s*$", re.I)
# Kazakh/Uzbek negate with verb endings that come AFTER the object ("қайтарамыз деп уәде бере алмаймыз")
NEG_AFTER = re.compile(r"^(?:\s+\S+){0,4}?\s+(?:бере\s+алмаймыз|ете\s+алмаймыз|алмаймыз|емес|жоқ|qila\s+olmaymiz|bera\s+olmaymiz|olmaymiz|emas)(?![\w])", re.I)
REASSURE = re.compile(
    r"не\s+(?:волну\w*|переживай\w*|беспоко\w*|сомнева\w*|бойтесь)|don'?t\s+worry|do\s+not\s+worry|no\s+worries|"
    r"no\s+se\s+preocupe\w*|n[aã]o\s+se\s+preocupe\w*|алаңдамаңыз|xavotir\s+olmang|tashvishlanmang", re.I)
CLAUSE_SPLIT = re.compile(
    r",(?!\s*(?:что|чтобы|that|whether)(?![\w]))|[;:—–]|\s-\s|(?<![\w])(?:но|зато|однако|а\s+вот|but|however|though|yet|biroq|lekin|pero|mas|бірақ|алайда)(?![\w])", re.I)

PROMISES = [
    # Russian
    r"(?:будет|будут|получит|получат|понесёт|понесет|понесут)\s+(?:наказан\w*|заблокирован\w*|забанен\w*|мут\w*|бан\w*|варн\w*|блокировк\w*|штраф\w*)",
    r"(?:выдан[аоы]?|выдадим|выдаст)\s+(?:мут|бан|варн|наказани\w*|блокировк\w*)",
    r"(?:накажем|забаним|заблокируем|накажут|забанят|покараем)",
    r"разбан\w*|снимем\s+(?:бан|мут|варн|наказани\w*|блокировк\w*)|снят[аоы]?\s+(?:бан|мут|наказани\w*)",
    r"(?:вернём|вернем|вернут|возместим|возместят|компенсируем|компенсируют|восстановим|восстановят)",
    r"гарантир\w+", r"обязательно\s+(?:накаж|разбер|верн|восстанов)\w*",
    r"жалоб\w+\s+(?:одобрен\w*|удовлетвор\w+)", r"нарушени\w+\s+подтвержден\w*|нарушение\s+подтверждено",
    r"игрок\s+\S+\s+(?:будет|получит)", r"вин[ао]\s+(?:игрока\s+)?доказан\w*",
    r"(?:деньги|средства|донат)\s+(?:вернут\w*|будут\s+возвращен\w*|вернутся)", r"возврат\w*\s+(?:будет|гарантирован\w*|обеспечен\w*)",
    r"верн\w+\s+(?:вам\s+)?(?:деньги|средства|донат|коины|аккаунт|имущество|машину)", r"компенсац\w+\s+(?:будет|получите|вам)",
    r"восстанов\w+\s+(?:вам\s+)?(?:аккаунт|доступ|имущество)",
    # Russian: payouts, compensation, statements that an action has ALREADY been taken
    r"(?:зачисл\w+|начисл\w+|выплат\w+|переведём|переведем|переведут)\s+(?:вам\s+)?(?:компенсаци\w+|деньги|средства|коины|донат|выплат\w+)",
    r"(?:вам\s+)?(?:зачисл|начисл|выплат|компенсир)(?:ят|им|ят)\b", r"компенсаци\w+\s+(?:будет\s+)?(?:зачислен\w*|выплачен\w*|начислен\w*)",
    r"(?:ваш|ваше|ваша)\s+(?:бан|мут|варн|наказание|блокировка|аккаунт)\s+(?:снят\w*|отменен\w*|отменён\w*|восстановлен\w*|разблокирован\w*)",
    r"(?:бан|мут|варн|наказание|блокировка)\s+(?:снят\w*|отменен\w*|отменён\w*)", r"снят[аоы]?\s+(?:бан|мут|варн|наказани\w*|блокировк\w*)",
    r"(?:аккаунт|доступ|имущество|машина)\s+(?:восстановлен\w*|разблокирован\w*|возвращен\w*|возвращён\w*)",
    r"(?:деньги|средства|донат|коины)\s+(?:возвращен\w*|возвращён\w*|зачислен\w*|выплачен\w*)",
    r"(?:игрок|нарушитель)\s+(?:наказан\w*|заблокирован\w*|забанен\w*|замучен\w*)", r"жалоб\w+\s+(?:принят\w*|рассмотрен\w*|одобрен\w*|удовлетворен\w*)",
    r"мы\s+(?:уже\s+)?(?:проверили|наказали|заблокировали|забанили|выдали|отправили|списали|вернули|зачислили|восстановили|разбанили|сняли|исправили)",
    r"вам\s+(?:обязательно\s+)?(?:вернут|зачислят|выплатят|компенсируют|помогут\s+обязательно)", r"завтра\s+вам",
    # English
    r"(?:has|have)\s+been\s+(?:banned|muted|punished|blocked|refunded|unbanned|restored|credited|removed|lifted|compensated)",
    r"(?:your\s+)?(?:ban|mute|penalty|punishment)\s+(?:is|was|has\s+been)\s+(?:lifted|removed|cancel(?:l)?ed|revoked)",
    r"we\s+(?:have|'ve)\s+(?:banned|refunded|restored|credited|unbanned|punished|fixed|removed|checked)", r"will\s+be\s+credited",
    r"(?:you\s+will|you'll)\s+(?:receive|get)\s+(?:a\s+)?compensation", r"compensation\s+will\s+(?:be|arrive)",
    # Kazakh / Uzbek / Spanish / Portuguese: completed actions and payouts
    r"бан\s+алынды|бан\s+алынып\s+тасталды|жаза\s+алынды|ақша\s+қайтарылды|аккаунт\s+қалпына\s+келтірілді",
    r"ban\s+bekor\s+qilindi|ban\s+olib\s+tashlandi|pul\s+qaytarildi|akkaunt\s+tiklandi|kompensatsiya\s+beriladi",
    r"ha\s+sido\s+(?:baneado|reembolsado|restaurado|sancionado)|su\s+(?:ban|sanci[oó]n)\s+(?:ha\s+sido|fue)\s+(?:levantad[oa]|eliminad[oa])|recibir[aá]\s+(?:una\s+)?compensaci[oó]n",
    r"foi\s+(?:banido|reembolsado|restaurado|punido)|seu\s+ban\s+foi\s+(?:removido|cancelado)|receber[aá]\s+(?:uma\s+)?compensa[cç][aã]o",
    # English
    r"(?:we'?ll|we\s+are\s+going\s+to|we\s+shall)\s+(?:ban|refund|restore|unban|compensate|punish|return)", r"money\s+back",
    r"will\s+(?:get|receive)\s+(?:a\s+)?refund", r"(?:refund|compensation)\s+(?:will\s+be|is\s+guaranteed)",
    r"will\s+be\s+(?:banned|muted|punished|blocked|refunded|unbanned|restored|compensated)",
    r"we\s+will\s+(?:ban|mute|punish|block|refund|unban|restore|compensate|reimburse)",
    r"(?:you\s+will|you'll)\s+(?:get|receive)\s+(?:a\s+|your\s+)?(?:refund|money back)",
    r"\bunban\w*\b(?!\s+(?:or|requests?))", r"i\s+guarantee|we\s+guarantee|guaranteed",
    r"complaint\s+(?:is|was|has\s+been)\s+(?:approved|upheld|accepted)",
    # Kazakh
    r"жазаланады|бан\s+беріледі|мут\s+беріледі|қайтарамыз|қалпына\s+келтіреміз|өтейміз|кепілдік\s+береміз",
    # Uzbek
    r"jazolanadi|ban\s+beriladi|mute\s+beriladi|qaytaramiz|tiklaymiz|qoplaymiz|kafolat\s+beramiz",
    # Spanish
    r"ser[aá]\s+(?:baneado|sancionado|castigado)|le\s+(?:devolveremos|reembolsaremos|restauraremos)|garantizamos",
    # Portuguese
    r"ser[aá]\s+(?:banido|punido|castigado)|vamos\s+(?:devolver|reembolsar|restaurar)|garantimos",
    # ---- structural forms: WHO will do WHAT with a sanction/refund (not a list of exact sentences) --------------------
    # Russian: "Администратор отправит вам компенсацию", "Игроку назначат блокировку", "Вы получите средства обратно", "Средства поступят на счёт"
    r"(?:администратор\w*|сотрудник\w*|модератор\w*|поддержка|команда|мы)\s+(?:\w+\s+){0,2}?(?:отправит|отправим|выплатит|выплатим|переведёт|переведет|переведём|переведем|вернёт|вернет|"
    r"зачислит|назначит|выдаст|выдадим|применит|применим|наложит|наложим|снимет|накажет|заблокирует|забанит|восстановит|компенсирует|возместит|исправит|удалит)",
    r"(?:назначат|назначит|назначен\w*|наложат|наложен\w*|применят|применен\w*|вынесут|вынесен\w*)\s+(?:\w+\s+){0,2}?(?:наказани\w*|блокировк\w*|бан\w*|мут\w*|штраф\w*|санкци\w*)",
    r"(?:игроку|нарушителю|ему|ей|им)\s+(?:\w+\s+){0,2}?(?:назначат|выдадут|выдадим|дадут|вынесут)",
    r"получ(?:ите|ишь|ат|ит)\s+(?:\w+\s+){0,2}?(?:средств\w*|деньги|компенсац\w*|возврат\w*|донат\w*|коины)",
    r"(?:средств\w*|деньги|компенсац\w*|возврат\w*)\s+(?:\w+\s+){0,3}?(?:поступят|поступит|вернутся|вернётся|вернется|придут|зачислят|выплатят|будут\s+\w+)",
    # English
    r"(?:admin\w*|staff|moderator\w*|support|the\s+team|we)\s+will\s+(?:\w+\s+){0,2}?(?:send|ban|punish|refund|compensate|restore|unban|block|mute|issue|apply|remove|credit|return|reimburse|delete|pay)",
    r"(?:you\s+will|you'll)\s+(?:\w+\s+){0,2}?(?:get|receive)\s+(?:\w+\s+){0,3}?(?:funds|money|refund|compensation|coins|back)",
    r"(?:funds|money|refund|compensation|coins)\s+will\s+(?:\w+\s+){0,3}?(?:arrive|be|reach|appear|come|go|return)",
    r"(?:the\s+)?(?:player|user|violator|offender)\s+will\s+(?:\w+\s+){0,2}?(?:be|get|receive|face)",
    r"(?:ban|block|mute|penalty|punishment|sanction)\s+will\s+be\s+(?:issued|applied|assigned|imposed|given)",
    # Spanish / Portuguese
    r"(?:el\s+administrador|el\s+equipo|nosotros)\s+(?:\w+\s+){0,2}?(?:enviar[aá]|enviaremos|reembolsar[aá]|devolver[aá]|sancionar[aá]|banear[aá]|restaurar[aá]|compensar[aá])",
    r"recibir[aá]\s+(?:\w+\s+){0,2}?(?:dinero|fondos|reembolso|compensaci[oó]n)", r"(?:el\s+)?jugador\s+(?:ser[aá]|recibir[aá])",
    r"(?:o\s+administrador|a\s+equipe|n[oó]s)\s+(?:\w+\s+){0,2}?(?:enviar[aá]|enviaremos|reembolsar[aá]|devolver[aá]|punir[aá]|banir[aá]|restaurar[aá]|compensar[aá])",
    r"receber[aá]\s+(?:\w+\s+){0,2}?(?:dinheiro|fundos|reembolso|compensa[cç][aã]o)", r"(?:o\s+)?jogador\s+(?:ser[aá]|receber[aá])",
    # Uzbek / Kazakh
    r"(?:pul|mablag['ʻ‘’]?)\w*\s+(?:\w+\s+){0,2}?(?:qaytariladi|qaytadi|tushadi|o['ʻ‘’]?tkaziladi)",
    r"administrator\s+(?:\w+\s+){0,2}?(?:yuboradi|qaytaradi|jazolaydi|bloklaydi|tiklaydi)", r"o['ʻ‘’]?yinchi\s+(?:\w+\s+){0,2}?(?:jazolanadi|bloklanadi)",
    r"(?:ақша|қаражат)\w*\s+(?:\w+\s+){0,2}?(?:қайтарылады|түседі|аударылады)", r"әкімші\s+(?:\w+\s+){0,2}?(?:жібереді|қайтарады|жазалайды|бұғаттайды|қалпына\s+келтіреді)",
    r"ойыншы\s+(?:\w+\s+){0,2}?(?:жазаланады|бұғатталады)",
]
PROMISE_RX = [re.compile(p, re.I | re.UNICODE) for p in PROMISES]

SCREENSHOT_CLAIMS = re.compile(
    r"по\s+скриншот\w*\s+(?:видно|понятно|ясно)|на\s+скриншот\w*\s+(?:видно|виден|видны|запечатлен)|скриншот\w*\s+(?:подтвержда|показыва)\w*"
    r"|мы\s+(?:изучили|просмотрели|посмотрели|проверили|увидели)\s+(?:ваши\s+)?(?:скрин|видео|вложени|доказательств)\w*"
    r"|доказательств\w+\s+(?:подтвержда|достаточн)\w*"
    r"|screenshots?\s+(?:show|confirm|prove|clearly)|(?:we|i)\s+(?:reviewed|checked|viewed|watched|saw)\s+(?:your\s+)?(?:screenshots?|video|attachments?|evidence)"
    r"|from\s+the\s+screenshots"
    r"|skrinshot\w*\s+(?:ga\s+ko[‘'`ʻ]ra|bo[‘'`ʻ]yicha|tasdiqlay)|skrinshot\w*\s+ko[‘'`ʻ]rib\s+chiq\w*\s+(?:dik|ildi)"
    r"|скриншот\w*\s+бойынша|скриншот\w*\s+(?:растайды|көрсетеді)"
    r"|seg[uú]n\s+las\s+capturas|las\s+capturas\s+(?:muestran|confirman)|de\s+acordo\s+com\s+as\s+capturas|as\s+capturas\s+(?:mostram|confirmam)",
    re.I | re.UNICODE,
)
# The service forwards nothing: /triage stores no ticket and nobody is assigned. A draft may say that a person NEEDS to review
# the ticket, but never that it HAS BEEN passed on (in any of the six languages).
TRANSFER_CLAIM = re.compile(
    r"передан[аоы]?(?![\w])|переда(?:ли|л|ём|ем)(?![\w])|направлен[аоы]?(?![\w])|направили|отправлен[аоы]?\s+(?:администратор|сотрудник|в\s+поддержк)\w*|"
    r"(?:has|have)\s+been\s+(?:passed|forwarded|escalated)|(?:we|i)\s+(?:have\s+|'ve\s+)?(?:passed|forwarded|escalated)\s+(?:it|this|your)|passed\s+(?:it|this|your\s+\w+)\s+(?:on\s+)?to\s+(?:an?|the)\s+(?:admin|staff)\w*|"
    r"(?:hemos|he)\s+(?:pasado|derivado|transferido)|se\s+ha\s+(?:pasado|derivado|transferido)|encaminhamos|encaminhei|foi\s+encaminhad[oa]|foram\s+encaminhad[oa]s|"
    r"yubordik|yuborildi|uzatildi|jo['ʻ‘’]?natdik|jo['ʻ‘’]?natildi|жібердік|жіберілді|жөнелттік|тапсырылды", re.I | re.UNICODE)
URL_RX = re.compile(r"https?://[^\s)>,;]+|www\.[^\s)>,;]+", re.I)
SLASH_CMD = re.compile(r"(?<![\w/.:])/[a-zA-Z]{2,}\b")
_RULE_WORD = (r"(?:п\.|пп\.|пункт\w*|ст\.|стать\w*|rule|point|item|clause|section|article|punto|regla|art[ií]culo|ponto|regra|artigo|"
              r"band\w*|тармақ\w*|ереже\w*|qoida\w*)")
CITE_MENTION = re.compile(_RULE_WORD + r"\s*(?:№|no\.?|#|n[ºo°])?\s*(\d+(?:\.\d+){0,3})|(\d+(?:\.\d+){0,3})-(?:тармағы|bandi)", re.I)
DOTTED_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+){1,3})(?![\w])")
# a sentence that names a rule must never say that the forbidden thing is allowed
PERMISSIVE = re.compile(
    r"разрешен\w*|разрешается|дозволен\w*|допускается|можно(?!\s+ли)|allowed|permitted|is\s+legal|permissible|is\s+ok(?:ay)?|"
    r"permitid[oa]s?|se\s+permite|est[aá]\s+permitido|[eé]\s+permitido|permitem|рұқсат\s+етіл\w*|рұқсат\s+берілген|ruxsat\s+etil\w*|ruxsat\s+berilgan|mumkin(?!\s*emas)", re.I)
RULE_TALK = re.compile(
    r"правил\w*|пункт\w*|(?<![\w])п\.|rules?(?![\w])|regulation\w*|reglas?(?![\w])|regras?(?![\w])|ереже\w*|qoida\w*|band(?:i|ida)?(?![\w])|тармақ\w*|"
    r"punto|clause|article|статья|ст\.|ponto|artigo|art[ií]culo", re.I)


@dataclass
class GuardResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


_ABBR = re.compile(r"(?<![\w])(п|пп|ст|no|n|art|№)\.\s+", re.I)


def _sentences(text: str) -> list[str]:
    """Split into sentences without cutting after abbreviations such as 'п. 2.1' (the rule number stays with its sentence)."""
    protected = _ABBR.sub(lambda m: m.group(0).rstrip() + "\x00", text)
    return [s.replace("\x00", " ") for s in re.split(r"(?<=[.!?;])\s+|\n+", protected) if s.strip()]


def find_promises(text: str) -> list[str]:
    bad: list[str] = []
    for s in _sentences(text):
        s = REASSURE.sub(" ", s)  # "не волнуйтесь" is reassurance, not a negation of what follows
        for clause in CLAUSE_SPLIT.split(s):
            for rx in PROMISE_RX:
                hit = False
                for m in rx.finditer(clause):
                    before, after = clause[: m.start()], clause[m.end():]
                    if NEG_BEFORE.search(before) or NEG_AFTER.search(after):
                        continue  # this particular promise is negated
                    bad.append(m.group(0))
                    hit = True
                    break
                if hit:
                    break
    return bad


def check_draft(text: str, cited: list[str], rb: RuleBook, needs_human: bool | None = None, terms: list[str] | None = None) -> GuardResult:
    problems: list[str] = []
    for t in terms or []:  # game terms (DM, DB, ...) must exist in the rulebook's term section and be named in the text
        if t not in rb.terms:
            problems.append(f"term does not exist in the rulebook: {t}")
        elif t not in text:
            problems.append(f"term not mentioned in text: {t}")
    if not text.strip():
        problems.append("empty draft")
    for p in find_promises(text):
        problems.append(f"promise: {p!r}")
    if SCREENSHOT_CLAIMS.search(text):
        problems.append("claims that attachments were reviewed")
    for sent in _sentences(text):
        if asks_for_secret(sent):
            problems.append(f"asks for a secret (password, code, token or card data): {sent[:70]!r}")
    if TRANSFER_CLAIM.search(text):
        problems.append("claims that the ticket was passed on, but nothing is forwarded or assigned")
    allowed_urls = set(rb.urls)
    for u in URL_RX.findall(text):
        if u.rstrip(".,") not in allowed_urls:
            problems.append(f"url not in rules: {u}")
    if SLASH_CMD.search(text):
        problems.append("invented slash command")
    for rx, what in ((EMAIL_RX, "e-mail address"), (PHONE_RX, "phone number"), (HANDLE_RX, "messenger handle")):
        if rx.search(URL_RX.sub(" ", text)):
            problems.append(f"invented contact ({what})")
    for c in cited:
        doc = rb.resolve_cite(c)
        if doc is None:
            problems.append(f"cited rule does not exist: {c}")
            continue
        anchor = ANCHORS.get(c)
        if anchor and not re.search(anchor, doc.body, flags=re.I):
            problems.append(f"cited rule fails relevance anchor: {c}")
        if doc.ref not in text:
            problems.append(f"cited rule not mentioned in text: {c}")
    # a reference to a rule number in the text that is not in `cited`, in ANY form ("п. 5", "Пункт 99.99", "rule 3.1", "5.2-bandi")
    cited_refs = {rb.resolve_cite(c).ref for c in cited if rb.resolve_cite(c)}
    no_urls = URL_RX.sub(" ", text)
    for m in CITE_MENTION.finditer(no_urls):
        ref = m.group(1) or m.group(2)
        if ref not in cited_refs and not _is_claim_note_ref(no_urls, ref):
            problems.append(f"uncited rule number in text: {ref}")
    # any other dotted number (2.1, 99.99, 1.2.3) that is not a cited point is treated as a possible invented rule
    for m in DOTTED_NUMBER.finditer(no_urls):
        ref = m.group(1)
        if ref not in cited_refs and not _is_claim_note_ref(no_urls, ref):
            problems.append(f"uncited dotted number in text (possible invented rule): {ref}")
    # meaning check next to a citation: a sentence naming a rule must not say that something is permitted
    for s in _sentences(no_urls):
        if CITE_MENTION.search(s) and PERMISSIVE.search(s) and not re.search(r"(?<![\w])(?:не|нет|нельзя|not|no|emas|емес|n[aã]o)(?![\w])", s, re.I):
            problems.append(f"sentence cites a rule and says something is allowed: {s[:80]!r}")
    if needs_human is False and re.search(r"администратор\w*\s+(?:проверит|примет)|administrator\s+will", text, re.I):
        problems.append("text hands the case to an administrator but needs_human is false")
    return GuardResult(not problems, problems)


# ----------------------------------------------------------------------------------------------------------------------------
# Free model text (LLM mode). The promise list above is an incomplete blacklist: a new phrasing can always slip through it.
# So the text the MODEL writes itself gets a second, structural layer that does not depend on knowing the exact phrase:
#   * any mention of a sanction / refund / restoration / compensation topic must be NEGATED in its clause ("мы не можем обещать возврат")
#     or be a question; an un-negated mention is rejected whatever the verb is;
#   * no deadlines or time words ("24 часа", "завтра", "soon"), no numbers other than the ones present in the ticket;
#   * no procedures or instructions the facts do not contain ("нажмите", "обратитесь в", "call", "open the ...").
# Sensitive tickets (payment, hacking, lost access, appeals, threats) do not reach the model at all: the pipeline uses the verified
# templates for them. None of this is a proof of safety: it narrows what can slip through, and every draft still goes to a person.
OUTCOME_TOPIC = re.compile(
    r"наказан\w*|накаж\w*|блокир\w*|заблок\w*|(?<![\w])бан(?:а|у|е|ом|ы|ов)?(?![\w])|забан\w*|разбан\w*|(?<![\w])мут\w*|варн\w*|штраф\w*|санкц\w*|деморган\w*|"
    r"возврат\w*|верн\w*|возмещ\w*|возмест\w*|компенс\w*|восстанов\w*|средств\w*|деньг\w*|выплат\w*|зачисл\w*|начисл\w*|"
    r"punish\w*|(?<![\w])ban(?:ned|s)?(?![\w])|unban\w*|block\w*|(?<![\w])mut(?:e|ed)(?![\w])|penalt\w*|sanction\w*|refund\w*|(?<![\w])money(?![\w])|(?<![\w])funds?(?![\w])|"
    r"compensat\w*|restor\w*|reimburs\w*|credit\w*|suspend\w*|"
    r"sanci[oó]n\w*|castig\w*|banea\w*|bloque\w*|reembols\w*|devol\w*|dinero|fondos|compensa\w*|restaur\w*|expuls\w*|"
    r"puni[cç]\w*|ban(?:ido|ir|imento)\w*|dinheiro|fundos|"
    r"jazo\w*|(?<![\w])ban(?![\w])|blok\w*|qaytar\w*|(?<![\w])pul(?![\w])|mablag\w*|kompensat\w*|tikla\w*|qoplash|qoplay\w*|"
    r"жаза\w*|жазала\w*|бұғат\w*|қайтар\w*|ақша\w*|қаражат\w*|өтем\w*|қалпына\w*|айыппұл\w*",
    re.I | re.UNICODE)
TIME_WORDS = re.compile(
    r"сегодня|завтра|послезавтра|скоро|вскоре|немедленно|в\s+течение|в\s+ближайш\w+|в\s+кратчайш\w+|как\s+можно\s+скорее|"
    r"today|tomorrow|soon|shortly|immediately|within|as\s+soon\s+as|right\s+away|hoy|ma[nñ]ana|pronto|inmediatamente|en\s+breve|dentro\s+de|"
    r"hoje|amanh[aã]|em\s+breve|imediatamente|bugun|ertaga|tez\s+orada|darhol|жақын\s+арада|бүгін|ертең|дереу|тез\s+арада|"
    r"(?:пару|пара|несколько|нескольк\w+|couple\s+of|few|several|un\s+par\s+de|algunos|alguns|bir\s+necha|bir\s+juft|бірнеше)\s+(?:\w+\s+)?(?:час\w*|минут\w*|дн\w*|суток|недел\w*|hours?|minutes?|days?|weeks?|horas?|minutos?|d[ií]as?|soat|daqiqa|kun|сағат|минут|күн)|"
    r"полчаса|пол\s*часа|half\s+an\s+hour|an\s+hour|un\s+rato|mgnovenno|мгновенн\w*|быстро|quickly|rapidly|r[aá]pidamente|rapidamente|tezda|tez(?![\w])|тез(?![\w])|"
    r"\d+\s*(?:час\w*|ч(?![\w])|мин\w*|дн\w*|дней|сут\w*|недел\w*|hours?|hrs?|minutes?|mins?|days?|weeks?|horas?|d[ií]as?|semanas?|dias?|soat|kun|daqiqa|hafta|сағат|күн|апта)",
    re.I | re.UNICODE)
PROCEDURE_WORDS = re.compile(
    r"нажмите|перейдите|откройте|зайдите|введите|напишите\s+(?:в|на)|обратитесь\s+(?:в|к|на)|позвоните|отправьте\s+(?:на|в)|"
    r"click|go\s+to|open\s+the|press|call\s+(?:us|the)|e-?mail|contact\s+(?:the|our|us)|tap|visit|"
    r"haga\s+clic|llame|escriba\s+a|clique|ligue|escreva\s+para|bosing|qo['ʻ‘’]?ng['ʻ‘’]?iroq\s+qiling|ochib|басыңыз|хабарласыңыз|ашыңыз",
    re.I | re.UNICODE)
EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RX = re.compile(r"(?<![\d.])\+?\d[\d\s().-]{7,}\d(?![\d])")
HANDLE_RX = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{4,}")


# a negation also covers a short list ("не можем обещать наказание или возврат"): up to five words instead of three
NEG_BEFORE_WIDE = re.compile(r"(?<![\w'])" + NEG_WORD + r"(?![\w'])(?:\s+\S+){0,5}\s*$", re.I)


# ----------------------------------------------------------------------------------------------------------------------------
# WHITELIST for the model's own text. A blacklist of bad phrasings always lags behind a paraphrase («займёт пару часов», «потеряет доступ
# навсегда», «решение уже принято в вашу пользу»), so the model's sentences must belong to a few harmless TYPES, in the draft's language:
#   greeting / thanks · receipt of the message · a request for details (or a question about details) · "a person needs to review the ticket" ·
#   a NEGATED promise ("мы не можем обещать результат").  Every other sentence is rejected, and the verified template is used instead.
# This narrows what the model can say to what the templates already say; it is still a finite lexicon, not a proof.
_GREET = re.compile(r"здравствуй|привет|добр\w+\s+(?:день|вечер|утро)|спасибо|благодар|hello|(?<![\w])hi(?![\w])|dear|thank|hola|gracias|ol[aá]|obrigad|salom|assalomu|rahmat|сәлем|сәлеметсіз|рахмет|алғыс", re.I)
_RECEIPT = re.compile(r"получил\w*|получен\w*|приняли|принято\s+обращен|received|recibimos|hemos\s+recibido|recebemos|recebido|qabul\s+qildik|oldik|алдық|қабылдадық|келіп\s+түсті", re.I)
_OBJECT = re.compile(r"(?<![\w])(?:его|её|ее|это|этого|ваше|it|this|lo|isso|buni|uni|оны|мұны)(?![\w])|обращени|сообщени|жалоб|запрос|ситуаци|материал|ticket|message|complaint|request|report|situation|mensaje|queja|solicitud|situaci|mensagem|reclama|murojaat|xabar|shikoyat|so['ʻ‘’]?rov|өтініш|хабарлам|шағым|сұраныс", re.I)
_ASK_VERB = re.compile(r"укажите|уточните|опишите|приложите|напишите|сообщите|пришлите|расскажите|поясните|please\s+(?:provide|tell|describe|specify|attach|share|give|let\s+us\s+know|clarify)|provide|describe|specify|attach|clarify|"
                       r"indique|aclare|describa|adjunte|informe|cu[eé]ntenos|esclare[cç]a|descreva|anexe|conte|yozing|ko['ʻ‘’]?rsating|tasvirlang|biriktiring|ayting|aniqlashtiring|көрсетіңіз|сипаттаңыз|тіркеңіз|жазыңыз|айтыңыз|нақтылаңыз", re.I)
_DETAIL = re.compile(r"ник|сервер|время|дат[уаы]|мест[оа]|скрин|видео|доказательств|произошло|ситуаци|детал|подробност|когда|где|nickname|nick|server|time|date|place|location|screenshot|video|evidence|what\s+happened|situation|details|when|where|"
                     r"apodo|servidor|hora|fecha|lugar|captura|prueba|qu[eé]\s+ocurri|cu[aá]ndo|d[oó]nde|apelido|hor[aá]rio|data|local|print|o\s+que\s+aconteceu|quando|onde|vaqt|sana|joy|skrinshot|dalil|nima|qachon|qayerda|"
                     r"уақыт|күн|орын|дәлел|не\s+болды|қашан|қайда", re.I)
_PERSON = re.compile(r"администратор|сотрудник|модератор|человек|команда|administrator|admin|staff|moderator|team|person|administrador|equipo|persona|equipe|pessoa|xodim|administrator|mas['ʻ‘’]?ul|әкімші|қызметкер|модератор", re.I)
_REVIEW = re.compile(r"рассмотрит|проверит|изучит|должен\s+(?:рассмотреть|проверить)|нужно\s+рассмотреть|будет\s+рассмотрен\w*|will\s+(?:review|check|look)|needs?\s+to\s+(?:review|check)|should\s+review|"
                     r"revisar[aá]|debe\s+revisar|necesita\s+revisar|analisar[aá]|precisa\s+analisar|deve\s+analisar|ko['ʻ‘’]?rib\s+chiqadi|tekshiradi|ko['ʻ‘’]?rib\s+chiqishi\s+kerak|қарайды|тексереді|қарауы\s+керек|тексеруі\s+керек", re.I)
_ATTRIB = re.compile(r"^\s*(?:по\s+вашим\s+словам|according\s+to\s+you|seg[uú]n\s+(?:usted|lo\s+que\s+dice)|de\s+acordo\s+com\s+(?:voc[eê]|o\s+que\s+voc[eê])|sizning\s+aytishingizcha|сіздің\s+айтуыңызша)", re.I)
_NEG_W = re.compile(r"(?:не\s+(?:можем|могу|обещаем|гарантируем)|нельзя|cannot|can['ʻ‘’]?t|can\s+not|do\s+not\s+promise|don['ʻ‘’]?t\s+promise|unable|no\s+podemos|no\s+prometemos|n[aã]o\s+podemos|n[aã]o\s+prometemos|"
                    r"qila\s+olmaymiz|ayta\s+olmaymiz|bera\s+olmaymiz|va['ʻ‘’]?da\s+qilmaymiz|алмаймыз|уәде\s+бермейміз|айта\s+алмаймыз)", re.I)
_PROM_W = re.compile(r"обещать|гарантировать|сообщить|назвать|заранее|результат|решение|promise|guarantee|outcome|decision|in\s+advance|prometer|garantizar|resultado|decisi|va['ʻ‘’]?da|kafolat|natija|oldindan|уәде|кепіл|нәтиже|алдын\s+ала", re.I)


# A negated promise is accepted only in its proper shape: the negation GOVERNS the promise verb ("не можем обещать/гарантировать/сообщить
# заранее", "cannot promise"), not "a 'не' somewhere in the sentence".
_NEG_PROMISE_STRICT = re.compile(
    r"не\s+(?:можем|могу|будем)\s+(?:\w+\s+){0,3}?(?:обещать|гарантировать|сообщить|назвать|предсказать)|не\s+(?:обещаем|гарантируем)|"
    r"(?:обещать|гарантировать|сообщить|назвать)\s+(?:\w+\s+){0,4}?(?:мы\s+)?не\s+(?:можем|могу)|"
    r"(?:cannot|can['ʻ‘’]?t|can\s+not|do\s+not|don['ʻ‘’]?t|unable\s+to|are\s+not\s+able\s+to)\s+(?:\w+\s+){0,2}?(?:promise|guarantee|predict|tell\s+you)|"
    r"no\s+(?:podemos|prometemos|garantizamos)(?:\s+(?:\w+\s+){0,2}?(?:prometer|garantizar|anticipar))?|n[aã]o\s+(?:podemos|prometemos|garantimos)(?:\s+(?:\w+\s+){0,2}?(?:prometer|garantir|antecipar))?|"
    r"(?:va['ʻ‘’]?da|kafolat)\s+(?:\w+\s+){0,2}?(?:qila\s+olmaymiz|bera\s+olmaymiz|qilmaymiz|bermaymiz)|oldindan\s+(?:\w+\s+){0,2}?(?:ayta|va['ʻ‘’]?da\s+qila)\s+olmaymiz|"
    r"(?:уәде|кепіл)\w*\s+(?:\w+\s+){0,2}?(?:бере|ете)\s+алмаймыз|алдын\s+ала\s+(?:\w+\s+){0,2}?(?:айта|уәде\s+ете)\s+алмаймыз|уәде\s+бермейміз", re.I)


def _negated_promise(t: str) -> bool:
    return bool(_NEG_PROMISE_STRICT.search(t))


# Secrets a support draft must never ask for (and the model may never even mention): passwords, login/OTP/2FA/backup codes, tokens,
# card data. The verified templates WARN about them ("никому не сообщайте пароль") - those are template sentences, not model text.
SECRET = re.compile(
    r"парол\w*|пин[- ]?код\w*|(?<![\w])код\w*|otp|2fa|двухфактор\w*|двухэтап\w*|токен\w*|cvv|cvc|password\w*|passcode\w*|(?<![\w])pin(?![\w])|token\w*|"
    r"verification\s+code|backup\s+code\w*|(?<![\w])code\w*|contrase[ñn]a\w*|c[oó]digo\w*|senha\w*|parol\w*|(?<![\w])kod\w*|құпия\s*сөз\w*|құпиясөз\w*|"
    r"(?<![\w])sms(?![\w])|(?<![\w])смс(?![\w])|номер\w*\s+карт\w*|данн\w+\s+карт\w*|card\s+(?:number|details)|n[uú]mero\s+de\s+(?:la\s+)?tarjeta|n[uú]mero\s+do\s+cart[aã]o|karta\s+raqam\w*|карта\s+нөмір\w*",
    re.I | re.UNICODE)
SECRET_ASK = re.compile(
    r"пришлите|отправьте|сообщите|укажите|напишите|назовите|скиньте|дайте|введите|продиктуйте|send|tell|provide|give|share|enter|type|"
    r"env[ií]e|indique|d[ií]ganos|informe|mande|diga|yuboring|yozing|ayting|kiriting|жіберіңіз|көрсетіңіз|жазыңыз|айтыңыз|енгізіңіз", re.I)
_NEG_ASK = re.compile(r"(?:не|никому\s+не|никогда\s+не|do\s+not|don['ʻ‘’]?t|never|no|n[aã]o|nunca|hech\s+kimga|ешкімге)\s+(?:\w+\s+){0,2}?$", re.I)  # "no nos los envíe"


def asks_for_secret(sentence: str) -> bool:
    """A request for a secret, unless the request verb is negated ("никому не сообщайте пароль", "do not send your password")."""
    if not SECRET.search(sentence):
        return False
    for m in SECRET_ASK.finditer(sentence):
        if not _NEG_ASK.search(sentence[: m.start()]):
            return True
    return False


# Closed vocabulary of the model's own sentences, per draft language: every word must come from the verified templates (app/phrases.py) or
# from a short list of neutral support words (greetings, requests for details, the details themselves). A sentence with any other word is
# rejected: "рассмотрение завершено в вашу пользу", "оппонент отстранён навечно", "игрок потеряет доступ" are refused because "завершено",
# "пользу", "отстранён", "навечно", "потеряет" are not support vocabulary - no list of bad phrasings is needed.
_EXTRA_WORDS = {
    "ru": "здравствуйте добрый день вечер утро привет спасибо благодарим большое за ваше ваш вашу ваши вам вас вы мы нам нас пожалуйста укажите уточните опишите "
          "приложите напишите сообщите пришлите расскажите поясните ник никнейм игрока игрок игроке игроков сервер сервера сервере номер время времени дату дата место "
          "месте где когда как что произошло случилось скриншот скриншоты скрины видео запись записи доказательства подробнее подробно детали подробности ситуацию ситуации "
          "это и или а в на у с по о об этом этого какой какая какое каком каким какую именно примерно точное точно ли также еще ещё нужно нужны нужна для проверки "
          "администратор администратора администратору сотрудник сотрудника рассмотрит проверит обращение обращения обращением сообщение сообщения получили получено "
          "жалобу жалоба жалобы если есть можете могли бы был была было эту этот свой свою ситуация описание обещать гарантировать назвать предсказать решение результат заранее сейчас пока",
    "en": "hello hi dear thank thanks you for contacting us your we please provide tell describe specify attach share give let know clarify the a an and or of in on at to "
          "it is are was were this that what when where which who how exactly approximately approximate also nickname player server time date place location screenshot "
          "screenshots video evidence details situation happened promise guarantee outcome advance administrator staff member will review check received message report complaint ticket if any can could",
    "es": "hola gracias por su sus le nos usted favor indique aclare describa adjunte infórmenos cuéntenos apodo jugador servidor hora fecha lugar captura capturas video "
          "pruebas detalles situación qué cuándo dónde ocurrió pasó prometer garantizar resultado el la los las un una y o de del en a con es este esta esto si puede podría administrador revisará queja "
          "mensaje hemos recibido",
    "pt": "olá obrigado obrigada por sua seu nos você favor informe esclareça descreva anexe conte apelido jogador servidor horário hora data local captura capturas print "
          "prints vídeo provas detalhes situação o que quando onde aconteceu prometer garantir resultado a os as um uma e ou de do da em com é este esta isso se pode poderia administrador analisará "
          "reclamação mensagem recebemos",
    "uz": "assalomu alaykum salom rahmat murojaatingiz murojaatingizni uchun iltimos yozing ko rsating tasvirlang biriktiring ayting aniqlashtiring nik o yinchi server "
          "vaqt vaqtni sana sanani joy joyni nikni serverni skrinshot skrinshotni video dalil dalillarni tafsilot tafsilotlarni nima qachon qayerda bo ldi va yoki bu siz sizning biz administrator rib chiqadi shikoyat shikoyatni xabar oldik",
    "kk": "сәлеметсіз бе сәлем рахмет хабарласқаныңыз үшін өтініш өтінішіңізді көрсетіңіз сипаттаңыз тіркеңіз жазыңыз айтыңыз нақтылаңыз никнейм ойыншы сервер уақыт "
          "күн күнін орын орнын уақытты серверді никнеймді скриншот скриншотты видео дәлел дәлелдерді толығырақ не болды қашан қайда және немесе бұл сіз сіздің біз әкімші қарайды шағым шағымды хабарлама алдық",
}
_VOCAB: dict[str, frozenset] = {}
_TOK = re.compile(r"[^\W\d_]+", re.UNICODE)


def _vocab(lang: str) -> frozenset:
    if lang not in _VOCAB:
        from .phrases import P

        words: set[str] = set()
        for value in P.get(lang, P["ru"]).values():
            words |= {w.lower().replace("ё", "е") for w in _TOK.findall(value)}
        words |= {w.replace("ё", "е") for w in _EXTRA_WORDS.get(lang, "").split()}
        _VOCAB[lang] = frozenset(words)
    return _VOCAB[lang]


def check_llm_sentences(text: str, lang: str = "ru", extra_words: tuple[str, ...] = ()) -> list[str]:
    """Problems of the sentences the MODEL wrote itself (verbatim template sentences are checked by the caller, not here):
    (1) no secret may be asked for or mentioned; (2) every word belongs to the closed support vocabulary of the draft language;
    (3) the sentence is one of the harmless types: greeting/thanks · receipt of the message · request for details · "a person needs to review
    the ticket" · a NEGATED promise in its proper shape. An "according to you" restatement of the player's claim is NOT accepted from the
    model: the verified template already contains it."""
    bad: list[str] = []
    vocab = _vocab(lang) | {w.lower() for w in extra_words}
    for s in _sentences(text):
        t = s.strip()
        if not re.search(r"\w", t):
            continue
        if SECRET.search(t):
            bad.append(f"mentions a secret (password, code, token, card data): {t[:70]!r}")
            continue
        unknown = sorted({w for w in (x.lower().replace("ё", "е") for x in _TOK.findall(t)) if w not in vocab})
        if unknown:
            bad.append(f"words outside the support vocabulary {unknown[:4]}: {t[:70]!r}")
            continue
        words = len(t.split())
        ok = (
            (bool(_GREET.search(t)) and words <= 6)
            or (bool(_RECEIPT.search(t)) and bool(_OBJECT.search(t)))
            or ((bool(_ASK_VERB.search(t)) or "?" in t) and bool(_DETAIL.search(t)))
            or (bool(_PERSON.search(t)) and bool(_REVIEW.search(t)) and bool(_OBJECT.search(t)))
            or _negated_promise(t)
        )
        if not ok:
            bad.append(f"sentence of no allowed type: {t[:70]!r}")
    return bad


def _unnegated(sentence: str, rx: re.Pattern) -> list[str]:
    """Matches of `rx` that are not negated in their own clause (same scope rules as the promise check)."""
    out: list[str] = []
    s = REASSURE.sub(" ", sentence)  # "не волнуйтесь, деньги вернут" is reassurance, not a negation of what follows
    for clause in CLAUSE_SPLIT.split(s):
        for m in rx.finditer(clause):
            before, after = clause[: m.start()], clause[m.end():]
            if NEG_BEFORE_WIDE.search(before) or NEG_AFTER.search(after):
                continue
            out.append(m.group(0))
    return out


def check_free_text(text: str, allowed_numbers: set[str] | None = None) -> list[str]:
    """Structural checks for text written by the model (see the block comment above). Returns the problems found."""
    problems: list[str] = []
    allowed = allowed_numbers or set()
    for s in _sentences(text):
        hits = _unnegated(s, OUTCOME_TOPIC)  # also in questions: a question mark does not make a statement safe
        if hits:
            problems.append(f"outcome topic mentioned without negation ({hits[0]!r}): {s[:70]!r}")
        if TIME_WORDS.search(s):
            problems.append(f"deadline or time expression: {s[:70]!r}")
        if PROCEDURE_WORDS.search(s):
            problems.append(f"procedure or instruction not present in the facts: {s[:70]!r}")
    for m in re.finditer(r"\d+", text):
        if m.group(0) not in allowed:
            problems.append(f"number not present in the ticket: {m.group(0)}")
            break
    return problems


def _is_claim_note_ref(text: str, ref: str) -> bool:
    """Numbers the PLAYER typed are echoed inside the claim_note sentence ('(4.4)') - not a citation."""
    return bool(re.search(rf"\([^)]*{re.escape(ref)}[^)]*\)", text))
