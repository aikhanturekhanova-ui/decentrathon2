"""Triage: category, priority, safety flags, structured extraction.

The dataset has no category/priority labels, texts are 12-30 characters long on median and full of slang ("оск родни", "дб", "зз"),
and the taxonomy is a draft. So the hard constraints (safety flags, mandatory handoff, priority raises, `complaint.invalid`, report
categories) are a transparent hand-set lexicon. The COMPLAINT category is a hybrid when a model built by `make train` is present:
the lexicon scores are blended with a TF-IDF + logistic-regression model trained on weak labels from administrators' replies
(app/ml.py; measured in docs/metrics/ml_classifier.json). Without the model the lexicon decides alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schemas import TriageRequest
from .textutil import clean_for_matching, normalize_game_chars
from . import toxicity as tx  # the type names (tx.INSULT …) must not be shadowed by this module's INSULT/PROFANITY regexes
from .toxicity import ToxicityReport, analyze_messages

# --------------------------------------------------------------------------------------------
# helpers


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.I | re.UNICODE)


def _tok(*abbr: str) -> str:
    """Whole-token match for short slang such as 'дм' or 'зз'."""
    return r"(?<![\w])(?:" + "|".join(abbr) + r")(?![\w])"


# --------------------------------------------------------------------------------------------
# lexicon: complaints

INSULT = _rx(
    r"(?<![\w])оск\w*", r"(?<![\w])osk\w*", r"(?<![\w])ock(?![\w])", r"(?<![\w])oskr\w*", r"оскорб\w*", r"оскарб\w*", r"[оа]ск[оа]?р[бп]\w*",
    r"(?<![\w])мат(?:ы|ами|ом|ерится|юк\w*|ерн\w*|ерит\w*)(?![\w])", r"нацис\w*", r"националь\w*",
    r"упом\w*\s*род\w*", r"(?<![\w])род(?:ни|них|ных|ные|ных|ственник\w*|ител\w*)(?![\w])", r"(?<![\w])rodni\w*",
    r"(?<![\w])(?:мам[ауыие]|мамк\w*|мать|матер\w*|матери|семь[юиея])(?![\w])", r"(?<![\w])мат(?![\w])",
    r"нецензур\w*", r"ненорм\w*", r"обзыв\w*", r"унижа\w*", r"харас\w*", r"токсич\w*", r"расиз\w*", r"расист\w*",
    r"нацизм\w*", r"чурк\w*", r"хач(?![\w])", r"религ\w*", r"розн[ьи]",
    r"haqorat\w*", r"xaqorat\w*", r"(?<![\w])sok(?:di|indi|inyabdi|yabdi|ib|ish\w*|mo\w*)", r"so['`ʻ]?kdi",
    r"(?<![\w])ona(?:dan|si|mni|ni)?(?![\w])", r"қорла\w*", r"сөк(?:ті|ті|т\w*|ү\w*)", r"insult\w*", r"swear\w*",
    r"racis\w*", r"toxic\w*", r"ofend\w*", r"insult[oó]\w*", r"xingou\w*", r"ofendeu\w*",
)
INSULT_CLAIM = _rx(r"(?<![\w])2\s*[.,]\s*[146](?![\d])")
# profanity / personal abuse, used as an additional (weaker) insult signal
PROFANITY = _rx(
    r"ху[йеёяи]\w*", r"пизд\w*", r"[её]б[ауо]\w*", r"[её]бл\w*", r"сук[аиу](?![\w])", r"мраз\w*", r"(?<![\w])чмо(?![\w])", r"даун\w*",
    r"дебил\w*", r"идиот\w*", r"пидор\w*", r"пидр\w*", r"уеб\w*", r"гандон\w*", r"шлюх\w*", r"тварь", r"мудак\w*", r"мудил\w*",
    r"долбо[её]б\w*", r"ублюд\w*", r"уродин\w*", r"выродок", r"лох(?![\w])", r"нахуй", r"блять", r"бля(?![\w])", r"fuck\w*", r"bitch\w*",
    r"puta(?![\w])", r"mierda", r"gandon\w*", r"jalab\w*", r"сикт\w*", r"қотақ\w*", r"амын\w*",
)
INSULT_UZKK = _rx(r"s[uo]['`ʻ‘’]?k(?:di|ti|ib|inyabdi|yabdi|ish\w*|inish\w*|ingan|ayotg\w*|uvdi|mo\w*|yapti|ib)", r"tahdid\w*", r"тахдид\w*",
                  r"ona(?:m|ng|si|sini)\w*", r"nomus\w*", r"сөг\w*", r"сөк\w*", r"қорла\w*", r"сок(?:ти|ди|ин\w*)", r"шешеден", r"анадан",
                  r"сўк\w*", r"ҳақорат\w*", r"онам\w*")  # Uzbek written in Cyrillic
THREAT = _rx(
    r"угрож\w*", r"угроз\w*", r"доксинг\w*", r"деанон\w*", r"найд[уё]\s+(?:тебя|вас)", r"твой\s+адрес", r"swat",
    r"(?<![\w])сват(?![\w])", r"ддос\w*", r"ddos", r"убь[юё]\s+(?:тебя|вас)", r"приеду\s+к\s+тебе",
    r"qo['`ʻ]?rqit\w*", r"threat\w*", r"найд[уёе]\w*\s+(?:тебя|вас|меня|нас|мой|мою|твой|ваш)\w*", r"прид[ёе]т\s+(?:ко\s+мне|домой)", r"приед[еу]\w*\s+(?:ко\s+мне|к\s+тебе|домой)",
)
# A threat counts as OUTSIDE the game with an explicit real-life marker; "найду тебя / приеду к тебе / твой дом" count only when nothing says
# that the place is in the game ("найду тебя в игре возле дома" is a role-play threat at an in-game house).
THREAT_OUTSIDE_STRONG = _rx(
    r"в\s+реальн\w*\s+жизни", r"в\s+реал(?:е|ьности)", r"в\s+жизни", r"твой\s+адрес", r"(?:мой|наш|мою|нашу|твой|ваш)\s+(?:реальн\w+\s+)?адрес", r"где\s+(?:я|ты)\s+жив\w*",
    r"swat", r"(?<![\w])сват(?![\w])", r"ддос\w*", r"ddos", r"доксинг\w*", r"деанон\w*", r"(?<![\w])irl(?![\w])", r"real\s+life", r"vida\s+real",
    r"школ\w*", r"(?:в|по)\s+(?:вк|вконтакте|инст\w*|телеграм\w*)", r"номер\s+телефон\w*", r"haqiqiy\s+hayot\w*", r"нақты\s+өмір\w*",
)
THREAT_OUTSIDE_WEAK = _rx(
    r"домой\s+к", r"(?:мой|твой|ваш|наш)\s+(?:дом|квартир\w*)", r"приед\w+\s+(?:к\s+(?:тебе|вам|нам|мне)|домой)", r"(?<![\w])приехать",
    r"найти\s+(?:меня|мой|мою|нас)", r"найд[уёе]\w*\s+(?:тебя|вас|меня|нас)", r"семью\s+найд",
)
IN_GAME_PLACE = _rx(r"в\s+игр\w*", r"на\s+сервер\w*", r"по\s+рп", r"(?<![\w])(?:ic|ик|rp|рп)(?![\w])", r"в\s+(?:гта|самп|crmp|grand\s*mobile)", r"in\s+(?:the\s+)?game", r"o['`]?yinda", r"ойында")
THREAT_OUTSIDE = re.compile(THREAT_OUTSIDE_STRONG.pattern + "|" + THREAT_OUTSIDE_WEAK.pattern, re.I | re.UNICODE)  # (kept for the audit scripts)


def threat_outside(clean: str) -> bool:
    return bool(THREAT_OUTSIDE_STRONG.search(clean) or (THREAT_OUTSIDE_WEAK.search(clean) and not IN_GAME_PLACE.search(clean)))

DM = _rx(
    _tok("дм", "dm", "дэм", "дб", "db", "тк", "tk", "ск", "sk", "дм+дб", "dmdb", r"drive\s*by", "massdm", r"mass\s*sk", "мг", "пг", "рк", "rk"),
    r"убил\w*", r"убий\w*", r"убива\w*", r"убит\w*", r"расстрел\w*", r"застрел\w*", r"стрел[яь]\w*", r"стрельб\w*",
    r"урон\w*", r"сбил\w*", r"сбив\w*", r"сбива\w*", r"задавил\w*", r"переех\w*", r"наезд\w*", r"kill\w*", r"ubil\w*", r"o['`ʻ]ldir\w*",
    r"bosdi", r"urib\s*ket\w*", r"мёртв\w*", r"смерт\w*",
    _tok("зз", "гз", "gz", "zz"), r"зелен\w*\s*зон\w*", r"green\s*zone",
    r"верев\w*", r"связал\w*", r"связыва\w*", r"похит\w*", r"мешок", r"мешк\w*", r"грабеж\w*", r"ограбил\w*",
    r"уход\w*\s+от\s+(?:рп|rp|смерт\w*|ситуац\w*|перестрел\w*)", r"ливнул\w*", r"вышел\s+из\s+игр\w*", _tok("афк", "afk"),
    _tok(r"нон\s*рп", "nonrp", "нрп", "nrp", r"non\s*rp", r"nrp\s*kop", "нонрп"), r"метагейм\w*", r"фейл\s*рп", r"fail\s*rp",
    r"рп\s+(?:процесс|ситуац)\w*",
)
DM_EXTRA = _rx(
    r"удар\w*", r"збил\w*", r"избил\w*", r"толкн\w*", r"дмит\w*", r"дмить", r"дмят", r"(?<![\w])o['`ʻ‘’]ld[ir]\w*", r"(?<![\w])[ou]ldir\w*", r"(?<![\w])uldrd\w*",
    r"(?<![\w])otib\w*", r"(?<![\w])otdi(?![\w])", r"(?<![\w])otdirdi", r"(?<![\w])urdi", r"(?<![\w])urib", r"qurol\w*", r"(?<![\w])bosdi", r"(?<![\w])bosib", r"(?<![\w])ezib\w*", r"kishan\w*", r"arqon\w*",
    r"(?<![\w])qop(?:ga)?(?![\w])", r"bez\s*rp", r"bezrp", r"besabab", r"sababsiz", r"өлтір\w*", r"ұрды", r"атып\w*",
    r"ўлдир\w*", r"сабабсиз", r"сабабсиз",
)
DM_CLAIM = _rx(r"(?<![\w])5\s*[.,]\s*[34](?![\d])", r"(?<![\w])1\s*[.,]\s*[12]\s*(?:гз|зз|gz)", r"(?<![\w])(?:гз|gz)(?![\w])")

CHEAT = _rx(
    r"софт\w*", r"(?<![\w])soft(?![\w])", r"(?<![\w])чит(?:ы|ер\w*|ит\w*|ов\w*|ах|ом|ам)?(?![\w])", r"cheat\w*", r"аим\w*", r"(?<![\w])aim\w*",
    _tok("wh", "вх", "wallhack", "валлхак", "speedhack", "спидхак", "флай", "fly", "аимбот", "aimbot", "бот"),
    r"speed\s*hack", r"летает", r"летит\s", r"телепортиру\w*", r"бессмерт\w*", r"god\s*mode", r"годмод\w*",
    r"макрос\w*", r"скрипт\w*", r"автокликер\w*", r"стороннее\s+по", r"хак(?:и|ер\w*)?(?![\w])", r"hack(?:er|s|ing)?(?![\w])",
    r"fraps?\s+.*софт", r"пролетает", r"через\s+стен\w*", r"сквозь\s+стен\w*",
)
CHEAT_CLAIM = _rx(r"(?<![\w])5\s*[.,]\s*2(?![\d])")

SCAM = _rx(
    r"ска+м\w*", r"sk[ao]+m\w*", r"scam\w*", r"обман\w*", r"кинул\w*", r"кидает", r"кидал\w*", r"развел\w*", r"развод\w*", r"мошенн\w*",
    r"вымог\w*", r"не\s*отда\w*", r"не\s*вернул\w*", r"взял\s+(?:деньги|машин\w*|дом|бизнес)", r"слил\s+деньги",
    r"aldadi", r"aldab\w*", r"aldov\w*", r"алдад\w*", r"алдап\w*", r"fraud\w*", r"estaf\w*", r"golpe",
)
SCAM_CLAIM = _rx(r"(?<![\w])4\s*[.,]\s*[12](?![\d])")

RMT = _rx(
    r"реклам\w*", r"пиар\w*", r"(?<![\w])piar\w*", r"продаю(?![\w])", r"продаёт", r"продает", r"продажа\s+(?:акк|аккаунт|вирт|имущ)\w*",
    r"за\s+реал\w*", r"за\s+реальн\w*", r"за\s+(?:руб\w*|тенге|сум|доллар\w*)", r"тг\s*канал\w*", r"телеграм\w*", r"telegram",
    r"t\.me", r"дискорд\w*", r"другой\s+сервер", r"сайт\w*", r"sotadi", r"sotyapti", r"sotmoqchi", r"сатады", r"сатып\s+жүр",
    r"selling", r"\bads?\b", r"advert\w*", r"vender\w*", r"venda",
)
RMT_CLAIM = _rx(r"(?<![\w])2\s*[.,]\s*3(?![\d])", r"(?<![\w])4\s*[.,]\s*4(?![\d])")

# a punishment word together with a defence ("бан без причины", "мут не за что") is an appeal
DEFENCE = _rx(r"без\s+причин\w*", r"не\s+за\s+что", r"просто\s+так", r"ошибочн\w*", r"несправедлив\w*", r"незаслуж\w*", r"не\s+наруш\w*", r"неправильн\w*",
              r"nohaq", r"bekorga", r"sababsiz", r"be\s*sabab", r"unfair\w*", r"injust\w*", r"sin\s+motivo", r"sem\s+motivo", r"себепсіз", r"әділетсіз")
INSULT_TYPO = _rx(r"[оа]ск[аоэ]?[бр]\w*", r"(?<![\w])аск\w*")
ADMIN_WORD = _rx(r"админ\w*", r"admin\w*", r"модер\w*", r"(?<![\w])адм(?![\w])")
# first-person denial / demand for proof: the author is the one who was punished
DENIAL = _rx(
    r"не\s+за\s+что", r"не\s+наруш\w*", r"я\s+не\s+(?:дм|дб|ск|тк|читер\w*|оскорб\w*|использ\w*|делал\w*|убива\w*|наруш\w*)",
    r"в\s+каком\s+месте\s+я", r"где\s+(?:ты\s+|вы\s+)?(?:увидел\w*|видите|вы\s+видели)", r"дали\s+(?:бан|мут|варн|деморган|дмг)",
    r"(?:выдали|выдал\w*|получил\w*)\s+(?:бан|мут|варн|деморган|дмг)\s+(?:не\s+за\s+что|просто\s+так|без\s+причины)",
    r"посадил\w*\s+(?:не\s+за\s+что|просто\s+так|без\s+причины)", r"прошу\s+(?:снять|пересмотр\w*|разобраться\s+с\s+наказани\w*)",
    r"требую\s+(?:доказательств\w*|снять|пересмотр\w*|фрапс\w*)", r"докажите", r"предоставьте\s+(?:видео|доказательств\w*|фрапс\w*|ссылк\w*)",
    r"доки\s+за", r"по\s+какому\s+праву", r"за\s+что\s+(?:мне|меня)", r"наказани\w*\s+(?:не\s+)?(?:по\s+делу|справедлив\w*)",
    r"снять\s+наказани\w*", r"сними\w*\s+(?:наказани\w*|бан|мут|варн)", r"оспар\w*|оспор\w*", r"обжалов\w*|обжал\w*",
    r"неправильн\w*\s+(?:выдач\w*|наказани\w*)", r"не\s+справедлив\w*",
    # defence of oneself against a punishment ("хотя не оскорблял", "это не чит", "где ты видишь")
    r"хотя\s+(?:я\s+)?не\s+\w+", r"я\s+(?:его\s+|ее\s+|её\s+)?не\s+(?:оскорбл\w*|оскарбл\w*|убива\w*|использ\w*)",
    r"это\s+не\s+(?:чит\w*|софт\w*|дм\w*|оскорбл\w*)", r"не\s+(?:чит|софт)(?![\w])", r"я\s+(?:даже\s+)?не\s+(?:был|заходил|делал|писал)\w*", r"где\s+(?:ты\s+|вы\s+)?вид\w*",
    r"в\s+каком\s+месте", r"(?:дал\w*|даёт|дают|выдал\w*|выдают|выдан\w*|получил\w*)\s+(?:бан|мут|варн|банк|деморган|дмг|джейл)",
    r"(?:бан|мут|варн|деморган)\s+за\s+\w+", r"за\s+что\s+(?:мне\s+)?(?:бан|мут|варн|посадил\w*|кафнул\w*)",
    r"посадил\w*", r"кафнул\w*", r"кафнули", r"арестов\w*\s+(?:не\s+)?за\s+что",
    r"emasman", r"bandan\s+yeching", r"(?:ban|mut|mute|jail|warn)\s+(?:ber(?:di|ishdi|ildi|gan|ibdi)|oldim|yedim)", r"qamal\w*",
    r"qamoq\w*", r"kishanlad\w*", r"bekorga", r"be\s*sabab\s+(?:qamal|ban|mut|jail)\w*", r"nima\s+sabab\w*", r"nega\s+(?:ban|mut|jail|qamoq)\w*",
    r"мен\s+(?:бұзбадым|айтпадым|жасамадым)", r"бан\s+берді", r"неге\s+бан",
)
ADMIN = _rx(
    r"не\s*согла\w*", r"несогл\w*", r"не\s*согл(?![\w])", r"(?<![\w])опру(?![\w])", r"оспор\w*", r"оспар\w*",
    r"обжал\w*", r"апелл\w*", r"appeal\w*", r"снять\s+(?:бан|мут|варн|наказ)\w*", r"сними\w*", r"разбан\w*", r"unban\w*",
    r"за\s+что\s+(?:бан|мут|варн|наказ)\w*", r"выдал\w*\s+(?:бан|мут|варн|наказ)\w*", r"незаслуж\w*", r"несправедлив\w*",
    r"пересмотр\w*", r"перевыда\w*", r"наказани\w*\s+(?:выдан\w*)?\s*(?:неверн|ошибочн)\w*", r"ошибочно\s+(?:забан|наказ)\w*",
    r"забанил\w*", r"заблокир\w*", r"bloklab\w*", r"ban\s*ber(?:di|ishdi|ildi|gan|ibdi)", r"варн\s+ber(?:di|ishdi|ildi)",
)
PUNISHMENT = _rx(
    r"(?<![\w])(?:ванн?|бан+|ban+|мут|mut|mute|варн|warn|кик|kick|дмг|деморган\w*|demorgan|jail|blok)(?![\w])", r"заблокир\w*", r"bloklad\w*",
    r"посадил\w*", r"посадили", r"nohaq", r"заблок\w*", r"выдал\w*\s+(?:наказ|пред)\w*", r"наказал\w*", r"наказани\w*",
)
ADMIN_CLAIM = _rx(r"(?<![\w])3\s*[.,]\s*[24](?![\d])")
APPEAL = _rx(
    r"не\s*согла\w*", r"несогл\w*", r"(?<![\w])опру(?![\w])", r"оспор\w*", r"оспар\w*", r"обжал\w*", r"апелл\w*", r"appeal\w*",
    r"снять\s+(?:бан|мут|варн|наказ)\w*", r"сними\w*", r"разбан\w*", r"unban\w*", r"пересмотр\w*", r"перевыда\w*",
    r"за\s+что\s+(?:бан|мут|варн|наказ)\w*", r"незаслуж\w*", r"несправедлив\w*", r"ошибочно\s+(?:забан|наказ)\w*",
    r"меня\s+(?:забанил\w*|заблокир\w*)", r"мне\s+(?:выдал\w*|дали)\s+(?:бан|мут|варн)\w*", r"ban\s*ber(?:di|ishdi|ildi|gan|ibdi)",
)

OTHER = _rx(
    r"спам\w*", r"18\s*\+", r"порн\w*", r"наркот\w*", r"музык\w*", r"микрофон\w*", r"флуд\w*", r"фарм\s*репорт\w*",
    r"мешает", r"ивент\w*", r"ник(?:нейм)?(?![\w])", r"никнейм\w*", r"слив(?:\s|$)\w*", r"багоюз\w*", r"абуз\w*",
    r"использует\s+баг\w*", r"пропаганд\w*", r"стрим\w*", r"номерн\w*\s+знак\w*",
)
OTHER_CLAIM = _rx(r"(?<![\w])2\s*[.,]\s*[2-5](?![\d])", r"(?<![\w])1\s*[.,]\s*6(?![\d])", r"(?<![\w])5\s*[.,]\s*[5-9](?![\d])")

# Signals that force a human no matter which category wins (kind-independent, evidence-independent).
PAYMENT_STRONG = _rx(
    # "оплатить штрафы/налоги/аренду" are in-game payments, not a purchase for real money
    r"донат\w*", r"(?<![\w])оплат\w*(?![\w])(?!\s+(?:\w+\s+){0,2}?(?:штраф\w*|налог\w*|аренд\w*|коммунал\w*|кредит\w*|ипотек\w*|залог\w*|выкуп\w*|айыппұл\w*|jarima\w*))", r"платеж\w*", r"платёж\w*", r"списа(?:л\w*|ли|но|ны)\s+(?:\w+\s+){0,3}?(?:деньг\w*|донат\w*|коин\w*|средств\w*|баланс\w*|карт\w*|сч[её]т\w*|руб\w*)", r"refund\w*", r"верн\w*\s+деньги",
    r"вернуть\s+деньги", r"возврат\s+(?:денег|средств)", r"to['`ʻ]?lov\w*", r"grand\s*coin\w*", r"гранд\s*коин\w*",
    r"купил\w*\s+(?:вирт|коин|донат|вип)\w*", r"google\s*play", r"app\s*store",
    r"donaci[oó]n\w*", r"doa[cç][aã]o", r"monedas?(?![\w])", r"moedas?(?![\w])", r"pagu[eé]\w*", r"to['`ʻ‘’]?la(?:dim|b|gan)\w*", r"tanga\w*", r"төле\w*",
    r"pagado", r"i\s+paid", r"reembols\w*", r"reembolso", r"pagamento", r"payment",
    r"тўлов\w*", r"тўлад\w*", r"(?<![\w])donat\w*", r"xarid\w*", r"sotib\s+ol\w*", r"cobr(?:ad[oa]s?|aram|aron|ou|[oó])(?![\w])", r"dinheiro\s+de\s+volta", r"dinero\s+de\s+vuelta", r"devolv\w*\s+(?:meu\s+)?dinheiro", r"devuelv\w*\s+(?:mi\s+)?dinero", r"money\s+back", r"(?:charged|double\s+charge)\w*\s*(?:me\s+)?(?:twice|double)?", r"купил\w*\s+(?:в\s+магазин\w*|подписк\w*|пакет\w*)", r"покупк\w*", r"поступлен\w*\s+средств", r"деньги\s+за\s+покупк\w*",
    r"төлем\w*", r"төледім", r"сатып\s+алдым", r"to['`ʻ‘’]?lov\w*", r"sotib\s+oldim", r"purchase\w*", r"charged\s+me", r"compra\w*",
)
_PAY_MONEY = (r"(?:руб\w*|тенге|теңге|(?<![\w])тг(?=\s*(?:за|с\s+карт|списал|сняли))|доллар\w*|usd|eur|пакет\w*\s+(?:за|доната|коинов)|реальн\w+\s+деньг\w*|донат\w*|коин\w*|платеж\w*|платёж\w*|оплат\w*|"
              r"банковск\w*\s+карт\w*|(?<![\w])с\s+карты|подписк\w*|төлем\w*|to['`ʻ’]?lov\w*|тўлов\w*|real\s+money|dinero\s+real|dinheiro\s+real|haqiqiy\s+pul\w*|нақты\s+ақша\w*|"
              r"coins?|monedas?|moedas?|donat\w*|donaci[oó]n\w*|doa[cç][aã]o|so['`ʻ‘’]?m(?![\w])|сум(?![\w]))")  # bare "деньги" is NOT here: in-game money is not a payment
_PAY_VERB = (r"(?:верн\w*|возврат\w*|возмест\w*|возмещ\w*|списал\w*|списан\w*|сняли|снимают|заплатил\w*|оплатил\w*|купил\w*|"
             r"не\s*пришл\w*|не\s*зачисл\w*|не\s*поступ\w*|refund\w*|paid|bought|charged|reembols\w*|pagu[eé]\w*|paguei|comprei|compr[eé]\w*|devolv\w*|devuelv\w*|"
             r"qaytar\w*|to['`ʻ‘’]?lad\w*|sotib\s+old\w*|қайтар\w*|төле\w*|сатып\s+ал\w*|yechil\w*)")
PAY_INCIDENT = re.compile(
    _PAY_VERB + r"\W+(?:\w+\W+){0,6}?" + _PAY_MONEY + "|" + _PAY_MONEY + r"\W+(?:\w+\W+){0,6}?" + _PAY_VERB, re.I | re.UNICODE)
REFUND_STRONG = _rx(r"refund\w*", r"reembols\w*", r"возврат\w*\s+(?:денег|средств|доната|платеж\w*|покупк\w*)", r"верн\w+\s+(?:мне\s+)?(?:мой\s+)?донат\w*",
                    r"верн\w+\s+(?:мне\s+)?(?:реальн\w+\s+деньг\w*|деньги\s+за\s+(?:донат|покупк|подписк)\w*)", r"money\s+back", r"dinheiro\s+de\s+volta", r"dinero\s+de\s+vuelta")
# the same real-money purchase charged twice: "дважды сняли 500 тенге за один пакет", "екі рет ақша алынды", "ikki marta pul yechildi"
DOUBLE_CHARGE = _rx(
    r"(?:дважды|два\s+раза|двойн\w*|повторно|ещ[её]\s+раз)\s+(?:\w+\s+){0,2}?(?:сняли|списали|списал\w*|снял\w*|списан\w*|оплат\w*|взяли)",
    r"(?:сняли|списали|списан\w*)\s+(?:\w+\s+){0,4}?(?:дважды|два\s+раза|повторно)", r"двойн\w+\s+списани\w*",
    r"(?:charged|billed)\s+(?:me\s+)?twice", r"double\s+(?:charge|payment|billing)", r"cobr(?:aron|ado|ou)\s+(?:\w+\s+){0,2}?(?:dos|duas)\s+veces|cobr\w*\s+(?:\w+\s+){0,2}?duas\s+vezes",
    r"екі\s+рет\s+(?:\w+\s+){0,2}?(?:алынды|шешілді|төледім|төлендім|ақша)", r"ikki\s+marta\s+(?:\w+\s+){0,2}?(?:yechil\w*|to['`]?la\w*|pul|olindi)",
)
REAL_MONEY_CHANNEL = _rx(r"google\s*play", r"app\s*store", r"apple\s*id", r"банковск\w*\s+карт\w*", r"(?<![\w])с\s+карты", r"реальн\w+\s+деньг\w*\s+(?:списал|потерял|вернут)")
MONEY_SYMBOL = re.compile(r"\d\s*(?:₽|₸|\$|€)|(?:₽|₸|\$|€)\s*\d")
# "я не прошу вернуть деньги, просто скажите ..." / "not asking for a refund": the topic is mentioned in order to be rejected
# the player says the problem is solved by himself ("я поменял пароль и теперь всё работает"): not a running incident
RESOLVED = _rx(r"(?:теперь|уже)\s+(?:вс[её]\s+)?(?:работает|нормально|в\s+порядке|ок)", r"проблем\w*\s+(?:решен\w*|решил\w*|ушл\w*)", r"(?:сам|сама)\s+(?:решил\w*|разобрал\w*|справил\w*)",
               r"вопрос\s+(?:снят|закрыт)", r"(?:works|fine)\s+now", r"(?:problem|issue)\s+(?:is\s+)?(?:solved|fixed|resolved)", r"endi\s+ishlayapti", r"енді\s+жұмыс\s+істеп\s+тұр")
NEG_PAY_REQUEST = _rx(
    r"(?:не|без)\s+(?:прошу|требую|хочу|надо|нужн\w+|просил\w*|собираюсь)\s+(?:\w+\s+){0,4}?(?:возврат\w*|верн\w+|деньг\w*|компенсац\w*|донат\w*|оплат\w*)",
    r"(?:not|n['`]t|no)\s+(?:asking|requesting|looking|want\w*|need\w*)\s+(?:\w+\s+){0,3}?(?:refund\w*|money|compensat\w*)",
    r"no\s+(?:pido|quiero|necesito)\s+(?:\w+\s+){0,3}?(?:reembols\w*|dinero|devoluci\w*)", r"n[aã]o\s+(?:pe[cç]o|quero|preciso)\s+(?:\w+\s+){0,3}?(?:reembols\w*|dinheiro|devolu\w*)",
)
# a question ABOUT a rule ("можно ли оскорблять в чате?") is not a complaint about a player
RULE_QUESTION = _rx(
    r"(?:^|\s)можно\s+ли(?![\w])", r"(?:^|\s)можно(?![\w])", r"(?:^|\s)разрешено(?:\s+ли)?(?![\w])", r"(?:^|\s)запрещено(?:\s+ли)?(?![\w])",
    r"(?:^|\s)допустимо(?![\w])", r"(?:^|\s)нельзя\s+ли(?![\w])", r"что\s+будет\s+(?:если|за)", r"за\s+что\s+(?:дают|банят|наказывают)",
    r"какое\s+наказание", r"какие\s+правила", r"is\s+it\s+allowed", r"(?:^|\s)can\s+i(?![\w])", r"am\s+i\s+allowed", r"what\s+happens\s+if",
    r"mumkinmi", r"ruxsat\w*", r"taqiqlan\w*", r"мүмкін\s+бе", r"рұқсат",
    r"(?:^|\s)se\s+puede(?![\w])", r"est[aá]\s+permitido", r"(?:^|\s)puedo(?![\w])", r"est[aá]\s+prohibido", r"qu[eé]\s+pasa\s+si",
    r"[eé]\s+permitido", r"(?:^|\s)posso(?![\w])", r"(?:^|\s)se\s+pode(?![\w])", r"[eé]\s+proibido", r"o\s+que\s+acontece\s+se",
    r"bo['`ʻ‘’]?ladimi", r"qilsa\s+bo['`ʻ‘’]?ladi", r"болады\s+ма", r"бола\s+ма", r"рұқсат\s+па", r"тыйым\s+салынған\s+ба",
)
# topic -> rule point that can answer a question about it (order matters: first match wins)
QUESTION_TOPICS: list[tuple[str, re.Pattern]] = [
    ("2.4", _rx(r"расиз\w*", r"нацизм\w*", r"религ\w*", r"национальн\w*", r"розн[ьи]")),
    ("2.1", _rx(r"оскорб\w*", r"(?<![\w])оск\w*", r"ругат\w*", r"обзыв\w*", r"материть\w*", r"(?<![\w])мат(?![\w])", r"нецензур\w*", r"унижа\w*", r"insult\w*", r"haqorat\w*", r"қорла\w*")),
    ("5.2", _rx(r"(?<![\w])чит(?:ы|ер\w*|ить|ах|ами)?(?![\w])", r"софт\w*", r"макрос\w*", r"скрипт\w*", r"(?<![\w])бот(?:ы|ов|а)?(?![\w])", r"автокликер\w*", r"cheat\w*", r"aimbot", r"аимбот")),
    ("4.4", _rx(r"за\s+реал\w*", r"реальн\w+\s+деньг\w*", r"за\s+(?:руб\w*|тенге|сум|доллар\w*)", r"продать\s+(?:акк|аккаунт|вирт|донат)\w*", r"купить\s+(?:акк|аккаунт)\w*", r"продажа\s+аккаунт\w*", r"real\s+money")),
    ("4.1", _rx(r"передать\s+(?:акк|аккаунт)\w*", r"обман\w*", r"мошенн\w*", r"скам\w*", r"махинац\w*", r"scam\w*")),
    ("2.3", _rx(r"реклам\w*", r"пиар\w*", r"advert\w*", r"promot\w*")),
    ("GZ 1.1", _rx(_tok("зз", "гз"), r"зелен\w*\s+зон\w*", r"green\s*zone")),
    ("1.6", _rx(r"(?<![\w])ник(?:нейм\w*)?(?![\w])", r"nickname")),
    ("2.5", _rx(r"музык\w*", r"микрофон\w*", r"крич\w*", r"music", r"microphone")),
    ("3.1", _rx(r"репорт\w*", r"флуд\w*")),
    ("ЖБ 1.3.1", _rx(r"подать\s+жалоб\w*", r"как\s+написать\s+жалоб\w*", r"что\s+нужно\s+для\s+жалоб\w*")),
    ("3.4", _rx(r"жалоб\w+\s+на\s+админ\w*", r"оспорить\s+(?:действ|наказани)\w*")),
]
_Q_EXTRA = {
    "2.4": [r"racis\w*", r"raz[oó]n\s+racial", r"ксенофоб\w*"],
    "2.1": [r"insultar\w*", r"insulto\w*", r"xingar", r"insulting", r"swear\w*", r"offend\w*", r"ofender\w*", r"haqorat\w*", r"сөг\w*"],
    "5.2": [r"trampas?(?![\w])", r"hacks?(?![\w])", r"trapa[cç]a\w*", r"cheats?(?![\w])", r"(?<![\w])chit\w*", r"(?<![\w])чит\w*"],
    "4.4": [r"dinero\s+real", r"dinheiro\s+real", r"haqiqiy\s+pul\w*", r"нақты\s+ақша\w*", r"sell\w*\s+(?:an?\s+)?account", r"vender\s+(?:una?\s+)?cuenta", r"vender\s+(?:uma?\s+)?conta"],
    "4.1": [r"estafa\w*", r"enga[ñn]ar", r"engan\w*", r"fraud\w*", r"алдау\w*", r"aldash\w*"],
    "2.3": [r"publicidad", r"propaganda", r"reklama\w*", r"жарнама\w*", r"advertis\w*"],
}
QUESTION_TOPICS = [
    (c, re.compile(rx.pattern + "|" + "|".join(_Q_EXTRA[c]), re.I | re.UNICODE) if c in _Q_EXTRA else rx) for c, rx in QUESTION_TOPICS
]
VICTIM_CUE = _rx(
    r"меня\s+(?:\w+\s+)?(?:оскорб|убив|убил|дмят|дм(?![\w])|обман|кинул|взлом|забан|забанил|заблок)", r"мне\s+(?:\w+\s+)?(?:оскорб|выдал|дали)",
    r"(?:он|она|они|игрок)\s+(?:\w+\s+)?(?:оскорб|убил|убива)",
)
_SELF_NICK = _rx(r"(?:мой|моего|мое|моё|мои|my|mi|meu|mening|менің)\s+ник(?:нейм\w*|а|у|е|ом)?", r"(?:my|mi|meu)\s+nick(?:name)?")  # «мой ник Dias_Volkov» is not a nickname complaint
NICKNAME_COMPLAINT =_rx(r"(?<![\w])ник(?:нейм\w*|а|у|е|ом)?(?![\w])", r"никнейм\w*", r"(?<![\w])nick(?:name)?(?![\w])")
# --- extra vocabulary for English / Spanish / Portuguese / Uzbek / Kazakh (the client is localized in 6 languages) ---
INSULT_ML = _rx(
    r"insult\w*", r"ofend\w*", r"xing\w*", r"grosero\w*", r"hijo\s+de", r"filho\s+da", r"idiota\w*", r"est[uú]pid\w*", r"burro(?![\w])",
    r"abus\w*", r"call(?:ed|s)\s+me", r"swore", r"cursed", r"bully\w*", r"insulto", r"faltou\s+com", r"falt[oó]\s+el\s+respeto",
)
DM_ML = _rx(
    r"mat[óo](?![\w])", r"matou", r"matar\w*", r"mataron", r"mataram", r"asesin\w*", r"dispar\w*", r"atropell\w*", r"atropel\w*",
    r"sin\s+(?:motivo|raz[oó]n)", r"sem\s+(?:motivo|raz[aã]o)", r"zona\s+verde", r"baleou", r"atirou", r"killed", r"shot\s+me",
    r"ran\s+me\s+over", r"for\s+no\s+reason", r"without\s+(?:a\s+)?reason", r"spawn\s*kill", r"team\s*kill", r"tied\s+me", r"kidnap\w*",
    r"secuestr\w*", r"sequestr\w*", r"amarr\w*", r"(?<![\w])atou(?![\w])", r"(?<![\w])at[óo](?![\w])",
)
CHEAT_ML = _rx(
    r"trampa\w*", r"tramposo\w*", r"vuela(?![\w])", r"volando", r"voa(?![\w])", r"voando", r"flying", r"flies", r"wall\s*hack",
    r"hack(?:s|er|ers|ing)?(?![\w])", r"cheater\w*", r"cheating", r"mod\s+menu", r"ұшады", r"ұшып", r"uchyapti", r"havoda", r"chit\w*", r"hackers?",
)
SCAM_ML = _rx(
    r"estaf\w*", r"enga[ñn]\w*", r"engan\w*", r"golpe(?![\w])", r"golpista", r"fraud\w*", r"no\s+devolv\w*", r"n[aã]o\s+devolv\w*",
    r"roub\w*", r"rob[oó](?![\w])", r"ladr[oó]\w*", r"cheated\s+me", r"stole", r"did\s+not\s+return", r"didn'?t\s+return",
    r"never\s+gave", r"took\s+my\s+(?:money|car|house)", r"aldad\w*", r"алдад\w*", r"алдады", r"қайтармады", r"qaytarmadi", r"aldadi",
    # Kazakh: «ақшамды алды да, көлікті бермеді» (took my money and did not give the car), «алаяқ» (fraudster)
    r"ақша\w*\s+алды", r"(?:ақша|көлік|машина|үй|бизнес)\w*.{0,30}бермеді", r"алаяқ\w*", r"алдап\s+кетті",
    # Uzbek: «pulimni oldi, mashinani bermadi» (took my money, did not give the car), «firibgar» (fraudster)
    r"pul\w*\s+(?:\w+\s+)?oldi", r"(?:pul|mashina|uy|biznes|akkaunt)\w*.{0,30}bermadi", r"firib\w*", r"фириб\w*", r"пул\w*\s+(?:\w+\s+)?олди",
)
RMT_ML = _rx(
    r"dinero\s+real", r"dinheiro\s+real", r"vend[eéi]\w*", r"venda", r"publicidad", r"propaganda", r"anunci\w*", r"real\s+money",
    r"sells?(?![\w])", r"advertis\w*", r"haqiqiy\s+pul\w*", r"sotyapti", r"sotadi", r"нақты\s+ақша\w*", r"сатады", r"сатып", r"жарнама\w*", r"reklama\w*",
)
APPEAL_ML = _rx(
    r"ban(?:eó|eado|iu|ido)", r"injust\w*", r"unfair\w*", r"no\s+estoy\s+de\s+acuerdo", r"n[aã]o\s+concordo", r"disagree\w*", r"apelaci[oó]n",
    r"apela[cç][aã]o", r"appeal\w*", r"unban\w*", r"desban\w*", r"(?:banned|muted)\s+me", r"nohaq", r"rozi\s+emas\w*", r"келіспе\w*", r"әділетсіз",
    r"me\s+(?:baneó|baniu|silenció|silenciou)",
    # Kazakh: «бан бекер берілді» (the ban was given for nothing), «қайта қараңыз» (review it again), «кінәсізбін» (I am innocent)
    r"бекер\s+(?:бан|мут|жаза|бер)\w*", r"(?:бан|мут|жаза)\w*\s+бекер", r"қайта\s+қара\w*", r"кінәсіз\w*", r"(?:бан|жаза)\w*\s+алып\s+таста\w*",
)
ADMIN_ML = APPEAL_ML
EVIDENCE_WORDS = _rx(
    r"скрин\w*", r"видео", r"фрапс\w*", r"fraps", r"доказ\w*", r"доки(?![\w])", r"запис[ьи]\w*", r"proof", r"screen\w*", r"ссылк\w*",
    r"link", r"dalil\w*", r"skrin\w*", r"clip", r"yutub\w*", r"youtube", r"imgur", r"<url>", r"прикреп\w*", r"приложил\w*",
    r"pruebas?", r"capturas?", r"provas?", r"prints?", r"isbot\w*", r"дәлел\w*", r"пруф\w*",
)
# Evidence policy (see docs/ARCHITECTURE.md): the wording of the text and the attachment count are separate facts.
#   attached        attachments > 0                      (content never inspected; the text may even say "no proof": reported as a conflict)
#   claimed_only    "скрины есть" but 0 attachments      (a claim, not evidence)
#   stated_absent   "доказательств нет"                  (explicit statement, taken at face value only when nothing is attached)
#   asked_about     "какие доказательства нужны?"        (a question, not a claim)
_EVW = (r"(?:доказательств\w*|доказ\w*|скрин\w*|видео|видос\w*|фрапс\w*|fraps|запис[ьи]\w*|док[иа]?(?![\w])|пруф\w*|proof|evidence|screenshots?|video|"
        r"recording|dalil\w*|isbot\w*|skrin\w*|pruebas?|capturas?|provas?|prints?|дәлел\w*|скриншот\w*)")
_GAP = r"\s+(?:[\w']+\s+){0,2}?"
NEG_EVIDENCE = re.compile(
    r"(?:нет|нету|отсутству\w*|не\s+имеется|не\s+осталось|не\s+будет|не\s+предоставлю|не\s+прилагаю|без|никаких|"
    r"не\s+(?:приложил\w*|прикрепил\w*|записал\w*|успел\w*|снимал\w*|сделал\w*|сохранил\w*)|"
    r"(?:no|without|don'?t\s+have|do\s+not\s+have|have\s+no|has\s+no|didn'?t\s+(?:record|take|save)|lack\s+of)|"
    r"(?:no\s+tengo|sin|n[aã]o\s+tenho|sem)|(?:yo['`]?q|yoq)|жоқ)" + _GAP + _EVW + "|" +
    _EVW + r"\s+(?:[\w']+\s+){0,2}?(?:нет|нету|отсутству\w*|не\s+имеется|не\s+осталось|не\s+будет|не\s+(?:приложен\w*|прикреплен\w*|записан\w*|сделан\w*|сохранил\w*)|"
    r"yo['`]?q|yoq|жоқ|missing|absent|not\s+available|n[aã]o\s+(?:tenho|h[aá]))", re.I | re.UNICODE)
ASK_EVIDENCE = re.compile(
    r"(?:какие|какой|какая|что\s+нужно|нужн\w+\s+ли|нужно\s+ли|надо\s+ли|можно\s+ли|как\s+(?:приложить|прикрепить|отправить|подать|сделать)|чем\s+подтвердить|"
    r"what|which|do\s+i\s+need|how\s+(?:do\s+i\s+)?(?:attach|send|upload)|qanday|qaysi|qu[eé]|que|qual)" + _GAP + _EVW, re.I | re.UNICODE)
# a short question about a game term: "что такое DM?", "dm nima", "DB это что"
TERM_ANY = _rx(
    _tok("дм", "dm", "дэм", "дб", "db", "ск", "sk", "тк", "tk", "рк", "rk", "мг", "mg", "пг", "pg", "нрп", "nrp", r"нон\s*рп", r"non\s*rp", "nonrp"),
    r"метагейм\w*", r"metagam\w*", r"павергейм\w*", r"powergam\w*", r"drive\s*by", r"драйв\s*бай", r"спавн\s*кил\w*", r"spawn\s*kill", r"team\s*kill", r"тим\s*кил\w*",
    r"death\s*match", r"дезматч", r"revenge\s*kill",
)

# complaints that are not complaints
JUNK_TOKENS = {
    ".", "..", "...", "жб", "жалоба", "доки", "доказательства", "чат", "вот он", "вот", "фрапс", "видео", "скрин", "скрины",
    "смотрите", "смотри", "тест", "test", "+", "-", "?", "!", "я", "он", "ты", "привет", "ау", "хз", "нет", "да", "ок", "ok",
}
QUESTION = _rx(
    r"^(?:как|где|что|когда|почему|зачем|сколько|можно\s+ли|какой|какая|какие|кто|куда|чем|нужно\s+ли|подскажите|скажите|можно)(?![\w])",
    r"^(?:how|where|what|when|why|can\s+i|who|which|is\s+there|do\s+you)(?![\w])",
    r"^(?:nega|qanday|qayerda|nima|qachon|kim|qancha|qaysi|bormi)(?![\w])",
    r"^(?:қалай|қайда|не\s+үшін|неге|қашан|қанша|кім)(?![\w])",
    r"^(?:cómo|como|dónde|donde|qué|que|cuándo|onde|quando)(?![\w])", r"\?",
)

# --------------------------------------------------------------------------------------------
# lexicon: reports

PAYMENT = _rx(
    r"донат\w*", r"(?<![\w])оплат\w*(?![\w])(?!\s+(?:\w+\s+){0,2}?(?:штраф\w*|налог\w*|аренд\w*|коммунал\w*|кредит\w*|ипотек\w*|залог\w*|выкуп\w*|айыппұл\w*|jarima\w*))", r"платеж\w*", r"платёж\w*", r"купил\w*\s+(?:вирт\w*|коин\w*|гк|gc|донат|валют\w*|пакет\w*|vip|вип)",
    _tok("гк", "gc"), r"grand\s*coin\w*", r"гранд\s*коин\w*", r"списа(?:л\w*|ли|ны|но)\s+(?:\w+\s+){0,3}?(?:деньг\w*|донат\w*|коин\w*|средств\w*|баланс\w*|карт\w*|руб\w*|тенге)",
    # "не пришёл донат" is a payment; "не пришла зарплата за работу таксистом" (in-game salary) is not
    r"(?:донат\w*|коин\w*|покупк\w*|пакет\w*|оплат\w*|платеж\w*)\s+(?:\w+\s+){0,2}?не\s*(?:пришл|зачисл|поступ)\w*", r"не\s*(?:пришл|зачисл|поступ)\w*\s+(?:\w+\s+){0,2}?(?:донат\w*|коин\w*|покупк\w*|пакет\w*)",
    r"возврат\w*", r"верн\w*\s+деньги", r"вернуть\s+деньги", r"refund\w*", r"payment", r"(?<![\w])paid(?![\w])", r"to['`ʻ]?lov\w*",
    r"pul\s+(?:yechil\w*|o['`ʻ]tmadi)", r"google\s*play", r"app\s*store", r"apple\s*id", r"банковск\w*", r"(?<![\w])чек(?![\w])",
    r"pagu[eé]\w*", r"compr[eé]\w*", r"cobro\w*", r"bought",
)
ACCOUNT = _rx(
    r"взлом(?:али|ал[аи]?|ан\w*|ают|ом|а|у|е)?(?![\w])", r"хакн\w*", r"hacked", r"hackea\w*", r"hackeado", r"украли\s+(?:аккаунт|акк|машин\w*|дом|деньги|имущ\w*)",
    r"угнал\w*\s+акк", r"аккаунт\w*", _tok("акк", "акка", "аккаунта"), r"пропал\w*\s+(?:машин\w*|дом|аккаунт|акк|имущ\w*|деньги|бизнес|вещи|оружи\w*)",
    r"исчез\w*", r"не\s*могу\s+(?:зайти|войти|залогин\w*)", r"забыл\w*\s+(?:пароль|логин)", r"пароль", r"восстанов\w*",
    r"потерял\w*\s+(?:акк|доступ)", r"доступ\w*\s+к\s+акк", r"кто[- ]то\s+зашел", r"зашли\s+(?:на|в)\s+мой", r"login", r"password",
    r"parol", r"akkaunt\w*", r"buzildi", r"o['`ʻ]g['`ʻ]irlandi", r"cuenta\s+(?:robada|hackeada)", r"conta\s+(?:roubada|hackeada)",
    r"perdi\s+minha\s+conta", r"perd[ií]\s+mi\s+cuenta",
)
HACK_WORDS = _rx(
    r"взлом(?:али|ал[аи]?|ан\w*|ают|щик\w*)(?![\w])", r"взлом\w*\s+(?:мо\w+\s+)?акк\w*", r"акк\w*\s+взлом\w*", r"хакн\w*", r"hacked", r"hackea\w*", r"hackearam", r"украли\s+(?:мой\s+)?акк\w*", r"угнал\w*\s+акк\w*", r"увели\s+акк\w*",
    r"акк\w*\s+(?:украден|угнан|взломан|слит)\w*", r"кто[- ]то\s+(?:зашел|заходил|зашёл)", r"зашли\s+(?:на|в)\s+мой", r"vzlom\w*",
    # Uzbek (Latin and Cyrillic), Kazakh: "аккаунтимни бузишди", "аккаунтымды бұзды", "akkauntimni buzishdi"
    r"akkaunt\w*\s+(?:\w+\s+){0,3}?buz\w*", r"buz\w*\s+(?:\w+\s+){0,2}?akkaunt\w*", r"hack\s*qil\w*", r"hackladi\w*",
    r"аккаунт\w*\s+(?:\w+\s+){0,3}?буз\w*", r"буз\w*\s+(?:\w+\s+){0,2}?аккаунт\w*", r"хак\s*қил\w*", r"хаклад\w*",
    r"аккаунт\w*\s+(?:\w+\s+){0,3}?бұз\w*", r"бұз\w*\s+(?:\w+\s+){0,3}?аккаунт\w*", r"ұрла(?:п|ды|ған)", r"хакер\w*\s+аккаунт\w*",
    r"o['`ʻ‘’]?g['`ʻ‘’]?irlab\s+ol\w*", r"akkaunt\w*\s+o['`ʻ‘’]?g['`ʻ‘’]?irla\w*",
    r"roubada", r"robada", r"roubaram\s+(?:minha\s+)?conta", r"conta\s+(?:foi\s+)?invadida", r"cuenta\s+(?:fue\s+)?(?:robada|hackeada)",
    r"(?:someone|somebody)\s+(?:hacked|stole|took\s+over)", r"account\s+(?:was\s+)?(?:stolen|hacked|compromised)",
)
_ACC = r"(?:акк\w*|аккаунт\w*|учетн\w*(?:\s+запис\w*)?|учетк\w*|профил\w*|аккаунт|account|profile|cuenta|conta|hisob\w*|akkaunt\w*|аккаунт\w*)"
_FOREIGN = r"(?:чуж\w*|посторонн\w*|неизвестн\w*|кто[- ]то|кто-либо|незнаком\w*|стороннее\s+лиц\w*|левы\w+\s+человек|какой[- ]то\s+\w+)"
_TAKE = r"(?:угнал\w*|украл\w*|взломал\w*|хакн\w*|увел\w*|захватил\w*|отобрал\w*|слил\w*|украден\w*|угнан\w*|захвачен\w*|взломан\w*)"
# done TO me (3rd person plural / passive), never "я поменял мой пароль" (my own change)
_CHANGE = r"(?:сменили|поменяли|изменили|заменили|подменили|привязали|отвязали|сбросили|отвязан\w*|перепривязал\w*)"
HACK_TAKEOVER = _rx(
    # "мою учётную запись угнали", "учётку угнали", "угнали мой аккаунт"
    _TAKE + r"\s+(?:\w+\s+){0,3}?" + _ACC, _ACC + r"\s+(?:\w+\s+){0,3}?" + _TAKE,
    # "к моему профилю привязали чужую почту", "посторонний человек вошёл в мой аккаунт", "в мой аккаунт кто-то вошёл"
    r"(?:мо\w+|мне|меня)\s+(?:\w+\s+){0,2}?" + _ACC + r"\s+(?:\w+\s+){0,6}?" + _FOREIGN,
    _FOREIGN + r"\s+(?:\w+\s+){0,4}?(?:в|на|к|с)\s+мо\w+\s+(?:\w+\s+){0,1}?" + _ACC,
    r"(?:в|на|к)\s+мо\w+\s+" + _ACC + r"\s+(?:\w+\s+){0,3}?" + _FOREIGN,
    # "мне сменили почту/пароль" (the mail or the password of my account changed by someone else)
    r"мо\w+\s+(?:\w+\s+){0,3}?(?:почт\w*|парол\w*|email|e-mail)\s+(?:\w+\s+){0,3}?" + _CHANGE,
    _CHANGE + r"\s+(?:\w+\s+){0,2}?мо\w+\s+(?:почт\w*|парол\w*)", _CHANGE + r"\s+(?:\w+\s+){0,3}?(?:почт\w*|парол\w*)\s+(?:\w+\s+){0,4}?" + _ACC,
    # "мне отвязали почту", "у меня сменили почту", "в профиле теперь чужая почта", "больше не контролирую свою учётную запись"
    r"(?:мне|у\s+меня|с\s+моего|с\s+моей)\s+(?:\w+\s+){0,2}?" + _CHANGE + r"\s+(?:\w+\s+){0,2}?(?:почт\w*|email|e-mail|номер\w*|телефон\w*|парол\w*)",
    _CHANGE + r"\s+(?:\w+\s+){0,2}?(?:электронн\w+\s+)?(?:почт\w*|email|e-mail)",
    _ACC + r"\s+(?:\w+\s+){0,3}?(?:чуж\w*|не\s+моя|не\s+мой|посторонн\w*)\s+(?:\w+\s+){0,1}?(?:почт\w*|email|e-mail|номер\w*|телефон\w*)",
    r"(?:чуж\w*|не\s+моя)\s+(?:электронн\w+\s+)?(?:почт\w*|email|e-mail)\s+(?:\w+\s+){0,3}?(?:в|на|к)\s+(?:мо\w+\s+)?" + _ACC,
    r"(?:не\s+контролир\w*|потерял\w*\s+контроль|нет\s+контроля|утратил\w*\s+контроль)\s+(?:\w+\s+){0,3}?" + _ACC,
    _ACC + r"\s+(?:\w+\s+){0,3}?(?:не\s+под\s+моим\s+контролем|больше\s+не\s+мо[йяе])",
    r"(?:lost\s+control|no\s+longer\s+control)\s+(?:\w+\s+){0,2}?(?:account|profile)", r"(?:someone|somebody)\s+(?:\w+\s+){0,2}?(?:changed|unlinked|removed)\s+(?:\w+\s+){0,2}?(?:email|mail)",
    # Kazakh: "аккаунтымның поштасын басқа адам ауыстырды", "бөтен адам аккаунтыма кірді"
    r"аккаунт\w*\s+(?:\w+\s+){0,3}?(?:басқа|бөтен|бейтаныс|біреу)\w*\s+(?:\w+\s+){0,2}?(?:ауыстыр|өзгерт|кір|бұз|ұрла|алмастыр)\w*",
    r"(?:басқа|бөтен|бейтаныс)\s+(?:адам\s+)?(?:\w+\s+){0,3}?аккаунт\w*\s+(?:\w+\s+){0,2}?(?:кір|бұз|ұрла|ауыстыр|өзгерт)\w*",
    r"(?:біреу|кімдір)\s+(?:\w+\s+){0,3}?аккаунт\w*",
    # Uzbek: "akkauntimga begona odam kirgan", "akkauntimni boshqa odam oldi"
    r"akkaunt\w*\s+(?:\w+\s+){0,3}?(?:begona|notanish|boshqa|o['`]?zga)\s+(?:odam\s+)?(?:\w+\s+){0,2}?(?:kir|buz|o['`]?g['`]?irla|oldi|almashtir)\w*",
    r"(?:begona|notanish|boshqa)\s+(?:odam\s+)?(?:\w+\s+){0,3}?akkaunt\w*", r"(?:birov|kimdir)\s+(?:\w+\s+){0,3}?akkaunt\w*",
    r"аккаунт\w*\s+(?:\w+\s+){0,3}?(?:бегона|нотаниш|бошқа)\s+(?:одам\s+)?(?:\w+\s+){0,2}?(?:кир|буз)\w*",
    # kk/uz: control and the bound mail ("аккаунтымды басқара алмаймын", "поштамды біреу өзгертті", "akkauntimni boshqara olmayman", "pochtamni o'zgartirishdi")
    r"аккаунт\w*\s+(?:\w+\s+){0,2}?басқара\s+алмай\w*", r"(?:пошта|почта)\w*\s+(?:\w+\s+){0,2}?(?:біреу|басқа\s+адам)\s+(?:\w+\s+){0,1}?(?:өзгерт|ауыстыр|ажырат)\w*",
    r"(?:пошта|почта)\w*\s+(?:\w+\s+){0,2}?(?:ажыратып|өзгертіп)\s+тастады\w*",
    r"akkaunt\w*\s+(?:\w+\s+){0,2}?boshqara\s+olmay\w*", r"pochta\w*\s+(?:\w+\s+){0,2}?(?:o['`]?zgartirish\w*|almashtirish\w*|uzib\s+qo['`]?yish\w*|o['`]?chirish\w*)",
    r"pochta\w*\s+(?:\w+\s+){0,2}?(?:boshqa|begona)\s+(?:odam\w*|kishi\w*)",
    # English / Spanish / Portuguese
    r"(?:someone|somebody|stranger|unknown\s+(?:person|user))\s+(?:\w+\s+){0,3}?(?:logged|got|broke|entered|accessed|changed|took|stole)\s+(?:\w+\s+){0,4}?(?:account|profile|email|password)",
    r"(?:alguien|desconocido|extra[ñn]o)\s+(?:\w+\s+){0,3}?(?:entr[oó]|accedi[oó]|cambi[oó]|rob[oó])\s+(?:\w+\s+){0,3}?(?:cuenta|correo|contrase[ñn]a)",
    r"(?:algu[eé]m|desconhecido|estranho)\s+(?:\w+\s+){0,3}?(?:entrou|acessou|mudou|roubou)\s+(?:\w+\s+){0,3}?(?:conta|e-?mail|senha)",
)
HACK_WORDS = re.compile(HACK_WORDS.pattern + "|" + HACK_TAKEOVER.pattern, re.I | re.UNICODE)
# lost access to one's own account: "не могу зайти", "кіре алмаймын", "kira olmayman": always a person, never an automatic answer
ACCESS_LOSS = _rx(
    r"не\s*могу\s+(?:зайти|войти|залогин\w*|авториз\w*)\s+(?:в|на)\s+(?:\w+\s+){0,2}?(?:акк\w*|игр\w*|учетн\w*|профил\w*)", r"не\s*могу\s+(?:зайти|войти)\s*$", r"не\s+пуска\w+\s+(?:в|на)\s+(?:игр\w*|акк\w*)",
    r"потерял\w*\s+(?:доступ|акк\w*|пароль)", r"доступ\w*\s+(?:к\s+акк\w*\s+)?(?:потерян|утрачен|пропал)\w*", r"нет\s+доступа",
    r"забыл\w*\s+(?:пароль|логин)", r"акк\w*\s+не\s+открыва\w*", r"восстанов\w*\s+(?:доступ|акк\w*|пароль)",
    r"не\s+прих\w+\s+(?:код|смс|sms)",
    r"не\s+(?:могу|получается|удается|удаётся)\s+(?:\w+\s+){0,1}?восстанов\w*\s+(?:\w+\s+){0,2}?(?:акк\w*|аккаунт\w*|учетн\w*|учетк\w*|профил\w*|доступ\w*)",
    r"аккаунт\w*\s+(?:\w+\s+){0,2}?қалпына\s+келтіре\s+алмай\w*", r"akkaunt\w*\s+(?:\w+\s+){0,2}?tiklay\s+olmay\w*",
    r"кіре\s+алмай\w*", r"кіре\s+алма\w*", r"кірмейді", r"кіруге\s+болмай\w*", r"кіру\s*мүмкін\s*емес", r"қолжетім\w*\s+жоқ",
    r"қол\s*жеткізе\s+алмай\w*", r"құпия\s*сөз\w*\s+ұмыт\w*", r"пароль\w*\s+ұмыт\w*",
    r"kira\s+olmay\w*", r"kirolmay\w*", r"kirib\s+bo['`]?lmay\w*", r"kirish\s+imkon\w*", r"parol\w*\s+(?:unut|yo['`]?q)\w*",
    r"parolni\s+unut\w*",
    r"кира\s+олмай\w*", r"кира\s+олмаяпман", r"кириб\s+бўлмай\w*", r"парол\w*\s+унут\w*",
    r"(?:can'?t|cannot|can\s+not|unable\s+to|couldn'?t)\s+(?:log\s*in|login|sign\s*in|enter|access)", r"locked\s+out",
    r"lost\s+(?:access|my\s+account|my\s+password)", r"forgot\s+(?:my\s+)?(?:password|login)", r"no\s+access\s+to\s+(?:my\s+)?account",
    r"no\s+(?:puedo|logro|consigo)\s+(?:entrar|acceder|iniciar\s+sesi[oó]n|ingresar)", r"perd[ií]\s+(?:el\s+)?acceso",
    r"olvid[eé]\s+(?:mi\s+)?(?:contrase[ñn]a|clave)", r"olvidado\s+(?:mi\s+)?(?:contrase[ñn]a|clave)", r"esquecido\s+(?:minha\s+)?senha",
    r"n[aã]o\s+(?:consigo|posso)\s+(?:entrar|acessar|logar|fazer\s+login)", r"perdi\s+o\s+acesso", r"esqueci\s+(?:minha\s+)?senha",
)
HELP = _rx(
    r"застр\w*", r"баг(?:и|нул\w*|ан\w*)?(?![\w])", r"текстур\w*", r"tekstur\w*", r"qolib\s+ket\w*", r"қолиб\s+кет\w*", r"перевер\w*", _tok("порт", "тп"), r"телепорт\w*", r"вытащ\w*", r"упал\w*",
    r"в\s+воду", r"в\s+воде", r"наручник\w*", _tok("нары"), r"развяж\w*", r"почини\w*", r"починить", r"заглох", r"выбраться",
    r"не\s*могу\s+(?:выйти|двигать|пошевел)\w*", r"завис\w*", r"вылета\w*", _tok("лаг", "лаги"), r"где\s+я(?![\w])", r"унесите",
    _tok("спавн"), r"заспавн\w*", r"stuck", r"glitch", r"teleport", r"tiqilib", r"qotib", r"yopishib", r"ushlab", r"тұрып\s+қал\w*",
    r"qalib", r"atascad\w*", r"preso", r"presa", r"перевернулся", r"на\s+дорогу", r"поставьте", r"верните\s+на",
    r"размороз\w*", r"заморож\w*", r"аударыл\w*", r"қалып\s+қой\w*", r"tuzat\w*", r"tuzativor\w*", r"tuzatib", r"remont\w*", r"ta['`ʻ]?mirla\w*",
    r"voltear", r"volcar\w*", r"capotou", r"virou",
)
ACCOUNT_KK_UZ = _rx(r"аккаунт\w*\s+бұз\w*", r"бұзыл\w*", r"ұрла\w*", r"akkaunt\w*\s+buz\w*", r"buzdi", r"buzildi", r"o['`ʻ‘’]?g['`ʻ‘’]?irla\w*", r"таппай\s+қал\w*", r"жоғал\w*", r"жоғалды", r"yo['`ʻ‘’]?qol\w*", r"topolmay\w*", r"g['`ʻ‘’]?oyib", r"ғайып")
WHAT_IS = _rx(r"что\s+такое", r"(?<![\w])это\s+что\s*$", r"что\s+значит", r"what\s+is", r"what\s+does", r"nima\s+(?:edi|degani|u(?![\w]))", r"не\s+деген", r"qu[eé]\s+es", r"o\s+que\s+[eé]")
DELETE_ACCOUNT = _rx(r"удал\w*\s+(?:мой\s+)?(?:акк|аккаунт)\w*", r"delete\s+(?:my\s+)?account", r"akkaunt\w*\s+o['`ʻ‘’]?chir\w*", r"аккаунтты\s+жою")
MASS_BUG = _rx(
    r"(?<![\w])у\s+всех(?![\w])", r"всем\s+серверам", r"на\s+всех\s+сервер\w*", r"весь\s+сервер", r"сервер\s+(?:лежит|упал|не\s+работает|крашнул\w*)",
    r"массов\w*\s+(?:баг|краш)", r"(?<![\w])краш\w*\s+(?:сервер\w*|у\s+всех)", r"(?<![\w])сервер\w*\s+краш\w*", r"сервер\s+не\s+(?:открывается|грузится)",
)
REPORT_COMPLAINT = _rx(
    _tok("дм", "dm", "дб", "db", "тк", "tk", "ск", "sk", "нрп", "nrp", r"нон\s*рп", "афк", "afk"), r"убива\w*", r"убил\w*", r"оскорб\w*",
    r"(?<![\w])оск\w*", r"материт\w*", r"читер\w*", r"заломал\w*",
    r"(?<![\w])чит(?:ы|ит|ов)?(?![\w])", r"софт\w*", _tok("спек", "spec"), r"кикни\w*", r"выгони\w*", r"накажи\w*", r"(?<![\w])бан(?![\w])",
    r"(?<![\w])мут(?![\w])", r"мешает", r"кидал\w*", r"обман\w*", r"реклам\w*", r"стреля\w*", r"издева\w*", r"тролл\w*", r"дмят", r"дмит",
    r"убивают", r"стреляют", r"сбивает", r"killing", r"insult\w*", r"cheater",
)
OFFTOPIC_FILLER = {
    "помогите", "помоги", "помогит", "помогить", "помгите", "помагите", "памагите", "памагi", "памаги", "помаги", "помощь", "хелп",
    "help", "sos", "сос", "спасите", "пожалуйста", "пж", "плиз", "pls", "please", "админ", "админы", "админа", "срочно", "ау", "привет",
    "здравствуйте", "хай", "hi", "hello", "hola", "yordam", "bering", "спасибо", "пасиб", "ку", "алло", "ало", "эй", "эй,", "есть",
    "кто", "тут", "здесь", "ребят", "ребята", "люди", "а", "и", "ну", "да", "нет", "ok", "ок", "олá", "ola", "ajuda", "ayuda",
    "салем", "сәлем", "rahmat", "salom", "assalomu", "alaykum", "aleykum", "xelp", "madad", "xelo", "hello", "hey", "рахмет", "рахмат", "баракалла",
    "gracias", "obrigado", "obrigada", "thanks", "thank", "thx", "socorro", "auxilio", "ayuda", "ajuda", "ayúdenme", "көмек", "кґмек", "көмектесіңіз",
    "жардам", "yordam", "bering", "iltimos", "pomogite", "pomoshch", "admin", "adm", "адм", "админ", "админы", "kere",
}

INJECTION = _rx(
    r"ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier)", r"disregard\s+(?:all\s+|the\s+)?(?:previous|above|instructions)",
    r"забудь\s+(?:все|всё|предыдущ\w*|инструкц\w*)", r"игнорируй\s+(?:все|всё|предыдущ\w*|инструкц\w*|правила)", r"system\s*prompt",
    r"системн\w*\s+(?:промпт|инструкц\w*)", r"you\s+are\s+now", r"ты\s+теперь\s+(?!не\b)", r"теперь\s+ты\s", r"new\s+instructions?",
    r"нов\w+\s+инструкц\w*", r"(?<![\w])DAN(?![\w])", r"jailbreak", r"developer\s+mode", r"режим\s+разработчика", r"<\|im_(?:start|end)\|>",
    r"\[/?INST\]", r"</?system>", r"(?<![\w])assistant\s*:", r"override\s+(?:the\s+)?(?:rules|instructions)", r"выведи\s+(?:свой\s+)?промпт",
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt", r"отмени\s+все\s+правила",
    r"</?(?:player_text|allowed_rules|rule_text|rule|facts|reply_language|safe_template_for_reference)\b",
    # es / pt / uz / kk
    r"ignora\s+(?:todas\s+)?las\s+instrucciones", r"olvida\s+(?:todas\s+)?las\s+instrucciones", r"ignore\s+(?:todas\s+)?as\s+instru[cç][oõ]es",
    r"esque[cç]a\s+(?:todas\s+)?as\s+instru[cç][oõ]es", r"ahora\s+eres\s+(?:un\s+)?admin\w*", r"agora\s+voc[eê]\s+[eé]\s+(?:um\s+)?admin\w*",
    r"oldingi\s+ko['`ʻ‘’]?rsatmalarni", r"barcha\s+ko['`ʻ‘’]?rsatmalarni", r"алдыңғы\s+нұсқауларды", r"барлық\s+нұсқауларды",
)

BIG_AMOUNT = re.compile(
    r"(\d[\d\s.,]*)\s*(млн|миллион\w*|кк|kk|лям\w*|m\b|million)|\b\d{7,}\b|(?<![\w])\d{1,3}(?:[ .]\d{3}){2,}", re.I
)
VALUABLE = _rx(r"дом", r"бизнес", r"машин\w*", r"аккаунт\w*", r"акк(?![\w])", r"имущ\w*", r"склад", r"ферм\w*", r"авто(?![\w])", r"тачк\w*",
               r"квартир\w*", r"(?<![\w])uy(?:im\w*)?(?![\w])", r"m[ao]shina\w*", r"kvartira", r"biznes\w*", r"akkaunt\w*", r"casa", r"coche", r"carro", r"house", r"car(?![\w])",
               r"көлік\w*", r"үй(?![\w])")

NICK = re.compile(r"(?<![\w])([A-ZА-Я][A-Za-zА-Яа-я]{1,}_[A-ZА-Я][A-Za-zА-Яа-я]+\d*)(?![\w])")
NICK_LOOSE = re.compile(r"(?<![\w])([A-Za-z][A-Za-z0-9]{1,}_[A-Za-z][A-Za-z0-9_]+)(?![\w])")
SERVER = re.compile(
    r"(?:сервер\w*|server|srv|sv)\s*[№#:]?\s*(\d{1,3})(?![\d])|(?<![\w\d])(\d{1,3})\s*(?:сервер\w*|server)(?![\w])", re.I
)
REF_RE = re.compile(
    r"(?:(?P<pfx>жб|zhb|jb|gz|гз|зз|zz)\s*)?(?P<num>\d{1,2}(?:\s*[.,]\s*\d{1,2}){1,3})\s*(?P<sfx>оп|op|gz|гз|зз|жб|жалоб\w*)?",
    re.I,
)


# --------------------------------------------------------------------------------------------
# result containers


@dataclass
class Analysis:
    kind: str
    raw_text: str
    text: str  # game-character-normalized text of the player's messages
    clean: str
    category: str
    priority: str
    confidence: float
    attachments: int
    nickname: Optional[str]
    claimed_refs: list[str]  # normalized as written by the player, e.g. "2.1", "GZ 1.1"
    has_evidence: bool
    server: Optional[str]
    flags: set[str] = field(default_factory=set)
    scores: dict[str, float] = field(default_factory=dict)
    matched: dict[str, list[str]] = field(default_factory=dict)
    gamechars_fixed: int = 0
    reasons: list[str] = field(default_factory=list)
    needs_human: bool = True
    suspected_category: Optional[str] = None  # what it would be if evidence were present
    topic: Optional[str] = None  # the topical complaint category (insult/dm/cheat/...) even when the ticket is a `report`
    evidence_status: str = "none"  # attached | claimed_only | stated_absent | asked_about | none
    has_qmark: bool = False
    sensitive: list[str] = field(default_factory=list)  # mandatory-handoff topics found anywhere in the text (any category)
    ml_proba: Optional[dict[str, float]] = None  # class probabilities of the learned complaint model, when it is loaded
    toxicity: Optional[ToxicityReport] = None  # contextual abusive-language analysis (app/toxicity.py)


def _hits(rx: re.Pattern, text: str) -> list[str]:
    return [m.group(0).strip() for m in rx.finditer(text)]


_SELF_INTRO = re.compile(
    r"(?:мой|моего|мое|моё|my|mi|meu|mening|менің)\s+(?:ник|никнейм|nick|nickname|nik|apodo|apelido)\w*\s*[:\-]?\s*[\w]+(?:_[\w]+)?|"
    r"меня\s+зовут\s+[\w]+(?:_[\w]+)?|(?<![\w])я\s*,?\s+[a-z][a-z0-9]*_[a-z][a-z0-9]*", re.I | re.UNICODE)
CONTEXT_CHARS = 2000  # what the lexicon category / the learned model / the LLM see (head + tail, so the LAST message is never dropped)


def player_text_full(req: TriageRequest) -> str:
    """All the player's messages. The request size is bounded by the schema (app/schemas.py), so the safety flags can read all of it."""
    return " \n".join(m.text for m in req.messages if m.role == "player" and m.text).strip()


def player_text(req: TriageRequest) -> str:
    """Classification context: the whole text when short; otherwise its beginning AND its end (a hack reported in the last of many
    messages must not disappear), with a visible gap marker."""
    full = player_text_full(req)
    if len(full) <= CONTEXT_CHARS:
        return full
    return full[: CONTEXT_CHARS - 600].rstrip() + " … " + full[-560:].lstrip()


def reported_text(req: TriageRequest) -> str:
    parts = [m.text for m in req.messages if m.role == "reported_player" and m.text]
    return " \n".join(parts).strip()[:1000]


def attachments_total(req: TriageRequest) -> int:
    return sum(m.attachments for m in req.messages if m.role == "player")


def parse_rule_refs(*texts: str) -> list[str]:
    """Extract rule references the *player* wrote. These are claims, never facts."""
    found: list[str] = []
    for raw in texts:
        if not raw:
            continue
        t = raw.lower()
        # "3,5,7" / "1, 3" are lists of section numbers, not decimal references
        if re.fullmatch(r"\s*\d+(?:\s*[,;]\s*\d+){1,}\s*", t) and not re.fullmatch(r"\s*\d,\d\s*", t):
            continue
        for m in REF_RE.finditer(t):
            num = re.sub(r"\s*", "", m.group("num")).replace(",", ".")
            pfx = (m.group("pfx") or "").lower()
            sfx = (m.group("sfx") or "").lower()
            tag = ""
            if pfx in {"жб", "zhb", "jb"} or sfx.startswith("жалоб") or sfx == "жб":
                tag = "ЖБ "
            elif pfx in {"gz", "гз", "зз", "zz"} or sfx in {"gz", "гз", "зз"}:
                tag = "GZ "
            ref = f"{tag}{num}"
            if ref not in found:
                found.append(ref)
    return found


_SELF_BEFORE = re.compile(
    r"(?:(?<![\w])я|мо[йяеё]\s+ник(?:нейм)?|меня\s+зовут|мой\s+персонаж|ник\s+жалобщика|my\s+nick(?:name)?|i\s+am|i'm|this\s+is|from|от\s+(?:игрока\s+)?)[\s:,\-–—]*$", re.I)
_ADMIN_BEFORE = re.compile(r"(?:админ\w*|администратор\w*|модератор\w*|admin\w*|moderator)[\s:,\-–—]*$", re.I)
_ACCUSE_BEFORE = re.compile(
    r"(?:жалу\w+\s+на|жалоба\s+на|жалобу\s+на|на\s+игрока|на\s+нарушителя|игрок\w*|нарушител\w*|against|player|offender|reporting|vs\.?)[\s:,\-–—]*$", re.I)
_VIOLATION_AFTER = re.compile(
    r"^[\s:,]*(?:он\s+|она\s+)?(?:оскорб\w*|оск(?![\w])|убил\w*|убива\w*|кинул\w*|обман\w*|использ\w*|украл\w*|взломал\w*|угрож\w*|продает|продаёт|рекламир\w*|читер\w*|"
    r"insult\w*|kill\w*|scam\w*|cheat\w*|haqorat\w*|aldadi|sokdi|қорла\w*|алдад\w*)", re.I)


# Every reported_nick in train+valid (68,572 values) has this form; the field is still player-controlled input, so it is validated
# AS A WHOLE: a value with sentences, spaces or punctuation is dropped (never trimmed to a safe-looking prefix).
NICK_VALID = re.compile(r"[A-Za-z][A-Za-z0-9]{0,23}_[A-Za-z][A-Za-z0-9]{0,23}")
NICK_MAX_LEN = 32


def valid_nick(value: Optional[str]) -> bool:
    return bool(value) and len(value) <= NICK_MAX_LEN and bool(NICK_VALID.fullmatch(value))


def extract_nickname(req: TriageRequest, text: str) -> Optional[str]:
    """The ACCUSED player. `reported_nick` (a field the game fills) wins. Otherwise: the complainant ("я, X", "мой ник X") and the administrator
    ("админ X") are excluded; among the rest a nick after an accusation word ("жалуюсь на X", "игрок X") or before a violation verb is preferred;
    with several candidates and no such marker the field stays undefined rather than guessing."""
    if req.reported_nick and valid_nick(req.reported_nick):
        return req.reported_nick
    cands = []
    for m in list(NICK.finditer(text)) + [m for m in NICK_LOOSE.finditer(text) if not NICK.fullmatch(m.group(1))]:
        before, after = text[max(0, m.start() - 45): m.start()], text[m.end(): m.end() + 40]
        if _SELF_BEFORE.search(before) or _ADMIN_BEFORE.search(before):
            continue
        marked = bool(_ACCUSE_BEFORE.search(before) or _VIOLATION_AFTER.search(after))
        if len(m.group(1)) <= NICK_MAX_LEN:
            cands.append((m.start(), m.group(1), marked))
    cands.sort()
    seen: list[tuple[str, bool]] = []
    for _, nick, marked in cands:
        if nick not in [n for n, _ in seen]:
            seen.append((nick, marked))
    marked_only = [n for n, mk in seen if mk]
    if marked_only:
        return marked_only[0]
    return seen[0][0] if len(seen) == 1 else None


def extract_server(text: str) -> Optional[str]:
    m = SERVER.search(text)
    if not m:
        return None
    return m.group(1) or m.group(2)


def _score(text: str, clean: str, claims_text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    def s(rx, w, cap=3):
        hits = _hits(rx, clean)
        return min(len(hits), cap) * w, hits

    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    spec = {
        "complaint.insult": [(INSULT, 3.0, 2), (PROFANITY, 2.5, 2), (INSULT_UZKK, 3.0, 1), (THREAT, 3.0, 1), (INSULT_ML, 3.0, 1)],
        "complaint.cheat": [(CHEAT, 3.5, 2), (CHEAT_ML, 3.5, 1)],
        "complaint.dm": [(DM, 3.0, 2), (DM_EXTRA, 2.5, 1), (DM_ML, 3.0, 1)],
        "complaint.scam": [(SCAM, 3.5, 2), (SCAM_ML, 3.5, 1)],
        "complaint.rmt_ads": [(RMT, 3.0, 2), (RMT_ML, 3.0, 1)],
        "complaint.admin": [(ADMIN_WORD, 2.0, 1), (ADMIN, 3.0, 2), (PUNISHMENT, 2.0, 1), (DENIAL, 4.0, 2), (ADMIN_ML, 4.0, 1)],
        "complaint.other": [(OTHER, 2.5, 2)],
    }
    claim = {
        "complaint.insult": INSULT_CLAIM, "complaint.cheat": CHEAT_CLAIM, "complaint.dm": DM_CLAIM,
        "complaint.scam": SCAM_CLAIM, "complaint.rmt_ads": RMT_CLAIM, "complaint.admin": ADMIN_CLAIM, "complaint.other": OTHER_CLAIM,
    }
    for cat, items in spec.items():
        total, hits_all = 0.0, []
        for rx, w, cap in items:
            sc, hits = s(rx, w, cap)
            total += sc
            hits_all += hits
        ch = _hits(claim[cat], claims_text)
        if ch:  # a rule number written by the player is weak evidence (players are often wrong)
            total += 1.5
            hits_all += ["claim:" + c for c in ch[:1]]
        scores[cat] = total
        matched[cat] = hits_all
    # appeal wording is a stronger sign of complaint.admin than a lone "админ"
    if _hits(APPEAL, clean):
        scores["complaint.admin"] += 2.0
    if _hits(PUNISHMENT, clean) and _hits(DEFENCE, clean):
        scores["complaint.admin"] += 4.0  # "бан выдан без причины": the author is the punished one
    # "админ" next to a concrete violation word is usually a different category
    return scores, matched


# Weight of the learned model in the blend with the lexicon. Chosen on a reporter-grouped hold-out of TRAIN
# (0.6 maximised silver agreement there); valid was only used to confirm it (docs/metrics/ml_classifier.json).
ML_BLEND_WEIGHT = 0.6
# The model gets its full weight only when at least this share of the words in the text was seen in training; below that the
# weight shrinks linearly (a Kazakh sentence of unseen words must not overrule a clear lexicon match on character fragments).
FULL_TRUST_COVERAGE = 0.5


def _blend(scores: dict[str, float], ml_proba: dict[str, float], coverage: float = 1.0) -> dict[str, float]:
    """w * P_model + (1 - w) * lexicon scores normalised to a distribution (uniform when the lexicon found nothing)."""
    # the union: a category the model was not trained on can still win on clear lexicon evidence (its model probability is 0)
    classes = list(ml_proba) + [c for c in scores if c not in ml_proba]
    w = ML_BLEND_WEIGHT * min(1.0, coverage / FULL_TRUST_COVERAGE)
    lex = {c: max(scores.get(c, 0.0), 0.0) for c in classes}
    total = sum(lex.values())
    if total <= 0:
        w = ML_BLEND_WEIGHT  # the lexicon found nothing: there is no rule evidence to fall back on
    return {c: w * ml_proba.get(c, 0.0) + (1 - w) * (lex[c] / total if total > 0 else 1 / len(classes)) for c in classes}


def classify_complaint(clean: str, claims_text: str, attachments: int, has_ev_words: bool,
                       ml_proba: Optional[dict[str, float]] = None, coverage: float = 1.0,
                       boost: Optional[dict[str, float]] = None) -> tuple[str, float, dict, dict, Optional[str]]:
    scores, matched = _score(clean, clean, claims_text)
    for cat, add in (boost or {}).items():  # evidence from the toxicity analysis (quoted/reported abuse)
        scores[cat] = scores.get(cat, 0.0) + add
        matched.setdefault(cat, []).append("toxicity:reported")
    if ml_proba:
        return _classify_complaint_blended(clean, scores, matched, ml_proba, coverage)
    suspected = None
    tokens = clean.split()
    informative = [t for t in tokens if t not in JUNK_TOKENS and not re.fullmatch(r"[\W\d]+", t)]
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, top_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_s <= 0:
        # nothing matched: junk/question -> invalid, otherwise unknown violation -> other
        if not informative or QUESTION.search(clean) or len(clean) < 3:
            return "complaint.invalid", 0.7 if not informative else 0.55, scores, matched, None
        return "complaint.other", 0.35, scores, matched, None
    # taxonomy example: "никнейм с оскорблением" belongs to complaint.other (rule 1.6), not to insults
    if top == "complaint.insult" and NICKNAME_COMPLAINT.search(_SELF_NICK.sub(" ", clean)) and len(_hits(INSULT, clean)) <= 1 and not _hits(PROFANITY, clean):
        top, second_s = "complaint.other", top_s
        top_s = max(scores["complaint.other"], top_s - 0.5)
    margin = (top_s - second_s) / max(top_s, 1.0)
    conf = 0.35 + 0.35 * min(top_s / 6.0, 1.0) + 0.25 * margin
    if second_s > 0 and margin < 0.25:
        conf = min(conf, 0.5)
    if matched[top] and all(h.startswith("claim:") for h in matched[top]):
        conf = min(conf, 0.5)  # category inferred only from the rule number the player typed
        suspected = "claim_only"
    return top, round(min(conf, 0.95), 2), scores, matched, suspected


def _classify_complaint_blended(clean: str, scores: dict, matched: dict, ml_proba: dict[str, float],
                                coverage: float = 1.0) -> tuple[str, float, dict, dict, Optional[str]]:
    """Lexicon + learned model. The model has no `complaint.invalid` class (admins' replies give no such label), so the
    lexicon's structural decision stays: a ticket with no informative content or a mere question is `invalid`."""
    tokens = clean.split()
    informative = [t for t in tokens if t not in JUNK_TOKENS and not re.fullmatch(r"[\W\d]+", t)]
    lex_top = max(scores, key=lambda k: scores[k])
    lex_cat = lex_top if scores[lex_top] > 0 else None
    matched["_lexicon_category"] = [lex_cat]
    if lex_cat is None and (not informative or QUESTION.search(clean) or len(clean) < 3):
        return "complaint.invalid", 0.7 if not informative else 0.55, scores, matched, None
    blended = _blend(scores, ml_proba, coverage)
    ranked = sorted(blended.items(), key=lambda kv: -kv[1])
    top, top_p = ranked[0]
    second_p = ranked[1][1] if len(ranked) > 1 else 0.0
    if top == "complaint.insult" and NICKNAME_COMPLAINT.search(clean) and len(_hits(INSULT, clean)) <= 1 and not _hits(PROFANITY, clean):
        top = "complaint.other"  # taxonomy example: "никнейм с оскорблением" belongs to complaint.other (rule 1.6)
    conf = blended[top]
    if top_p - second_p < 0.15:
        conf = min(conf, 0.5)  # two categories almost tie: ambiguous
    suspected = None
    if lex_cat == top and matched.get(top) and all(h.startswith("claim:") for h in matched[top]) and ml_proba.get(top, 0) < 0.5:
        conf = min(conf, 0.5)  # only the player's rule number supports it, and the model is not convinced either
        suspected = "claim_only"
    matched["model"] = [f"{c}:{p:.2f}" for c, p in sorted(ml_proba.items(), key=lambda kv: -kv[1])[:2]]
    return top, round(max(0.3, min(conf, 0.95)), 2), scores, matched, suspected


def _gibberish(clean: str) -> bool:
    """Keyboard mashing: a long 'word' with almost no vowels or digits mixed into letters."""
    for w in clean.split():
        if len(w) >= 12:
            vowels = sum(1 for c in w if c in "аеёиоуыэюяaeiouіәөұү")
            if vowels / len(w) < 0.2 or re.search(r"[a-zа-я]\d[a-zа-я]", w):
                return True
    return False


def classify_report(clean: str, text: str, flags: Optional[set] = None) -> tuple[str, float, dict, dict]:
    """Category of a `report`. `flags` are the mandatory-handoff topics found anywhere in the text: they are applied
    AFTER the usual scoring, so a refund request hidden behind a rule question can no longer be answered as a question."""
    cat, conf, scores, matched = _classify_report(clean, text)
    flags = flags or set()
    if cat in {"report.question", "report.offtopic", "report.help", "report.complaint"}:
        # an account takeover / a refund request decides the category even when the same message also asks "что такое DM?" or names a player
        if flags & {"hack", "access"}:
            return "report.account", max(conf, 0.7), scores, matched
        if "payment" in flags:
            return "report.payment", max(conf, 0.7), scores, matched
    return cat, conf, scores, matched


def _classify_report(clean: str, text: str) -> tuple[str, float, dict, dict]:
    matched: dict[str, list[str]] = {}
    tokens = clean.split()
    informative = [t for t in tokens if t not in OFFTOPIC_FILLER and not re.fullmatch(r"[\W\d_]+", t)]
    clean_pay = NEG_PAY_REQUEST.sub(" ", clean)
    pay, acc = _hits(PAYMENT, clean_pay) or _hits(PAYMENT_STRONG, clean_pay), _hits(ACCOUNT, clean)
    acc = acc + _hits(ACCOUNT_KK_UZ, clean) + _hits(ACCESS_LOSS, clean)
    cmp_ = _hits(REPORT_COMPLAINT, clean) + _hits(DM_EXTRA, clean) + _hits(INSULT_UZKK, clean) + _hits(DM_ML, clean) + _hits(INSULT_TYPO, clean) + _hits(SCAM, clean) + _hits(SCAM_ML, clean) + _hits(CHEAT_ML, clean) + _hits(INSULT_ML, clean)
    hlp = _hits(HELP, clean)
    matched.update({"report.payment": pay, "report.account": acc, "report.complaint": cmp_, "report.help": hlp})
    scores = {
        "report.payment": 4.0 * min(len(pay), 2), "report.account": 3.5 * min(len(acc), 2),
        "report.complaint": 3.0 * min(len(cmp_), 2), "report.help": 3.0 * min(len(hlp), 2),
    }
    # "аккаунт" alone is weak (players write it in questions too); require a trouble word
    if scores["report.account"] and not _hits(HACK_WORDS, clean) and not _hits(ACCESS_LOSS, clean) and not re.search(
        r"пропал|исчез|не могу|забыл|потерял|доступ|украл|угнал|восстанов|кто.то|login|parol\w*\s+(?:unut|yo)|buzil|buzish|buzdi|perd|robad|таппай|жоғал|бұз|ұрла|yo.qol|topolmay", clean
    ):
        scores["report.account"] = 0.0  # the bare word "аккаунт" without any trouble word is not an incident
    if not informative:
        return "report.offtopic", 0.8 if len(clean) < 25 else 0.6, scores, matched
    if (WHAT_IS.search(clean) or DELETE_ACCOUNT.search(clean) or (TERM_ANY.search(clean) and (QUESTION.search(clean) or "?" in text))) \
            and not _hits(HACK_WORDS, clean) and not VICTIM_CUE.search(clean) and not scores["report.payment"]:
        matched["report.question"] = ["term-or-definition-question"]
        return "report.question", 0.68, scores, matched  # "что такое дм?", "как удалить аккаунт?" are questions, not incidents
    if (RULE_QUESTION.search(clean) and any(rx.search(clean) for _, rx in QUESTION_TOPICS) and not VICTIM_CUE.search(clean)
            and not scores["report.payment"] and not scores["report.account"]):
        matched["report.question"] = ["rule-question"]
        return "report.question", 0.72, scores, matched
    best = max(scores, key=lambda k: scores[k])
    if scores[best] <= 0:
        if _gibberish(clean):
            return "report.offtopic", 0.6, scores, matched
        if QUESTION.search(clean) and len(tokens) >= 2:
            return "report.question", 0.6, scores, matched
        # NOTE: merely mentioning "игрок" ("я купил машину у игрока") is NOT a complaint: without a violation
        # word the message is unclear, so it goes to a human with a low confidence and a clarifying question.
        if len(informative) <= 1 and len(clean) < 6:
            return "report.offtopic", 0.55, scores, matched
        return "report.question", 0.35, scores, matched  # unknown content: ask for clarification via a human
    # precedence: payment/account beat everything (always routed to a human anyway)
    order = ["report.payment", "report.account", "report.complaint", "report.help"]
    ranked = sorted(order, key=lambda c: (-scores[c], order.index(c)))
    top = ranked[0]
    second = scores[ranked[1]]
    margin = (scores[top] - second) / max(scores[top], 1.0)
    conf = 0.4 + 0.3 * min(scores[top] / 6.0, 1.0) + 0.25 * margin
    if second > 0 and margin < 0.2:
        conf = min(conf, 0.5)
    return top, round(min(conf, 0.95), 2), scores, matched


PRIORITY_BY_CATEGORY = {
    "complaint.insult": "p2", "complaint.cheat": "p1", "complaint.dm": "p2", "complaint.scam": "p2",
    "complaint.rmt_ads": "p1", "complaint.admin": "p2", "complaint.other": "p2", "complaint.invalid": "p3",
    "report.question": "p3", "report.help": "p2", "report.complaint": "p2", "report.account": "p0",
    "report.payment": "p0", "report.offtopic": "p3",
}
NEEDS_HUMAN_ALWAYS = {"report.account", "report.payment", "complaint.admin"}


PRIORITY_RANK = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
# Flags that force p0 whatever the category is (taxonomy: account/payment/threat/mass failure -> p0).
CRITICAL_PRIORITY_FLAGS = {
    "payment": "платёж, донат или возврат денег",
    "hack": "взлом аккаунта",
    "access": "потеря доступа к аккаунту",
    "threat_outside_game": "угроза вне игры",
    "mass_bug": "массовый сбой",
}
# Topics that always need a person and must survive any change of the main category.
HANDOFF_FLAGS = ("appeal", "hack", "access", "payment", "threat_outside_game", "mass_bug")
# Flags that stop "no evidence -> complaint.invalid -> please file again": the player must not get a plain refusal here.
NO_REFUSAL_FLAGS = {"payment", "hack", "access", "threat_outside_game", "appeal", "mass_bug"}


def compute_priority(category: str, flags: set, text: str, clean: str) -> tuple[str, list[str]]:
    """Priority policy. Evaluation order matters, and a later step may only RAISE the priority:
      1. the base priority of the category (taxonomy draft: PRIORITY_BY_CATEGORY);
      2. complaint.scam: p1 with a big amount or valuable property, otherwise p2 (still a base decision);
      3. mandatory raises to p0: payment/refund, hacking, lost access, threat outside the game, mass failure.
    Evidence never changes the priority (the taxonomy text 'complaints with proof -> p1' contradicts the PDF example insult + 2
    screenshots = p2, and 99% of complaints carry screenshots; the category decides). See docs/ARCHITECTURE.md."""
    reasons: list[str] = []
    prio = PRIORITY_BY_CATEGORY.get(category, "p2")
    if category == "complaint.scam":
        prio = "p1" if (BIG_AMOUNT.search(text) or VALUABLE.search(clean)) else "p2"
    for flag, why in CRITICAL_PRIORITY_FLAGS.items():
        if flag in flags and PRIORITY_RANK[prio] > PRIORITY_RANK["p0"]:
            prio = "p0"
            reasons.append(f"Приоритет повышен до p0: {why}")
    if "appeal" in flags and PRIORITY_RANK[prio] > PRIORITY_RANK["p2"]:  # an appeal sent through a report is still an appeal
        prio = "p2"
        reasons.append("Приоритет повышен до p2: апелляция наказания (как у complaint.admin)")
    return prio, reasons


_PUNCT_SPLIT = re.compile(r"[.,;:!?\n()«»\"]+")


def evidence_state(att: int, text: str, rule_refs_raw: str = "") -> tuple[str, bool, bool]:
    """(status, claims_evidence, text_denies_evidence).

    The attachment COUNT and the WORDING are separate facts and neither proves guilt (content is never inspected):
      attached       attachments > 0
      claimed_only   no attachment, but the text says there is proof / a link  ("скрины есть")
      stated_absent  the text explicitly says there is no proof                ("доказательств нет")
      asked_about    the text only asks which proof is needed                  ("какие доказательства нужны?")
      none           nothing about evidence
    If attachments exist while the text says "no proof", the attachments win (status attached) and the conflict is reported."""
    src = (text + " " + (rule_refs_raw or "")).lower().replace("ё", "е")
    link = "<url>" in src or bool(re.search(r"https?://|youtu\.?be|imgur", src))
    src = re.sub(r"\s+", " ", _PUNCT_SPLIT.sub(" | ", src))
    negated = bool(NEG_EVIDENCE.search(src))
    asked = bool(ASK_EVIDENCE.search(src))
    stripped = ASK_EVIDENCE.sub(" | ", NEG_EVIDENCE.sub(" | ", src))
    claimed = link or bool(EVIDENCE_WORDS.search(stripped))
    if att > 0:
        status = "attached"
    elif claimed:
        status = "claimed_only"
    elif negated:
        status = "stated_absent"
    elif asked:
        status = "asked_about"
    else:
        status = "none"
    return status, claimed, negated and not claimed


def prepare_text(req: TriageRequest) -> tuple[str, str, int, str]:
    """(raw player text, game-char-normalized text, number of fixed symbols, matching text).

    The matching text is what the lexicon AND the learned model see; training (scripts/train_classifier.py) calls this same
    function, so the model is never served a differently prepared text than it was trained on."""
    raw = player_text(req)
    text, fixed = normalize_game_chars(raw, req.server_lang)
    clean = clean_for_matching(text)
    # The «пункт правил» field is text typed by the player too, and in real data it often holds the real topic
    # ("оск родни", "дм", "связал в зз") while the message itself is just a link. It is scored as player text.
    raw_ref_clean = clean_for_matching(req.rule_refs_raw or "")
    if raw_ref_clean:
        clean = (clean + " " + raw_ref_clean).strip()
    # the complainant introducing himself ("мой ник Dias_Volkov", "я, Dias_Volkov") says nothing about the violation; left in, the word «ник»
    # made both the lexicon and the learned model read a nickname complaint (rule 1.6)
    clean = re.sub(r"\s+", " ", _SELF_INTRO.sub(" ", clean)).strip()
    return raw, text, fixed, clean


def analyze(req: TriageRequest, model=None) -> Analysis:
    """`model` is an optional app.ml.ComplaintModel; without it the category comes from the lexicon alone."""
    raw, text, fixed, clean = prepare_text(req)
    claims_source = " ".join(x for x in [text, req.rule_refs_raw or ""] if x)
    claims_clean = clean_for_matching(claims_source)
    claims_text = claims_clean + " " + clean_for_matching(req.rule_refs_raw or "")
    att = attachments_total(req)
    ev_status, has_ev_words, ev_denied = evidence_state(att, text, req.rule_refs_raw or "")
    flags: set[str] = set()
    reasons: list[str] = []
    has_qmark = "?" in text

    # ---- safety flags first: they never depend on the category or on the kind ----------------
    # They read the WHOLE dialogue (bounded by the schema), not the classification context: a sensitive last message is never cut off.
    full_text, _fx = normalize_game_chars(player_text_full(req), req.server_lang)
    sens = clean_for_matching(full_text + " " + (req.rule_refs_raw or ""))
    if INJECTION.search(text) or INJECTION.search(clean) or INJECTION.search(sens):
        flags.add("injection")
    # mandatory-handoff topics are detected independently of the final category, the ticket kind and the attachments
    if APPEAL.search(sens) or DENIAL.search(sens) or APPEAL_ML.search(sens):
        flags.add("appeal")
    if HACK_WORDS.search(sens):
        flags.add("hack")
    if ACCESS_LOSS.search(sens):
        flags.add("access")
    pay_clean = NEG_PAY_REQUEST.sub(" ", clean_for_matching(re.sub(r"[₽₸$€]", " руб ", full_text) + " " + (req.rule_refs_raw or "")))
    incident = bool(PAY_INCIDENT.search(pay_clean)) or bool(DOUBLE_CHARGE.search(pay_clean))
    refund = bool(REFUND_STRONG.search(pay_clean)) or bool(REAL_MONEY_CHANNEL.search(pay_clean))
    # "Как оплатить донат?" is a question about paying (the category scoring of a `report` handles it); "Верните 500 рублей за покупку. Можно ли
    # подать жалобу?" contains a refund request and is a payment incident; "вышел за донат из тюрьмы" mentions donate but is not one
    if incident or refund:
        flags.add("payment")
    if THREAT.search(sens) and threat_outside(sens):
        flags.add("threat_outside_game")
    if "access" in flags and "hack" not in flags and RESOLVED.search(sens):
        # "не мог зайти, но теперь всё работает": the player says the access problem is over; a takeover stays an incident even if solved
        flags.discard("access")
        reasons.append("Проблема с доступом, по словам игрока, уже решена: обязательная передача человеку по доступу не применяется")
    tox = analyze_messages(req.messages)
    if tox.real_life_threat:  # «найду тебя в реальной жизни», «o'ldiraman … uyingga»: same handling as the lexicon's threat flag
        flags.add("threat_outside_game")
    # abuse the author QUOTES or DESCRIBES (or the accused's own comment) is evidence of an insult complaint;
    # the author's own swearing is not (a victim who swears is still a victim, and profanity alone never raises the priority)
    abuse_about_other = [f for f in tox.findings if f.direction in {"quoted_or_reported", "accused_reply"}
                         and f.type in {tx.INSULT, tx.FAMILY, tx.HATE, tx.SEXUAL, tx.PROFANITY}]
    if MASS_BUG.search(sens):
        flags.add("mass_bug")

    suspected = None
    topic = None
    if req.kind == "complaint":
        ml_proba = model.proba(clean) if model is not None else None
        coverage = model.coverage(clean) if model is not None else 1.0
        boost = {"complaint.insult": 3.0} if abuse_about_other else None
        category, conf, scores, matched, suspected = classify_complaint(clean, claims_text, att, has_ev_words, ml_proba, coverage, boost)
        if ml_proba:
            lex_cat = matched.pop("_lexicon_category", [None])[0]
            top3 = sorted(ml_proba.items(), key=lambda kv: -kv[1])[:3]
            reasons.append("Категория: модель, обученная на решениях администраторов («" + ", ".join(f"{c} {p:.2f}" for c, p in top3)
                           + f"»), объединена со словарём (словарь: {lex_cat or 'нет совпадений'}; знакомых модели слов {coverage:.0%})")
        if suspected == "claim_only":
            flags.add("claim_only")
            reasons.append("Категория определена только по номеру пункта, указанному игроком (игрок мог ошибиться)")
            suspected = None
        # A complaint without any evidence (no attachment, no link, no claim) is 'invalid' per taxonomy, EXCEPT when the
        # ticket carries a mandatory-handoff topic (appeal, hacking, lost access, payment, a threat outside the game):
        # such a player must not be refused with "file it again". An appeal of one's OWN punishment is not a complaint
        # against someone either. In these cases the category stays and the draft uses the sensitive-topic strategy.
        if att == 0 and not has_ev_words and category not in {"complaint.invalid", "complaint.admin"} and not (flags & NO_REFUSAL_FLAGS):
            suspected = category
            category, conf = "complaint.invalid", min(conf, 0.6)
            if ev_status == "stated_absent":
                reasons.append("В тексте прямо сказано, что доказательств нет, вложений 0: по правилам жалоб (ЖБ) такая жалоба не принимается")
            else:
                reasons.append("Не приложено доказательств (вложений 0, ссылок нет): по правилам жалоб (ЖБ) такая жалоба не принимается")
            flags.add("no_evidence")
        elif att == 0 and not has_ev_words and category not in {"complaint.invalid", "complaint.admin"}:
            flags.add("no_evidence_sensitive")
            reasons.append("Доказательств нет, но тема чувствительная (угроза, взлом, доступ, платёж или апелляция): обращение не отклоняется, его рассматривает человек")
        topic = suspected or (category if category not in {"complaint.invalid", "complaint.admin"} else None)
    else:
        category, conf, scores, matched = classify_report(clean, text, flags)
        if category in {"report.offtopic", "report.question"} and any(f.direction == "quoted_or_reported" for f in abuse_about_other):
            category, conf = "report.complaint", max(conf, 0.6)  # «он пишет мне: ты …» is a report about a player, not a question
            matched["report.complaint"] = matched.get("report.complaint", []) + ["toxicity:reported"]
        if "mass_bug" in flags and category in {"report.help", "report.question"}:
            category = "report.help"
            conf = max(conf, 0.6)
        if category == "report.complaint":  # the subject of the violation decides which rules are relevant (retrieval by substance)
            sc, _m = _score(clean, clean, claims_text)
            best = max(sc, key=lambda k: sc[k])
            topic = best if sc[best] > 0 else None
    if att > 0 and ev_denied:
        reasons.append(f"В тексте написано, что доказательств нет, но вложений: {att}; содержимое вложений не проверялось, наличие вложения не подтверждает нарушение")
    elif req.kind == "complaint" and ev_status == "claimed_only":
        reasons.append("В тексте упомянуты доказательства (скрины, видео или ссылка), но вложений нет (0): это заявление игрока, а не проверенный факт")

    # ---- priority: base by category, then mandatory raises (never lowered afterwards) --------
    priority, prio_reasons = compute_priority(category, flags, text, clean)
    reasons.extend(prio_reasons)

    nick = extract_nickname(req, text)
    if req.reported_nick and not valid_nick(req.reported_nick):
        reasons.append("Поле reported_nick отклонено: это не ник игрока (ожидается Имя_Фамилия, до 32 символов, без пробелов и знаков); в ответе используется нейтральное «игрок»")
    claimed = parse_rule_refs(text, req.rule_refs_raw or "")
    server = extract_server(text)
    has_evidence = att > 0 or has_ev_words
    return Analysis(
        kind=req.kind, raw_text=raw, text=text, clean=clean, category=category, priority=priority, confidence=conf,
        attachments=att, nickname=nick, claimed_refs=claimed, has_evidence=has_evidence, server=server, flags=flags,
        scores=scores, matched={k: v for k, v in matched.items() if v}, gamechars_fixed=fixed, reasons=reasons,
        suspected_category=suspected, topic=topic, evidence_status=ev_status, has_qmark=has_qmark,
        sensitive=[f for f in HANDOFF_FLAGS if f in flags], ml_proba=ml_proba if req.kind == "complaint" else None, toxicity=tox,
    )
