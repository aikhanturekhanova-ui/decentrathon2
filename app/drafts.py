"""Template drafts: deterministic, citation-safe, no promises.

This is both the default mode (no API key needed) and the fallback when an LLM draft fails
validation. Every sentence comes from app/phrases.py; rule sentences are built only from points
that app/rag.py has already verified.
"""
from __future__ import annotations

from .classify import Analysis
from .knowledge import COMPLAINT_RULES_SECTION, GREEN_ZONE_SECTION, RuleBook, RuleDoc
from .phrases import GISTS, P

COMPLAINT_ACTS = {
    "complaint.insult": "act_insult",
    "complaint.cheat": "act_cheat",
    "complaint.dm": "act_dm",
    "complaint.scam": "act_scam",
    "complaint.rmt_ads": "act_rmt",
    "complaint.other": "act_other",
}


def reply_lang(detected: str, server_lang: str | None) -> str:
    """Language of the draft: the player's language; 'lat' (transliterated Russian) -> ru."""
    if detected in P:
        return detected
    if detected == "lat":
        return "ru"
    sl = (server_lang or "").lower()
    if sl == "kk":
        return "kk"
    if sl == "uz":
        return "uz"
    return "ru"


def _label(doc: RuleDoc, lang: str) -> str:
    T = P[lang]
    key = {GREEN_ZONE_SECTION: "label_gz", COMPLAINT_RULES_SECTION: "label_zb"}.get(doc.section_id, "label_gen")
    return T[key].format(ref=doc.ref)


def rule_sentence(doc: RuleDoc, lang: str, key: str = "rule_line") -> str:
    T = P[lang]
    gist = GISTS.get(doc.cite, {}).get(lang)
    label = _label(doc, lang)
    if gist is None:
        return T["rule_line_nogist"].format(label=label)
    return T[key].format(label=label, gist=gist)


def term_sentence(term, lang: str) -> str:
    """'From the rules section «RP terms»: DM — killing without a solid RP reason.' (no sanction is ever quoted)."""
    from .phrases import TERM_GISTS

    gist = TERM_GISTS[term.abbr][lang]
    return P[lang]["term_line"].format(abbr=term.abbr, gist=gist)


def _attachments(a: Analysis, lang: str) -> str:
    T = P[lang]
    return T["att_n"].format(n=a.attachments) if a.attachments > 0 else T["att_none"]


def compose(a: Analysis, cited: list[RuleDoc], claimed_irrelevant: list[str], lang: str, rb: RuleBook,
            is_duplicate: bool = False, dup_rule: RuleDoc | None = None, terms: list | None = None, zone_hint: str = "") -> tuple[str, list[RuleDoc]]:
    """Return (draft text, rule docs actually quoted in it)."""
    T = P[lang]
    cat = a.category
    parts: list[str] = [T["hello"]]
    used: list[RuleDoc] = []

    def quote(doc: RuleDoc, key: str = "rule_line") -> None:
        parts.append(rule_sentence(doc, lang, key))
        used.append(doc)

    if cat in COMPLAINT_ACTS:
        who = T["who_nick"].format(nick=a.nickname) if a.nickname else T["who_unknown"]
        parts.append(T["said"].format(who=who, act=T[COMPLAINT_ACTS[cat]]))
        parts.append(_attachments(a, lang))
        for d in cited:
            quote(d)
        for t in terms or []:
            parts.append(term_sentence(t, lang))
        if claimed_irrelevant:
            parts.append(T["claim_note"].format(refs=", ".join(claimed_irrelevant)))
        parts.append(T["next_complaint"])
        asks = []
        # no general DM rule exists in the rulebook: ask for what is MISSING (no place at all / a listed object without its city), never for what is already said
        if zone_hint == "ask_city":
            asks.append(T["ask_city"])
        elif zone_hint == "ask_place" and not cited and not terms:
            asks.append(T["ask_place"])
        if not a.nickname:
            asks.append(T["ask_nick"])
        if not a.server:
            asks.append(T["ask_when"])
        if a.attachments == 0:
            asks.append(T["ask_evidence"])
        parts.extend(asks)
    elif cat == "complaint.admin":
        parts.append(T["admin_body"])
        for d in cited:
            quote(d)
        if claimed_irrelevant:
            parts.append(T["claim_note"].format(refs=", ".join(claimed_irrelevant)))
        if a.attachments > 0:
            parts.append(_attachments(a, lang))
        parts.append(T["admin_ask"])
    elif cat == "complaint.invalid":
        first = cited[0].cite if cited else ""
        if first == "ЖБ 1.2.2" or "no_evidence" in a.flags:
            parts.append(T["inv_noev"])
            ask = T["inv_noev_ask"]
        elif first == "ЖБ 1.1.2":
            parts.append(T["inv_question"])
            urls = ", ".join(rb.urls[:2])
            ask = T["inv_question_ask"].format(urls=f" ({urls})" if urls else "")
        else:
            parts.append(T["inv_vague"])
            ask = T["inv_vague_ask"]
        for d in cited:
            quote(d)
        parts.append(ask)
    elif cat == "report.question":
        if cited or terms:  # a question about a rule or a defined game term: answer from the verified source only
            parts.append(T["q_rule_body"])
            for d in cited:
                quote(d)
            for t in terms or []:
                parts.append(term_sentence(t, lang))
            parts.append(T["q_rule_ask"])
        else:
            # a clearly formulated question does not need "what exactly are you interested in?" again
            parts += [T["q_body"], T["q_ask"] if len(a.clean.split()) < 4 else T["q_ask_short"]]
    elif cat == "report.help":
        parts.append(T["help_body"])
        for d in cited:
            quote(d, "off_rule")
        parts.append(T["help_ask"])
    elif cat == "report.complaint":
        parts.append(T["rc_body"])
        for d in cited:  # retrieval by the substance of the violation (a.topic), verified and anchor-checked
            quote(d)
        for t in terms or []:
            parts.append(term_sentence(t, lang))
        if claimed_irrelevant:
            parts.append(T["claim_note"].format(refs=", ".join(claimed_irrelevant)))
        if zone_hint == "ask_city":
            parts.append(T["ask_city"])
        parts.append(T["rc_ask"])
    elif cat == "report.account":
        parts.append(T["acc_body"])
        for d in cited:
            quote(d, "acc_rule")
        parts += [T["acc_tip"], T["acc_ask"]]
    elif cat == "report.payment":
        parts += [T["pay_body"], T["pay_tip"], T["pay_ask"]]
    else:  # report.offtopic
        parts.append(T["off_body"])
        for d in cited:
            quote(d, "off_rule")
        parts.append(T["off_ask"])

    # ---- sensitive topics (threat outside the game, hacking, lost access, payment) -------------------
    # They are never answered with the ordinary strategy alone, and never with a refusal for missing evidence: the text says
    # that a person has to review it, asks for the facts a person needs, and promises nothing.
    if cat not in {"report.account", "report.payment"}:
        crit: list[str] = []
        if "threat_outside_game" in a.flags:
            crit.append(T["crit_threat"])
        if a.flags & {"hack", "access"}:
            crit += [T["acc_body"], T["acc_tip"], T["acc_ask"]]
        if "payment" in a.flags:
            crit += [T["pay_body"], T["pay_tip"], T["pay_ask"]]
        if "appeal" in a.flags and cat != "complaint.admin":  # an appeal that arrived as a report or inside another category asks for the appeal data, not "what do you mean"
            crit += [T["admin_body"], T["admin_ask"]]
        if crit:
            # an invalid-complaint refusal ("это вопрос / нет доказательств") is replaced even when it quoted a complaint-rules point
            replace_body = cat in {"complaint.other", "complaint.invalid", "report.question", "report.offtopic", "report.help"} and (not cited or cat == "complaint.invalid")
            if replace_body:
                parts = [T["hello"]] + crit
                used = []
                if cat == "complaint.other" and (a.nickname or a.attachments):
                    parts.append(T["also_complaint"])
            else:
                parts.extend(crit)

    if is_duplicate and a.kind == "complaint":
        parts.append(T["dup"])
        if dup_rule is not None and dup_rule.key not in {u.key for u in used}:
            quote(dup_rule)
    return " ".join(parts), used
