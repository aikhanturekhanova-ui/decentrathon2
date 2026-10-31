"""Single entry point used by the HTTP API, the batch CLI, the admin panel and the Telegram bot."""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .classify import NEEDS_HUMAN_ALWAYS, Analysis, analyze
from .config import ROOT, Settings, load_settings
from .dedup import DedupIndex, related_reason, spam_reason
from .drafts import compose, reply_lang, rule_sentence, term_sentence
from .faq import Faq
from .guard import RULE_TALK, check_draft, check_free_text, check_llm_sentences
from .knowledge import RuleBook
from .langseg import LANG_RU, profile
from .llm import LLMClient
from .ml import ComplaintModel
from .rag import _anchored, select_rules
from .schemas import DraftReply, Extracted, TriageDebug, TriageRequest, TriageResponse
from .textutil import detect_lang, to_game_chars

CATEGORY_CODES = {
    "complaint.insult", "complaint.cheat", "complaint.dm", "complaint.scam", "complaint.rmt_ads", "complaint.admin",
    "complaint.other", "complaint.invalid", "report.question", "report.help", "report.complaint", "report.account",
    "report.payment", "report.offtopic",
}
PRIORITY_CODES = {"p0", "p1", "p2", "p3"}


@dataclass
class Taxonomy:
    categories: set[str]
    priorities: set[str]
    needs_human_always: set[str]
    source: str

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                cats = {c["code"] for c in raw.get("categories", [])}
                pris = {p["code"] for p in raw.get("priorities", [])}
                if cats and pris:
                    return cls(cats, pris, set(raw.get("needs_human_always", [])) | NEEDS_HUMAN_ALWAYS, str(path))
            except (ValueError, KeyError, TypeError, OSError):
                pass
            return cls(set(CATEGORY_CODES), set(PRIORITY_CODES), set(NEEDS_HUMAN_ALWAYS), f"built-in ({path} is unreadable)")
        return cls(set(CATEGORY_CODES), set(PRIORITY_CODES), set(NEEDS_HUMAN_ALWAYS), "built-in (taxonomy.json not found)")

    def fit(self, category: str, priority: str, kind: str) -> tuple[str, str, list[str]]:
        """If the organizers' final taxonomy renamed/removed a code, never crash and never emit a code outside it."""
        notes: list[str] = []
        if category not in self.categories:
            same = sorted(c for c in self.categories if c.startswith(kind + "."))
            pick = next((c for c in (f"{kind}.other", f"{kind}.question", f"{kind}.offtopic") if c in self.categories), same[0] if same else category)
            notes.append(f"Категория «{category}» отсутствует в таксономии; использована «{pick}»")
            category = pick
        if priority not in self.priorities:
            pick = "p2" if "p2" in self.priorities else sorted(self.priorities)[0]
            notes.append(f"Приоритет «{priority}» отсутствует в таксономии; использован «{pick}»")
            priority = pick
        return category, priority, notes


HUMAN_REASONS = {
    "report.account": "Взлом, доступ к аккаунту или пропажа имущества: такие обращения всегда передаются сотруднику",
    "report.payment": "Платёж или донат: такие обращения всегда передаются сотруднику",
    "complaint.admin": "Оспаривание наказания или жалоба на администратора: решение принимает человек",
    "report.help": "Нужно действие администратора в игре; ассистент не может ничего выдать, телепортировать или починить",
    "report.complaint": "Жалоба на игрока через репорт: проверку и решение выполняет администратор",
    "report.question": "Ответа на вопрос нет в доступных правилах и базе знаний; ассистент не отвечает наугад",
}


_MARKER = re.compile(r"\[\[R(\d+)\]\]")


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _norm_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def template_sentences(template: str) -> set[str]:
    """Sentences of the verified reference template (markers removed): the model may repeat them word for word."""
    return {n for n in (_norm_sentence(x) for x in _SENT_SPLIT.split(_MARKER.sub(" ", template))) if n}


def merge_llm_text(raw: str, docs: list, lang: str, extras: Optional[list[str]] = None,
                   allowed_numbers: Optional[set] = None, verified: Optional[set[str]] = None,
                   extra_words: tuple[str, ...] = ()) -> tuple[Optional[str], Optional[str]]:
    """Turn the model's text into the final draft, or explain why it is rejected.

    The model never writes rule sentences: it places markers, and the verified paraphrase of the rule
    (phrases.GISTS, anchor-checked against rules.json) is inserted here. So a model cannot invert the meaning of a rule,
    cite a rule that was not selected, or invent one. The language of the model text must match the draft language.
    """
    markers = [int(x) for x in _MARKER.findall(raw)]
    if any(i < 1 or i > len(docs) for i in markers):
        return None, "model used an unknown rule marker"
    bare = _MARKER.sub(" ", raw)
    if RULE_TALK.search(bare):
        return None, "model wrote about rules itself (only markers are allowed)"
    detected = detect_lang(bare)
    if detected not in {lang, "und"} and not (lang == "ru" and detected == "lat"):
        return None, f"language mismatch: model wrote {detected}, draft language is {lang}"
    # Sentences repeated WORD FOR WORD from the verified template are our own text (it already passed check_draft); the structural and
    # sentence-type checks apply to what the model wrote itself. Without this, the template's own safe sentences («Заранее сообщить результат
    # или наказание мы не можем.», «Вложений в обращении: 1; …») were rejected in 12 of 14 real Gemini drafts on the demo tickets.
    own = " ".join(x for x in _SENT_SPLIT.split(bare) if _norm_sentence(x) and _norm_sentence(x) not in (verified or set()))
    # structural check of what the MODEL wrote itself (sanction/refund topics must be negated, no deadlines, no invented numbers)
    free = check_free_text(own, allowed_numbers)
    if free:
        return None, "free text: " + "; ".join(free[:3])
    off_type = check_llm_sentences(own, lang, extra_words)  # closed vocabulary + harmless sentence types + no secrets (guard.check_llm_sentences)
    if off_type:
        return None, "sentence outside the allowed types: " + "; ".join(off_type[:2])
    used: set[int] = set()

    def sub(m: re.Match) -> str:
        i = int(m.group(1))
        if i in used:
            return ""
        used.add(i)
        return rule_sentence(docs[i - 1], lang)

    text = _MARKER.sub(sub, raw)
    missing = [i for i in range(1, len(docs) + 1) if i not in used]
    inserts = [rule_sentence(docs[i - 1], lang) for i in missing] + list(extras or [])  # + verified game-term sentences
    if inserts:  # every selected rule/term must appear: insert the verified sentences after the first sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences[1:1] = inserts
        text = " ".join(sentences)
    text = re.sub(r"\.\s*\.", ".", re.sub(r"\s{2,}", " ", text)).strip()
    return text, None


def template_with_markers(text: str, docs: list) -> str:
    """The reference template shown to the model: sentences that quote a rule become [[Rn]] (no citable numbers leak)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for i, d in enumerate(docs, 1):
        pat = re.compile(rf"(?<![\d.]){re.escape(d.ref)}(?![\d.])")
        for k, s in enumerate(sentences):
            if pat.search(s) and not s.startswith("[["):
                sentences[k] = f"[[R{i}]]"
                break
    # The model may not write about rules itself (guard.RULE_TALK), so the reference it imitates must not either: a lead-in such as
    # «Что говорят правила: п.» is dropped and a sentence that mixes a marker with rule words is reduced to the marker. Otherwise the
    # model copies the lead-in and every such draft is rejected (measured with Gemini on the demo tickets).
    cleaned = []
    for s in sentences:
        if RULE_TALK.search(_MARKER.sub(" ", s)):
            markers = _MARKER.findall(s)
            if markers:
                cleaned.append(" ".join(f"[[R{m}]]" for m in markers))
            continue
        cleaned.append(s)
    return " ".join(cleaned)


def find_data_file(name: str, preferred: Path, hint_dirs: Optional[list[Path]] = None) -> Optional[Path]:
    """Locate an organizers' file. The repository must not contain them, so a judge's machine may keep them anywhere
    near the tickets: we look in the configured path, next to the input file, in ./handout, ./data and the working directory."""
    cands: list[Path] = [preferred]
    for d in hint_dirs or []:
        d = Path(d)
        cands += [d / name, d / "handout" / name, d.parent / "handout" / name, d / "data" / "handout" / name]
    cands += [ROOT / name, ROOT / "handout" / name, ROOT / "data" / name, ROOT / "data" / "handout" / name,
              Path.cwd() / name, Path.cwd() / "handout" / name, Path.cwd() / "data" / "handout" / name]
    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


class TriageService:
    def __init__(self, settings: Optional[Settings] = None, rules: Optional[RuleBook] = None,
                 llm: Optional[LLMClient] = None, dedup: Optional[DedupIndex] = None, hint_dirs: Optional[list[Path]] = None,
                 model: Optional[ComplaintModel] = None):
        self.settings = settings or load_settings()
        # learned complaint classifier (optional): an explicit model wins, otherwise the file from `make train` if enabled
        self.model = model if model is not None else (ComplaintModel.load(self.settings.model_path) if self.settings.use_model else None)
        self.rules_source: Optional[str] = "provided" if rules is not None else None
        if rules is None:
            found = find_data_file("rules.json", self.settings.rules_path, hint_dirs)
            if found is not None:
                rules = RuleBook.load(found)
                self.rules_source = str(found)
            else:
                # Do not crash a batch run: degrade loudly. Without rules no rule can be cited, so no rule is cited
                # (nothing is invented); every draft says so in `reasons`.
                print(f"[WARNING] rules.json not found (looked at {self.settings.rules_path} and next to the input). "
                      "Running WITHOUT rules: citations are disabled. Put rules.json into data/handout/.", file=sys.stderr)
                rules = RuleBook([])
        self.rules = rules
        tax_path = find_data_file("taxonomy.json", self.settings.taxonomy_path,
                                  [Path(self.rules_source).parent] if self.rules_source and self.rules_source != "provided" else hint_dirs)
        self.taxonomy = Taxonomy.load(tax_path or self.settings.taxonomy_path)
        self.faq = Faq.load(self.settings.faq_path, self.rules)
        self.dedup = dedup if dedup is not None else DedupIndex()
        self.llm = llm if llm is not None else (LLMClient(self.settings) if self.settings.llm_enabled else None)

    @property
    def mode(self) -> str:
        if self.llm:
            return f"llm+template-fallback ({self.settings.llm_provider}: {self.settings.llm_model})"
        return "template (demo: no LLM key, no model calls)"

    @property
    def classifier(self) -> str:
        return "hybrid: lexicon + learned model" if self.model else "lexicon only (no trained model: run `make train`)"

    # ------------------------------------------------------------------------------------------
    def triage(self, req: TriageRequest) -> tuple[TriageResponse, TriageDebug]:
        t0 = time.perf_counter()
        dbg = TriageDebug(mode="template")
        a = analyze(req, self.model)
        lang = detect_lang(a.text, req.server_lang)

        reasons: list[str] = list(a.reasons)
        dup = self.dedup.check_and_add(req, a)
        spam = spam_reason(a)
        is_spam_or_dup = bool(dup.is_duplicate or spam)
        if dup.reason:  # a confident duplicate AND a mere suspicion are both explained; only the first sets is_spam_or_duplicate
            reasons.append(dup.reason)
        if spam:
            reasons.append(spam)
        rel = related_reason(dup.related)
        if rel:
            reasons.append(rel)
        langs = profile(a.raw_text, req.server_lang)
        if langs.mixed:
            reasons.append("Сообщение на нескольких языках: " + ", ".join(LANG_RU.get(k, k) for k in langs.languages if k != "und"))
        if a.toxicity and a.toxicity.has_abuse:
            reasons.extend(f"Модерация: {x}" for x in a.toxicity.explanation[:2])

        rag = select_rules(a, self.rules)
        if "rules_missing" in rag.notes:
            reasons.append("База правил (rules.json) не найдена: цитирование отключено, ответ дан без ссылок на правила")
        for ref in rag.claimed_unknown:
            reasons.append(f"Указанного игроком пункта «{ref}» нет в правилах; в ответе он не цитируется")
        if rag.claimed_irrelevant:
            reasons.append("Указанные игроком пункты не соответствуют описанной ситуации: " + ", ".join(rag.claimed_irrelevant))
        if rag.claimed_ok:
            reasons.append("Номер пункта игрока подтверждён сопоставлением с текстом правил: " + ", ".join(rag.claimed_ok))

        dup_rule = _anchored(self.rules, "ЖБ 1.2.9") if (dup.is_duplicate and a.kind == "complaint") else None
        rl = reply_lang(lang, req.server_lang)
        text, used = compose(a, rag.cited, rag.claimed_irrelevant, rl, self.rules, dup.is_duplicate, dup_rule, terms=rag.terms, zone_hint=rag.zone_hint)
        cited = [d.cite for d in used]
        used_terms = [t for t in rag.terms if term_sentence(t, rl) in text]  # only the terms the draft really names
        term_codes = [t.abbr for t in used_terms]

        answered_from_rule = a.category == "report.question" and (bool(cited) or bool(term_codes))
        faq_hit = None
        if a.category == "report.question" and not cited and not term_codes and "injection" not in a.flags and not a.sensitive:
            faq_hit = self.faq.match(a.clean, rl)
            if faq_hit:  # a reviewed, sourced answer written by people; used only in the language it was written in
                from .phrases import P as _P

                text = f"{_P[rl]['hello']} {faq_hit[1]} {_P[rl]['q_rule_ask']}"
        needs_human, nh_reasons = self._needs_human(a, dup.is_duplicate, spam, answered_from_rule or bool(faq_hit and faq_hit[0].reviewed))
        if faq_hit:
            nh_reasons.append(f"Ответ взят из базы знаний: «{faq_hit[0].id}», источник: {faq_hit[0].source}; проверено человеком: {'да' if faq_hit[0].reviewed else 'нет'}")
            if not faq_hit[0].reviewed:
                needs_human = True
        reasons.extend(nh_reasons)
        if rl in {"kk", "uz"}:
            reasons.append(f"Черновик на языке «{rl}» собран из шаблонов и не проверен носителем языка")

        # ---- optional LLM rewrite, strictly validated, template is the fallback ---------------
        # Sensitive tickets (payment, hacking, lost access, appeal, threat, mass failure) never reach the model: their text is limited to
        # the verified wordings. Free generation is allowed only for ordinary tickets, behind the structural + pattern checks.
        sensitive = bool(a.sensitive) or a.category in {"report.account", "report.payment", "complaint.admin"}
        if self.llm and sensitive:
            dbg.guard_notes.append("LLM не вызывался: чувствительная тема, используются только проверенные формулировки")
        remaining = self.settings.llm_total_budget_s - (time.perf_counter() - t0)
        if self.llm and not sensitive and not spam and "injection" not in a.flags and remaining < 0.3:
            dbg.llm_error = "no time budget left for the model call"
            dbg.guard_notes.append("LLM не вызывался: бюджет времени на обращение исчерпан; использован шаблон")
        elif self.llm and not sensitive and not spam and "injection" not in a.flags:
            docs = list(used)
            reference = template_with_markers(text, docs)
            llm_res = self.llm.draft(a, docs, rl, needs_human, reference, time_budget_s=remaining)
            dbg.llm_calls = llm_res.calls
            dbg.llm_requests_sent = llm_res.sent
            if llm_res.note:
                dbg.guard_notes.append(llm_res.note)
            dbg.cost_estimate_max_usd = llm_res.cost_estimate_max_usd
            dbg.llm_tokens_in, dbg.llm_tokens_out = llm_res.tokens_in, llm_res.tokens_out
            dbg.cost_usd = self.llm.cost_usd(llm_res)
            if not llm_res.text:
                dbg.llm_error = llm_res.error or "empty model answer"
            else:
                allowed = {d for d in re.findall(r"\d+", a.raw_text + " " + (a.nickname or "") + " " + str(a.server or ""))} | {str(a.attachments)}
                merged, why = merge_llm_text(llm_res.text, docs, rl, extras=[term_sentence(t, rl) for t in used_terms], allowed_numbers=allowed,
                                             verified=template_sentences(reference),
                                             extra_words=tuple(w.lower() for w in re.findall(r"[^\W\d_]+", a.nickname or "")))
                if merged is None:
                    dbg.llm_error = why
                else:
                    new_cited = [d.cite for d in docs]
                    g = check_draft(merged, new_cited, self.rules, needs_human, terms=term_codes)
                    if g.ok:
                        text, cited, dbg.mode = merged, new_cited, "llm"
                    else:
                        dbg.llm_error = "guard: " + "; ".join(g.problems[:3])
            if dbg.llm_error:
                dbg.guard_notes.append(f"LLM-черновик отклонён ({dbg.llm_error}); использован шаблон")
                reasons.append(f"Черновик модели отклонён автоматической проверкой ({dbg.llm_error[:140]}); использован проверенный шаблон")

        final = check_draft(text, cited, self.rules, needs_human, terms=term_codes)
        if not final.ok:  # must never happen for templates; if it does, ship the safest possible text
            dbg.guard_notes.extend(final.problems)
            text, cited, term_codes = self._safe_fallback(rl), [], []
            needs_human = True
            reasons.append("Автоматическая проверка отклонила черновик; подставлен нейтральный текст, нужен человек")

        if needs_human is False and not reasons:
            reasons.append("Сообщение без содержания или не по теме: достаточно уточняющего ответа без решений и обещаний")

        extracted = Extracted(
            nickname=a.nickname,
            rule_refs=[r for r in a.claimed_refs if self.rules.resolve_cite(r)],
            has_evidence=a.has_evidence,
            server=a.server,
            attachments=a.attachments,
            evidence_verified=False,
            rule_refs_unverified=rag.claimed_unknown,
            evidence_status=a.evidence_status,
        )
        if rl == "kk" and self.settings.kk_output_game_symbols:
            text = to_game_chars(text)  # typed the way the game (cp1251) expects; checks above ran on normal Kazakh
        out_lang = lang if lang in {"ru", "kk", "uz", "en", "es", "pt", "lat", "und"} else "und"
        if self.settings.lang_strict_contract:  # PDF contract lists only ru|kk|uz|en|und
            out_lang = {"lat": "ru", "es": "und", "pt": "und"}.get(out_lang, out_lang)
        category, priority, tax_notes = self.taxonomy.fit(a.category, a.priority, a.kind)
        reasons.extend(tax_notes)
        resp = TriageResponse(
            lang=out_lang,  # type: ignore[arg-type]
            category=category,
            priority=priority,
            is_spam_or_duplicate=is_spam_or_dup,
            extracted=extracted,
            draft_reply=DraftReply(lang=rl, text=text, cited=cited, terms=term_codes),
            needs_human=needs_human,
            confidence=round(float(a.confidence), 2),
            reasons=reasons,
        )
        dbg.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        dbg.signals = {"flags": sorted(a.flags), "matched": a.matched, "scores": {k: v for k, v in a.scores.items() if v},
                       "gamechars_fixed": a.gamechars_fixed, "suspected": a.suspected_category,
                       "model_proba": {k: round(v, 3) for k, v in (a.ml_proba or {}).items()},
                       "duplicate_similarity": dup.similarity, "related_complaints": dup.related,
                       "languages": langs.as_dict(), "toxicity": a.toxicity.as_dict() if a.toxicity else None}
        dbg.retrieved = rag.retrieved
        return resp, dbg

    # ------------------------------------------------------------------------------------------
    def _needs_human(self, a: Analysis, is_dup: bool, spam: Optional[str], answered_from_rule: bool = False) -> tuple[bool, list[str]]:
        cat = a.category
        reasons: list[str] = []
        nh = False
        # Mandatory topics are checked on their own signals, NOT only through the final category:
        # an appeal without attachments, or a payment/hack mentioned inside another kind of ticket, still goes to a person.
        for flag, why in (("appeal", "Признаки апелляции наказания или спора с решением администратора"),
                          ("hack", "Признаки взлома аккаунта"),
                          ("access", "Признаки потери доступа к аккаунту"),
                          ("payment", "Платёж, донат или просьба вернуть деньги")):
            if flag in a.flags:
                nh = True
                reasons.append(f"{why}: такие обращения всегда передаются человеку независимо от категории и вложений")
        if cat in self.taxonomy.needs_human_always:
            nh = True
            reasons.append(HUMAN_REASONS.get(cat, "Категория всегда передаётся человеку"))
        elif cat.startswith("complaint.") and cat != "complaint.invalid":
            nh = True
            reasons.append(
                f"Решение о нарушении и наказании принимает администратор; вложения (число: {a.attachments}) автоматически не проверены"
            )
        elif cat in HUMAN_REASONS and not (cat == "report.question" and answered_from_rule):
            nh = True
            reasons.append(HUMAN_REASONS[cat])
        elif answered_from_rule:
            reasons.append("Ответ основан на проверенном источнике (пункт правил или проверенная база знаний) и не содержит решений по конкретному случаю")
        if "injection" in a.flags:
            nh = True
            reasons.append("В тексте обращения есть попытка повлиять на инструкции ассистента; текст обработан как данные")
        if "threat_outside_game" in a.flags:
            nh = True
            reasons.append("Возможна угроза вне игры: нужен человек")
        if "mass_bug" in a.flags:
            nh = True
            reasons.append("Возможен массовый сбой: нужен человек")
        if a.confidence < self.settings.human_conf_min and not nh:
            nh = True
            reasons.append(f"Низкая уверенность классификации ({a.confidence:.2f})")
        elif a.confidence < self.settings.human_ambiguous_min and cat != "report.offtopic" and not nh:
            nh = True
            reasons.append(f"Неоднозначное обращение, уверенность {a.confidence:.2f}")
        if "claim_only" in a.flags:
            nh = True
        if cat == "complaint.invalid" and a.suspected_category in {"complaint.cheat", "complaint.scam", "complaint.rmt_ads"}:
            nh = True
            reasons.append("Жалоба без доказательств, но описывает серьёзное нарушение: стоит просмотреть")
        return nh, reasons

    def _assert_taxonomy(self, r: TriageResponse) -> None:
        if r.category not in self.taxonomy.categories:
            raise ValueError(f"category {r.category} is not in taxonomy ({self.taxonomy.source})")
        if r.priority not in self.taxonomy.priorities:
            raise ValueError(f"priority {r.priority} is not in taxonomy")

    @staticmethod
    def _safe_fallback(lang: str) -> str:
        from .phrases import P

        T = P.get(lang, P["ru"])
        return f"{T['hello']} {T['off_body']} {T['off_ask']}"
