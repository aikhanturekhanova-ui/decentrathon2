"""Rule corpus (rules.json) + BM25 retrieval + citation resolving.

Facts about the real data that shape this module (checked against the provided files):
  * point refs repeat across sections ("2.1" exists in 'Общие правила игры' and elsewhere), so a
    point is addressed by (section, ref) and shown as "2.1", "GZ 1.1" or "ЖБ 1.2.7";
  * some points are headings (text ends with ':'), some sections have text but no points
    (RP terms) - they are indexed as context but never cited as rules.
The corpus is loaded from the organizers' file at runtime; its text is not stored in the repo.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Citation prefix per section id of rules.json. Only these three sections are citable in replies:
# they are what players' complaints are about. Others (leaders, OPG, wars...) stay searchable.
SECTION_PREFIX: dict[int, str] = {30: "", 31: "ЖБ", 32: "GZ"}
GENERAL_SECTION = 30
COMPLAINT_RULES_SECTION = 31
GREEN_ZONE_SECTION = 32

_URL = re.compile(r"https?://[^\s)>,;]+")


@dataclass(frozen=True)
class RuleDoc:
    section_id: int
    section_title: str
    ref: str
    body: str
    sanction: str = ""
    is_heading: bool = False

    @property
    def key(self) -> str:
        return f"{self.section_id}:{self.ref}"

    @property
    def citable(self) -> bool:
        return self.section_id in SECTION_PREFIX and not self.is_heading

    @property
    def cite(self) -> str:
        prefix = SECTION_PREFIX.get(self.section_id)
        if prefix is None:
            return f"{self.section_title} {self.ref}"
        return f"{prefix} {self.ref}".strip()


def _stem(word: str) -> str:
    return word[:5] if len(word) > 5 else word


_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    t = text.lower().replace("ё", "е")
    return [_stem(w) for w in _TOKEN.findall(t) if len(w) > 1]


@dataclass(frozen=True)
class TermDoc:
    """A game term defined in a rules section that has no numbered points (e.g. «RP термины»: DM, DB, SK, TK, MG, NonRP)."""
    abbr: str
    name: str
    definition: str
    sanction: str
    section_id: int
    section_title: str


_TERM_LINE = re.compile(r"^\s*[•\-*]\s*(?P<name>[^(]+?)\s*\(\s*сокращ\w*\s+(?P<abbr>[^)]+?)\s*\)\s*[—–-]\s*(?P<rest>.+)$")


def parse_terms(section_id: int, title: str, text: str) -> list[TermDoc]:
    out: list[TermDoc] = []
    for line in str(text).split("\n"):
        m = _TERM_LINE.match(line)
        if not m:
            continue
        definition, _, sanction = m.group("rest").partition(" | ")
        out.append(TermDoc(m.group("abbr").strip(), m.group("name").strip(), definition.strip(), sanction.strip(), section_id, title))
    return out


class RuleBook:
    def __init__(self, docs: list[RuleDoc], context_docs: list[RuleDoc] | None = None, terms: list[TermDoc] | None = None):
        self.docs = docs
        self.context_docs = context_docs or []
        self.terms: dict[str, TermDoc] = {t.abbr: t for t in (terms or [])}
        self._by_key = {d.key: d for d in docs}
        self._index_docs = docs + self.context_docs
        self._tf: list[Counter] = []
        self._len: list[int] = []
        df: dict[str, int] = defaultdict(int)
        for d in self._index_docs:
            toks = tokenize(d.body)
            c = Counter(toks)
            self._tf.append(c)
            self._len.append(len(toks) or 1)
            for term in c:
                df[term] += 1
        n = max(len(self._index_docs), 1)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._avg = sum(self._len) / n
        text_blob = " ".join(d.body for d in docs)
        self.urls = sorted(set(u.rstrip(".,") for u in _URL.findall(text_blob)))

    # ----- loading --------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "RuleBook":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        sections = raw["rules"] if isinstance(raw, dict) else raw
        docs: list[RuleDoc] = []
        context: list[RuleDoc] = []
        terms: list[TermDoc] = []
        for sec in sections:
            sid = int(sec["id"])
            title = str(sec.get("category", sid))
            points = sec.get("points") or []
            for p in points:
                text = str(p.get("text", "")).strip()
                body, _, sanction = text.partition(" | ")
                heading = body.rstrip().endswith(":") and len(body) < 90
                docs.append(RuleDoc(sid, title, str(p["ref"]).strip(), body.strip(), sanction.strip(), heading))
            if not points and sec.get("text"):
                terms.extend(parse_terms(sid, title, str(sec["text"])))
                # sections with useful text but no points (e.g. RP terms): one context doc per bullet
                for i, chunk in enumerate(s.strip() for s in re.split(r"\n+", str(sec["text"])) if s.strip()):
                    context.append(RuleDoc(sid, title, f"t{i + 1}", chunk.split(" | ")[0], "", False))
        return cls(docs, context, terms)

    # ----- lookup ---------------------------------------------------------------------------
    def get(self, section_id: int, ref: str) -> RuleDoc | None:
        return self._by_key.get(f"{section_id}:{ref}")

    def resolve_cite(self, cite: str) -> RuleDoc | None:
        """'2.1' -> general rule, 'GZ 1.1' -> green zone, 'ЖБ 1.2.7' -> complaint rules."""
        m = re.fullmatch(r"\s*(ЖБ|GZ)?\s*(\d+(?:\.\d+){0,3})\s*", cite or "", flags=re.I)
        if not m:
            return None
        prefix = (m.group(1) or "").upper()
        section = {"": GENERAL_SECTION, "ЖБ": COMPLAINT_RULES_SECTION, "GZ": GREEN_ZONE_SECTION}[prefix]
        doc = self.get(section, m.group(2))
        return doc if doc and doc.citable else None

    def all_cites(self) -> set[str]:
        return {d.cite for d in self.docs if d.citable}

    # ----- BM25 -----------------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, citable_only: bool = False) -> list[tuple[RuleDoc, float]]:
        q = Counter(tokenize(query))
        if not q:
            return []
        k1, b = 1.5, 0.75
        scored: list[tuple[RuleDoc, float]] = []
        for doc, tf, ln in zip(self._index_docs, self._tf, self._len):
            if citable_only and not doc.citable:
                continue
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self._idf.get(term, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / self._avg))
            if s > 0:
                scored.append((doc, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def score_of(self, doc: RuleDoc, query: str) -> float:
        for d, s in self.search(query, top_k=len(self._index_docs)):
            if d.key == doc.key:
                return s
        return 0.0


@dataclass
class KnowledgeBase:
    rules: RuleBook
    taxonomy: dict = field(default_factory=dict)
