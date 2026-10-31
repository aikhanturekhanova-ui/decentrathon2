"""Verified FAQ for recurring game questions ("как устроиться в полицию?").

The organizers gave rules and tickets, but no knowledge base about game mechanics, and answers must not be invented.
This module loads answers that people with real knowledge (admins) wrote, together with their provenance:

    {"entries": [{"id": "police-job", "patterns": ["устро\\w+\\s+в\\s+полици"], "reviewed": true,
                  "source": "who/when/where this answer comes from",
                  "answers": {"ru": "...", "kk": "...", "uz": "...", "en": "...", "es": "...", "pt": "..."}}]}

Rules enforced when loading (an entry that fails is skipped and listed in `Faq.problems`):
  * every answer passes the same output guard as any draft (no promises, no invented links/commands, no rule numbers);
  * `source` is mandatory; only `reviewed: true` entries may remove the human handoff;
  * an answer is used only in a language it is written in (no machine translation, no mixed languages).
The file lives in data/kb/faq.json (git-ignored, like all organizer-derived content).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .guard import check_draft
from .knowledge import RuleBook
from .phrases import P


@dataclass
class FaqEntry:
    id: str
    patterns: list[re.Pattern]
    answers: dict[str, str]
    source: str
    reviewed: bool


@dataclass
class Faq:
    entries: list[FaqEntry] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, rules: RuleBook) -> "Faq":
        faq = cls()
        if not Path(path).is_file():
            return faq
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            faq.problems.append(f"cannot read {path}: {exc}")
            return faq
        for e in raw.get("entries", []) if isinstance(raw, dict) else []:
            try:
                eid, src = str(e["id"]), str(e.get("source", "")).strip()
                if not src:
                    faq.problems.append(f"{eid}: missing 'source' (provenance is mandatory)")
                    continue
                answers = {k: str(v) for k, v in e["answers"].items() if k in P and str(v).strip()}
                bad = [k for k, v in answers.items() if not check_draft(v, [], rules).ok]
                if bad or not answers:
                    faq.problems.append(f"{eid}: answers rejected by the output guard or empty ({bad})")
                    continue
                faq.entries.append(FaqEntry(eid, [re.compile(p, re.I | re.UNICODE) for p in e["patterns"]], answers, src, bool(e.get("reviewed", False))))
            except (KeyError, TypeError, re.error) as exc:
                faq.problems.append(f"invalid entry {e!r:.60}: {type(exc).__name__}")
        return faq

    def match(self, clean_text: str, lang: str) -> Optional[tuple[FaqEntry, str]]:
        for e in self.entries:
            if lang in e.answers and any(p.search(clean_text) for p in e.patterns):
                return e, e.answers[lang]
        return None
