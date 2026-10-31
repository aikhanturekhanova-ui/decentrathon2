"""Bonus: suggest new reply templates from clusters of similar tickets.

Greedy single-pass clustering inside one category: a ticket joins the first cluster whose token set has
Jaccard >= 0.5 with the cluster's seed, otherwise it starts a new cluster. For each cluster of at least
`min_size` tickets we propose a template:
  * if administrators edited drafts of cluster members, the MOST COMMON edited text wins (source "admin edits");
  * otherwise the draft of the cluster's first ticket is proposed (source "auto draft").
A suggestion is only a proposal for a human to review; nothing is applied automatically and nothing is sent.
Limits: lexical similarity only (no embeddings), nicknames are masked so that "X оск родни" and "Y оск родни" fall together.
"""
from __future__ import annotations

import collections
import re
from typing import Any

from .textutil import clean_for_matching

_NICK = re.compile(r"[a-z0-9а-я]+_[a-z0-9а-я_]+", re.I)
JACCARD = 0.5


def signature(text: str) -> frozenset[str]:
    t = _NICK.sub(" ", clean_for_matching(text))
    return frozenset(w for w in t.split() if len(w) > 1 or w.isdigit())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def suggest(tickets: list[dict[str, Any]], min_size: int = 3, limit: int = 20) -> list[dict[str, Any]]:
    """`tickets`: dicts from Store.get(): category, request, response, final_text (edited text or None)."""
    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for t in tickets:
        by_cat[t["category"]].append(t)
    out: list[dict[str, Any]] = []
    for cat, items in by_cat.items():
        clusters: list[dict[str, Any]] = []
        for t in items:
            text = next((m.get("text", "") for m in t["request"]["messages"] if m.get("role") == "player"), "")
            sig = signature(text)
            for c in clusters:
                if _jaccard(sig, c["seed"]) >= JACCARD:
                    c["members"].append((t, text))
                    break
            else:
                clusters.append({"seed": sig, "members": [(t, text)]})
        for c in clusters:
            if len(c["members"]) < min_size:
                continue
            edited = [t["final_text"] for t, _ in c["members"] if t.get("final_text") and t["final_text"] != t["response"]["draft_reply"]["text"]]
            if edited:
                text, n = collections.Counter(edited).most_common(1)[0]
                source, support = "admin edits", n
            else:
                text, source, support = c["members"][0][0]["response"]["draft_reply"]["text"], "auto draft", 0
            out.append({
                "category": cat, "size": len(c["members"]), "source": source, "edit_support": support,
                "examples": [m[1][:80] for m in c["members"][:3]], "suggested_template": text,
                "ticket_ids": [m[0]["id"] for m in c["members"]][:20],
            })
    out.sort(key=lambda x: (-x["size"], -x["edit_support"]))
    return out[:limit]
