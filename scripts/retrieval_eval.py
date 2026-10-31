"""Measure the rule retrieval -> docs/metrics/retrieval.json.

Silver ground truth: complaints whose administrator reply names a general-rule code («2.1 ОП», «5.2 ОП», ...) that exists
in rules.json (see scripts/_common.silver_category). Weak (admins make mistakes, and the reply is not an annotation of the
player's text), so this compares methods against each other rather than claiming absolute quality:
  * BM25 over all citable points, with the jargon expansion of app/rag.py;
  * BM25 WITHOUT the expansion (what a plain keyword retriever would do);
  * the system's actual selection (curated candidates + anchors + verified claims).
"""
from __future__ import annotations

import json
import random

from _common import DATA, METRICS, load_jsonl, silver_category
from app.adapter import record_to_request
from app.classify import analyze
from app.dedup import DedupIndex
from app.pipeline import TriageService
from app.rag import build_query, select_rules


def main(sample: int = 30000) -> None:
    svc = TriageService(dedup=DedupIndex())
    rb = svc.rules
    train = load_jsonl(DATA / "tickets_train.jsonl.gz")
    random.Random(11).shuffle(train)
    recs = load_jsonl(DATA / "tickets_valid.jsonl") + train[:sample]
    n = 0
    hit = {"bm25_expanded": [0, 0, 0], "bm25_plain": [0, 0, 0]}
    chosen_hit = chosen_cited = chosen_empty = 0
    for r in recs:
        cat, rule = silver_category(r)
        if not rule or rule.startswith("GZ"):
            continue
        gold = rb.resolve_cite(rule)
        if gold is None:
            continue
        a = analyze(record_to_request(r))
        if not a.clean.strip():
            continue
        n += 1
        for name, q in (("bm25_expanded", build_query(a)), ("bm25_plain", a.clean)):
            top = [d.key for d, _ in rb.search(q, top_k=5, citable_only=True)]
            for i, k in enumerate((1, 3, 5)):
                hit[name][i] += gold.key in top[:k]
        sel = select_rules(a, rb)
        if sel.cited:
            chosen_cited += 1
            chosen_hit += gold.key in {d.key for d in sel.cited}
        else:
            chosen_empty += 1
    out = {
        "note": "silver ground truth = rule named by the admin reply; weak; compares methods",
        "tickets_with_named_rule": n,
        "recall_at_1_3_5": {k: [round(x / max(n, 1), 3) for x in v] for k, v in hit.items()},
        "system_selection": {"cited_something": chosen_cited, "cited_equals_admin_rule": chosen_hit,
                             "precision_when_citing": round(chosen_hit / max(chosen_cited, 1), 3), "cited_nothing": chosen_empty,
                             "coverage": round(chosen_cited / max(n, 1), 3)},
    }
    (METRICS / "retrieval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
