"""Weak check of duplicate detection against the administrators' verdict ('duplicate') -> docs/metrics/dedup.json.

verdict == 'duplicate' means an admin answered "this complaint duplicates another one". It is NOT the labelling of
the hidden test (unknown) and admins may also close non-duplicates. We run train+valid chronologically through the same
DedupIndex that the batch mode uses (author = reporter_hash, target = reported_hash, 60-minute window) and compare.
"""
from __future__ import annotations

import json
from collections import Counter

from _common import DATA, METRICS, load_jsonl
from app.adapter import record_to_request
from app.classify import analyze
from app.dedup import DedupIndex, parse_ts, spam_reason


def main() -> None:
    recs = load_jsonl(DATA / "tickets_valid.jsonl") + load_jsonl(DATA / "tickets_train.jsonl.gz")
    recs.sort(key=lambda r: parse_ts(r.get("created_at")) or parse_ts("1970-01-01T00:00:00"))
    idx = DedupIndex()
    tp = fp = fn = tn = 0
    flagged_by_verdict: Counter = Counter()
    kinds = Counter()
    rel = Counter()  # (has related complaints from other authors, verdict == already_punished) over complaints with an admin verdict
    for r in recs:
        req = record_to_request(r)
        a = analyze(req)
        res = idx.check_and_add(req, a)
        flagged = bool(res.is_duplicate or spam_reason(a))
        if r["kind"] == "complaint" and r.get("verdict"):
            rel[(bool(res.related), r.get("verdict") == "already_punished")] += 1
        truth = r.get("verdict") == "duplicate"
        tp += flagged and truth
        fp += flagged and not truth
        fn += (not flagged) and truth
        tn += (not flagged) and not truth
        if flagged:
            flagged_by_verdict[r.get("verdict")] += 1
            kinds[r["kind"]] += 1
    out = {
        "note": "weak: truth = admin verdict 'duplicate' (complaints only; admins close duplicates inconsistently); not the hidden labelling",
        "tickets": len(recs), "verdict_duplicate": tp + fn, "flagged": tp + fp,
        "precision_vs_verdict": round(tp / max(tp + fp, 1), 4), "recall_vs_verdict": round(tp / max(tp + fn, 1), 4),
        "flagged_by_verdict": dict(flagged_by_verdict.most_common()), "flagged_by_kind": dict(kinds),
        "related_complaints": {
            "note": "complaints that have earlier complaints about the same accused player from OTHER authors within 24 h; "
                    "weak check: share of admin verdict 'already_punished' with and without such related complaints",
            "complaints_with_related": rel[(True, True)] + rel[(True, False)],
            "complaints_without_related": rel[(False, True)] + rel[(False, False)],
            "already_punished_rate_with_related": round(rel[(True, True)] / max(rel[(True, True)] + rel[(True, False)], 1), 4),
            "already_punished_rate_without_related": round(rel[(False, True)] / max(rel[(False, True)] + rel[(False, False)], 1), 4),
        },
    }
    (METRICS / "dedup.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
