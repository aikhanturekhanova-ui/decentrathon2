"""Weak ("silver") agreement check + hard-constraint statistics on real data -> docs/metrics/silver.json.

IMPORTANT: silver labels come from the administrator's reply (rule code / appeal verdict). They are
NOT human annotation of the tickets and the numbers below are NOT accuracy. They tell us whether the
classifier broadly agrees with what administrators did, and they feed the error breakdown.
The administrator's text is used only to derive the label for scoring; it never goes to the service.
"""
from __future__ import annotations

import collections
import json
import random

from _common import DATA, METRICS, macro_f1, load_jsonl, silver_category
from app.adapter import record_to_request
from app.dedup import DedupIndex
from app.guard import check_draft
from app.pipeline import TriageService


def run(records: list[dict], svc: TriageService):
    out = []
    for r in records:
        resp, dbg = svc.triage(record_to_request(r))
        out.append((r, resp, dbg))
    return out


def main(train_sample: int = 20000) -> None:
    svc = TriageService(dedup=DedupIndex())
    valid = load_jsonl(DATA / "tickets_valid.jsonl")
    train = load_jsonl(DATA / "tickets_train.jsonl.gz")
    random.Random(7).shuffle(train)
    sets = {"valid": valid, f"train_sample_{train_sample}": train[:train_sample]}
    report: dict = {"note": "silver = derived from administrator reply; NOT human labels; agreement, not accuracy",
                    "classifier": svc.classifier,
                    "caveat": ("with the learned model loaded, the TRAIN sample is IN-SAMPLE for the model (it was fitted on train silver labels), "
                               "so its agreement is optimistic; only the VALID numbers are out-of-sample"
                               if svc.model else "lexicon only")}
    for name, recs in sets.items():
        res = run(recs, svc)
        yt, yp = [], []
        by_lang = collections.defaultdict(lambda: [0, 0])
        confusion = collections.Counter()
        cite_checked = cite_agree = 0
        for r, resp, dbg in res:
            sc, rule = silver_category(r)
            if sc:
                yt.append(sc), yp.append(resp.category)
                by_lang[resp.lang][0] += int(sc == resp.category)
                by_lang[resp.lang][1] += 1
                if sc != resp.category:
                    confusion[(sc, resp.category)] += 1
                if rule and resp.draft_reply.cited and sc == resp.category:
                    cite_checked += 1
                    cite_agree += int(rule in resp.draft_reply.cited)
        n = len(res)
        guard_fail = sum(not check_draft(x.draft_reply.text, x.draft_reply.cited, svc.rules, x.needs_human).ok for _, x, _ in res)
        report[name] = {
            "tickets": n,
            "silver_labelled": len(yt),
            "silver_agreement_category": round(sum(a == b for a, b in zip(yt, yp)) / max(len(yt), 1), 4),
            "silver_macro_f1_category": round(macro_f1(yt, yp), 4),
            "agreement_by_detected_lang": {k: {"n": v[1], "agree": round(v[0] / v[1], 3)} for k, v in by_lang.items()},
            "top_disagreements(silver->ours)": [[a, b, c] for (a, b), c in confusion.most_common(8)],
            "cited_rule_equals_admin_rule_when_both_exist": {"checked": cite_checked, "equal": cite_agree},
            "needs_human_share": round(sum(x.needs_human for _, x, _ in res) / n, 4),
            "category_distribution": dict(collections.Counter(x.category for _, x, _ in res).most_common()),
            "priority_distribution": dict(collections.Counter(x.priority for _, x, _ in res).most_common()),
            "lang_distribution": dict(collections.Counter(x.lang for _, x, _ in res).most_common()),
            "draft_lang_distribution": dict(collections.Counter(x.draft_reply.lang for _, x, _ in res).most_common()),
            "with_citation_share": round(sum(bool(x.draft_reply.cited) for _, x, _ in res) / n, 4),
            "with_term_share": round(sum(bool(x.draft_reply.terms) for _, x, _ in res) / n, 4),
            "with_citation_or_term_share": round(sum(bool(x.draft_reply.cited or x.draft_reply.terms) for _, x, _ in res) / n, 4),
            "complaint_dm_grounded_share": round(
                sum(bool(x.draft_reply.cited or x.draft_reply.terms) for _, x, _ in res if x.category == "complaint.dm")
                / max(sum(x.category == "complaint.dm" for _, x, _ in res), 1), 4),
            "term_counts": dict(collections.Counter(t for _, x, _ in res for t in x.draft_reply.terms).most_common()),
            "cited_rule_counts": dict(collections.Counter(c for _, x, _ in res for c in x.draft_reply.cited).most_common()),
            "guard_failures_on_final_drafts": guard_fail,
            "template_fallbacks_due_to_guard": sum(bool(d.guard_notes) for _, _, d in res),
        }
    (METRICS / "silver.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:3500])


if __name__ == "__main__":
    main()
