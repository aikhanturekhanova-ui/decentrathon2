"""Score the service against HUMAN labels (the only place where Accuracy / macro-F1 are reported).

Refuses to print headline numbers unless enough rows carry a valid label and an annotator name.
Optional second annotator file -> Cohen's kappa. Output: aggregates only -> docs/metrics/human_eval.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys

from _common import DATA, METRICS, load_jsonl, macro_f1_report
from app.adapter import record_to_request
from app.dedup import DedupIndex
from app.pipeline import CATEGORY_CODES, PRIORITY_CODES, TriageService

MIN_ROWS = 50


def read(path: str) -> dict[str, dict]:
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["human_category"].strip() or r["human_priority"].strip():
                rows[r["ticket_id"]] = r
    return rows


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--labels2", help="second annotator, same ticket ids (for kappa)")
    ap.add_argument("--source", default=str(DATA / "tickets_valid.jsonl"))
    a = ap.parse_args()
    labels = read(a.labels)
    bad = [i for i, r in labels.items() if r["human_category"] not in CATEGORY_CODES or r["human_priority"] not in PRIORITY_CODES
           or not r["annotator"].strip()]
    if bad:
        print(f"{len(bad)} rows have an invalid/missing category, priority or annotator, e.g. {bad[:3]}. Fix them first.")
        return 2
    if len(labels) < MIN_ROWS:
        print(f"Only {len(labels)} labelled rows (< {MIN_ROWS}): not enough for a meaningful metric. Not reporting.")
        return 2
    recs = {r["ticket_id"]: r for r in load_jsonl(a.source)}
    svc = TriageService(dedup=DedupIndex())
    yt_c, yp_c, yt_p, yp_p, langs = [], [], [], [], []
    conf = collections.Counter()
    by_cat = collections.defaultdict(lambda: [0, 0])
    by_lang = collections.defaultdict(lambda: [0, 0])
    for tid, row in labels.items():
        resp, _ = svc.triage(record_to_request(recs[tid]))
        yt_c.append(row["human_category"]), yp_c.append(resp.category)
        yt_p.append(row["human_priority"]), yp_p.append(resp.priority)
        hl = row["human_lang"].strip() or resp.lang
        langs.append(hl)
        ok = resp.category == row["human_category"]
        by_cat[row["human_category"]][0] += ok
        by_cat[row["human_category"]][1] += 1
        by_lang[hl][0] += ok
        by_lang[hl][1] += 1
        if not ok:
            conf[(row["human_category"], resp.category)] += 1
    n = len(labels)
    out = {
        "labelled_rows": n, "annotators": sorted({r["annotator"] for r in labels.values()}),
        "category_accuracy": round(sum(x == y for x, y in zip(yt_c, yp_c)) / n, 4),
        # macro-F1 over the classes that occur in the gold labels OR in our predictions (scikit-learn convention, zero_division=0);
        # the gold-only variant, the classes used by one side only and the per-class table are reported next to it
        "priority": macro_f1_report(yt_p, yp_p),
        "category": macro_f1_report(yt_c, yp_c),
        "priority_macro_f1": macro_f1_report(yt_p, yp_p)["macro_f1"],
        "category_macro_f1": macro_f1_report(yt_c, yp_c)["macro_f1"],
        "methodology": "docs/ARCHITECTURE.md#evaluation",
        "priority_accuracy": round(sum(x == y for x, y in zip(yt_p, yp_p)) / n, 4),
        "sensitive_topics": "see docs/metrics/coverage.json (mandatory-handoff counters are checked separately, they are not an average)",
        "recall_by_true_category": {k: {"n": v[1], "recall": round(v[0] / v[1], 3)} for k, v in sorted(by_cat.items())},
        "accuracy_by_language": {k: {"n": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in sorted(by_lang.items())},
        "top_confusions(human->ours)": [[x, y, c] for (x, y), c in conf.most_common(10)],
    }
    if a.labels2:
        other = read(a.labels2)
        common = sorted(set(labels) & set(other))
        if len(common) >= MIN_ROWS:
            out["inter_annotator"] = {
                "common_rows": len(common),
                "kappa_category": round(kappa([labels[i]["human_category"] for i in common], [other[i]["human_category"] for i in common]), 3),
                "kappa_priority": round(kappa([labels[i]["human_priority"] for i in common], [other[i]["human_priority"] for i in common]), 3),
            }
    (METRICS / "human_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
