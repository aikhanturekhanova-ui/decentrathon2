"""Score the service against an INDEPENDENT AI-ASSISTED review (NOT human annotation).

`data/local/ai_labels_dev.json` (git-ignored) holds labels assigned by the AI assistant that built this repo
by reading ticket texts of data/local/review_dev.csv WITHOUT looking at the service's output; ambiguous rows are
left out. It is a second opinion used to find classifier bugs on the TRAIN-derived dev sample. It is biased
(same author as the lexicon) and small, so it is NOT an accuracy estimate and never replaces human labels.
Output: aggregates -> docs/metrics/ai_review.json.
"""
from __future__ import annotations

import collections
import csv
import json
import sys

from _common import DATA, LOCAL, METRICS, macro_f1, load_jsonl
from app.adapter import record_to_request
from app.dedup import DedupIndex
from app.pipeline import TriageService

CAT = {"ins": "insult", "cheat": "cheat", "dm": "dm", "scam": "scam", "scam1": "scam", "rmt": "rmt_ads", "adm": "admin", "oth": "other", "inv": "invalid",
       "q": "question", "help": "help", "rc": "complaint", "acc": "account", "pay": "payment", "off": "offtopic"}
PRI = {"ins": "p2", "cheat": "p1", "dm": "p2", "scam": "p2", "scam1": "p1", "rmt": "p1", "adm": "p2", "oth": "p2", "inv": "p3", "q": "p3", "help": "p2",
       "rc": "p2", "acc": "p0", "pay": "p0", "off": "p3"}


def main() -> int:
    labels = {int(k): v for k, v in json.loads((LOCAL / "ai_labels_dev.json").read_text(encoding="utf-8")).items()}
    rows = list(csv.DictReader(open(LOCAL / "review_dev.csv", encoding="utf-8-sig")))
    train = {r["ticket_id"]: r for r in load_jsonl(DATA / "tickets_train.jsonl.gz")}
    svc = TriageService(dedup=DedupIndex())
    yt, yp, tp, pp = [], [], [], []
    bad = []
    by_lang = collections.defaultdict(lambda: [0, 0])
    for i, code in sorted(labels.items()):
        r = rows[i]
        kind = r["kind"]
        truth = f"{kind}.{CAT[code]}"
        resp, _ = svc.triage(record_to_request(train[r["ticket_id"]]))
        yt.append(truth), yp.append(resp.category), tp.append(PRI[code]), pp.append(resp.priority)
        ok = truth == resp.category
        by_lang[resp.lang][0] += ok
        by_lang[resp.lang][1] += 1
        if not ok or PRI[code] != resp.priority:
            bad.append((i, truth, PRI[code], resp.category, resp.priority, r["text"][:70].replace("\n", " ")))
    n = len(yt)
    out = {
        "labelled_by": "AI assistant (NOT humans), blind to the service output, ambiguous rows excluded", "rows": n,
        "category_accuracy": round(sum(a == b for a, b in zip(yt, yp)) / n, 4), "priority_macro_f1": round(macro_f1(tp, pp), 4),
        "priority_accuracy": round(sum(a == b for a, b in zip(tp, pp)) / n, 4),
        "accuracy_by_detected_lang": {k: {"n": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in by_lang.items()},
        "caveat": "small, biased, AI-labelled; use only to find bugs",
    }
    (METRICS / "ai_review.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    for b in bad:
        print(b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
