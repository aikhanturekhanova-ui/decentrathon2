"""Historical ticket intelligence from TRAIN -> docs/metrics/history.json (AGGREGATED numbers only, shown in the admin dashboard).

What administrators did in the past is a pattern, not a rule: these tables describe the history (e.g. how often a category ended in
«need evidence»), they are never used as ground truth or to decide a ticket. Strings (punishments, violation codes) are published only
when they occur at least MIN_COUNT times, so no individual ticket text can leak through the file.
"""
from __future__ import annotations

import collections
import json
import random
import re
import statistics

from _common import DATA, METRICS, load_jsonl
from app.adapter import record_to_request
from app.config import load_settings
from app.ml import ComplaintModel
from app.pipeline import TriageService
from app.dedup import DedupIndex

MIN_COUNT = 50
SAMPLE = 15000


def top(counter: collections.Counter, k: int = 10) -> list[list]:
    return [[key, n] for key, n in counter.most_common() if n >= MIN_COUNT and key][:k]


def norm_punishment(p: str) -> str:
    p = re.sub(r"\s+", " ", p.lower()).strip(" .")
    return p[:60]


def main() -> None:
    recs = load_jsonl(DATA / "tickets_train.jsonl.gz")
    comps = [r for r in recs if r["kind"] == "complaint"]
    reports = [r for r in recs if r["kind"] == "report"]
    out: dict = {"source": "tickets_train.jsonl.gz (TRAIN only)", "min_count_for_strings": MIN_COUNT,
                 "note": "historical administrator decisions: a pattern, not ground truth"}
    out["tickets"] = {"total": len(recs), "complaints": len(comps), "reports": len(reports)}
    out["dataset_lang"] = dict(collections.Counter(r.get("lang") or "und" for r in recs).most_common())
    verdicts = collections.Counter(r.get("verdict") or "none" for r in comps)
    out["complaint_verdicts"] = dict(verdicts.most_common())
    out["top_punishments"] = top(collections.Counter(norm_punishment(r["punishment"]) for r in comps if r.get("punishment")))
    out["images_per_complaint_median"] = statistics.median(r.get("images_count") or 0 for r in comps)
    per_rep = collections.Counter(r["reporter_hash"] for r in comps)
    per_acc = collections.Counter(r.get("reported_hash") for r in comps if r.get("reported_hash"))
    out["repeat_behaviour"] = {
        "reporters_with_2plus_complaints_share": round(sum(n >= 2 for n in per_rep.values()) / max(len(per_rep), 1), 4),
        "accused_with_3plus_complaints_share": round(sum(n >= 3 for n in per_acc.values()) / max(len(per_acc), 1), 4),
        "complaints_about_accused_with_3plus": round(sum(n for n in per_acc.values() if n >= 3) / max(len(comps), 1), 4),
        "note": "train holds all tickets of a random 20 % of reporters, so accused players' counts are undercounted",
    }

    # how past complaints ended, by the category our classifier assigns (a sample, same service as production)
    settings = load_settings()
    svc = TriageService(settings, dedup=DedupIndex(), model=ComplaintModel.load(settings.model_path))
    sample = random.Random(11).sample(comps, min(SAMPLE, len(comps)))
    by_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in sample:
        cat = svc.triage(record_to_request(r))[0].category
        by_cat[cat][r.get("verdict") or "none"] += 1
    out["verdict_by_predicted_category"] = {
        "classifier": svc.classifier, "sample": len(sample),
        "rows": {c: {"n": sum(v.values()), **{k: round(n / sum(v.values()), 3) for k, n in v.most_common()}}
                 for c, v in sorted(by_cat.items(), key=lambda kv: -sum(kv[1].values())) if sum(v.values()) >= MIN_COUNT},
    }
    rsample = random.Random(12).sample(reports, min(SAMPLE, len(reports)))
    out["report_categories_sample"] = dict(collections.Counter(svc.triage(record_to_request(r))[0].category for r in rsample).most_common())
    (METRICS / "history.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
