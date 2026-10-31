"""Build CSV files for HUMAN annotation (nothing is pre-labelled, predictions are hidden).

  data/local/review_dev.csv    - drawn from the TRAIN file; use it while tuning the lexicon
  data/local/review_final.csv  - drawn from the VALID file; touch only for the final measurement

Rules that keep the final number honest:
  * different source files, and any final text that also occurs in the dev sample is removed;
  * one ticket per distinct normalized text, at most 3 tickets per reporter (repeats and the
    "same author, many tickets" series would otherwise dominate);
  * coverage quotas by *keyword-guessed* group so rare categories are present (the model's own
    prediction is not shown to annotators, to avoid anchoring).
The files contain real (anonymized) ticket text, therefore they live in git-ignored data/local/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import random

from _common import DATA, LOCAL, load_jsonl, player_text
from app.adapter import record_to_request
from app.dedup import DedupIndex
from app.pipeline import TriageService
from app.textutil import clean_for_matching

COLUMNS = ["ticket_id", "kind", "dataset_lang_label", "attachments", "rule_refs_raw", "text",
           "human_category", "human_priority", "human_lang", "annotator", "notes"]


def pick(records: list[dict], n: int, seed: int, exclude_texts: set[str], exclude_reporters: set[str] | None = None,
         prefer_reporters_not_in: set[str] | None = None) -> list[dict]:
    svc = TriageService(dedup=DedupIndex())
    rng = random.Random(seed)
    rng.shuffle(records)
    if prefer_reporters_not_in:  # hold-out first: reporters the development data never saw (stable sort keeps the shuffle inside each part)
        records.sort(key=lambda r: r["reporter_hash"] in prefer_reporters_not_in)
    exclude_reporters = exclude_reporters or set()
    groups = collections.defaultdict(list)
    seen_text, per_reporter = set(exclude_texts), collections.Counter()
    for r in records:
        key = clean_for_matching(player_text(r))
        if key in seen_text or per_reporter[r["reporter_hash"]] >= 3 or r["reporter_hash"] in exclude_reporters:
            continue
        seen_text.add(key)
        per_reporter[r["reporter_hash"]] += 1
        resp, _ = svc.triage(record_to_request(r))
        groups[(resp.category, resp.lang if resp.lang in {"kk", "uz"} else "other")].append(r)
    # round-robin over groups -> rare categories and kk/uz get represented
    chosen: list[dict] = []
    keys = sorted(groups)
    while len(chosen) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop())
    rng.shuffle(chosen)
    return chosen


def write(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r["ticket_id"], r["kind"], r["lang"], r.get("images_count") or 0, r.get("rule_refs_raw") or "",
                        player_text(r).replace("\r", " "), "", "", "", "", ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--final", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    train = load_jsonl(DATA / "tickets_train.jsonl.gz")
    valid = load_jsonl(DATA / "tickets_valid.jsonl")
    dev = pick(train, a.dev, a.seed, set())
    dev_texts = {clean_for_matching(player_text(r)) for r in dev}
    dev_reporters = {r["reporter_hash"] for r in dev}
    train_reporters = {r["reporter_hash"] for r in train}
    # hold-out: different texts AND different reporters than the development sample, reporters unseen in train first
    final = pick(valid, a.final, a.seed + 1, dev_texts, exclude_reporters=dev_reporters, prefer_reporters_not_in=train_reporters)
    unseen = sum(r["reporter_hash"] not in train_reporters for r in final)
    print(f"final sample: {unseen} of {len(final)} tickets come from reporters absent from train (the rest are not independent of the training reporters; "
          "valid texts were also read during development, so this is a hold-out, not an untouched set)")
    write(LOCAL / "review_dev.csv", dev)
    write(LOCAL / "review_final.csv", final)
    print(f"dev={len(dev)} final={len(final)} -> {LOCAL}. Fill human_category / human_priority (see docs/ANNOTATION_GUIDE.md),")
    print("save as review_final_labeled.csv, then run: python scripts/score_review.py --labels data/local/review_final_labeled.csv")


if __name__ == "__main__":
    main()
