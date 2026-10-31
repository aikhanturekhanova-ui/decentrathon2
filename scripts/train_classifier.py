"""Train the learned complaint-category model -> data/models/complaint_clf.joblib (git-ignored) and
docs/metrics/ml_classifier.json (aggregated numbers only).

Labels are WEAK (silver): derived from the administrator's reply in TRAIN (rule code such as «2.1 ОП», appeal verdicts),
see scripts/_common.silver_category. The administrator's text is used only to make the label; the model input is the
player's text prepared by app.classify.prepare_text (the exact function the service uses).

Protocol (no validation labels are used for training or for choosing anything):
  1. split TRAIN by reporter (20 % of reporters held out, so the same author never sits on both sides);
  2. fit on the rest, choose the blend weight on the hold-out, report lexicon / model / hybrid there;
  3. report the same on VALID silver labels (confirmation only);
  4. refit on all TRAIN silver labels and save.
`--fast` skips 1-3 (used by `make predict` when no model exists yet).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from datetime import datetime, timezone

from _common import DATA, METRICS, ROOT, load_jsonl, macro_f1, macro_f1_report, silver_category
from app import classify
from app.adapter import record_to_request
from app.classify import analyze, prepare_text
from app.config import load_settings
from app.ml import ComplaintModel, train
from app.schemas import Message, TriageRequest
from app.store import Store

GOLD_WEIGHT = 3  # an administrator's explicit decision counts as three weak labels


def gold_rows(settings, classes: set[str]) -> list[tuple[dict, str, str]]:
    """Administrators' final decisions (POST /api/feedback, «Решение администратора») for complaints of a class the model knows."""
    if not settings.db_path.is_file():
        return []
    out = []
    for d in Store(settings.db_path).decisions():
        if d["kind"] == "complaint" and d["final_category"] in classes and d["player_text"]:
            req = TriageRequest(kind="complaint", messages=[Message(role="player", text=d["player_text"])])
            out.append(({"source": "admin_decision"}, prepare_text(req)[3], d["final_category"]))
    return out


def labelled(records: list[dict]) -> list[tuple[dict, str, str]]:
    out = []
    for r in records:
        if r.get("kind") != "complaint":
            continue
        y, _ = silver_category(r)
        if y:
            out.append((r, prepare_text(record_to_request(r))[3], y))
    return out


def held_out(rec: dict) -> bool:
    return int(hashlib.md5(str(rec.get("reporter_hash")).encode()).hexdigest(), 16) % 5 == 0


def evaluate(name: str, rows: list[tuple[dict, str, str]], model: ComplaintModel, weights: list[float]) -> dict:
    y = [lab for _, _, lab in rows]
    reqs = [record_to_request(r) for r, _, _ in rows]
    lex = [analyze(q).category for q in reqs]
    probs = model.proba_many([t for _, t, _ in rows])
    alone = [max(p, key=p.get) for p in probs]
    out = {"n": len(rows), "label_distribution": dict(collections.Counter(y).most_common()),
           "lexicon": {"agreement": acc(y, lex), "macro_f1": round(macro_f1(y, lex), 4)},
           "model_alone": {"agreement": acc(y, alone), "macro_f1": round(macro_f1(y, alone), 4)}, "hybrid_by_weight": {}}
    saved = classify.ML_BLEND_WEIGHT
    try:
        for w in weights:
            classify.ML_BLEND_WEIGHT = w
            hyb = [analyze(q, model).category for q in reqs]
            out["hybrid_by_weight"][str(w)] = {"agreement": acc(y, hyb), "macro_f1": round(macro_f1(y, hyb), 4)}
            if w == saved:
                out["hybrid"] = macro_f1_report(y, hyb) | {"agreement": acc(y, hyb), "weight": w}
                out["hybrid_top_disagreements"] = [[a, b, c] for (a, b), c in
                                                   collections.Counter((a, b) for a, b in zip(y, hyb) if a != b).most_common(8)]
    finally:
        classify.ML_BLEND_WEIGHT = saved
    print(name, json.dumps({k: v for k, v in out.items() if k in {"n", "lexicon", "model_alone", "hybrid_by_weight"}}, ensure_ascii=False))
    return out


def acc(y: list[str], p: list[str]) -> float:
    return round(sum(a == b for a, b in zip(y, p)) / max(len(y), 1), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="only fit and save (no evaluation)")
    args = ap.parse_args()
    settings = load_settings()
    t0 = time.time()
    train_rows = labelled(load_jsonl(DATA / "tickets_train.jsonl.gz"))
    print(f"silver-labelled train complaints: {len(train_rows)} ({time.time() - t0:.0f} s)")
    report: dict = {"note": ("silver labels from administrators' replies (TRAIN only); NOT human annotation. Agreement with silver labels "
                             "is not the hidden-test accuracy, and silver labels exist only for complaints with a rule code or an appeal verdict."),
                    "features": "TF-IDF char_wb 2-5 (200k) + word 1-2; LogisticRegression C=4, class_weight=balanced",
                    "blend": "w * P_model + (1 - w) * normalised lexicon scores; complaint.invalid and safety flags stay rule-based"}
    if not args.fast:
        fit_rows = [x for x in train_rows if not held_out(x[0])]
        dev_rows = [x for x in train_rows if held_out(x[0])]
        m = train([t for _, t, _ in fit_rows], [y for _, _, y in fit_rows])
        weights = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
        report["train_holdout_by_reporter"] = evaluate("holdout", dev_rows, m, weights)
        best = max(weights, key=lambda w: report["train_holdout_by_reporter"]["hybrid_by_weight"][str(w)]["agreement"])
        report["best_weight_on_holdout"] = best
        report["weight_in_use"] = classify.ML_BLEND_WEIGHT
        report["valid_silver_confirmation"] = evaluate("valid", labelled(load_jsonl(DATA / "tickets_valid.jsonl")), m, weights)
    gold = gold_rows(settings, {y for _, _, y in train_rows})
    if gold:
        print(f"administrators' decisions added as human labels: {len(gold)} (weight {GOLD_WEIGHT})")
    train_rows = train_rows + gold * GOLD_WEIGHT
    report["admin_decisions_used"] = len(gold)
    model = train([t for _, t, _ in train_rows], [y for _, _, y in train_rows],
                  meta={"trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n_train": len(train_rows),
                        "blend_weight": classify.ML_BLEND_WEIGHT, "admin_decisions": len(gold),
                        "labels": "silver (admin replies, train only)" + (f" + {len(gold)} administrators' decisions" if gold else "")})
    model.meta["classes"] = model.classes
    model.save(settings.model_path)
    print(f"saved {settings.model_path.relative_to(ROOT) if settings.model_path.is_relative_to(ROOT) else settings.model_path} "
          f"({time.time() - t0:.0f} s)")
    if not args.fast:
        report["final_model"] = {"n_train": len(train_rows), "classes": model.classes}
        (METRICS / "ml_classifier.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
