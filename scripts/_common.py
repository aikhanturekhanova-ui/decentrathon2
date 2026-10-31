"""Shared helpers for the evaluation scripts. Everything these scripts write stays in data/local/
(git-ignored) except AGGREGATED numbers, which go to docs/metrics/."""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import os

# These scripts read the organizers' REAL tickets. Real player text must never be sent to an external model from them (data terms; a free
# LLM tier may also use the text for training), whatever keys a local .env holds. Only scripts/llm_smoke.py (artificial tickets) turns it on.
os.environ["LLM_ENABLED"] = "0"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "handout"
LOCAL = ROOT / "data" / "local"
METRICS = ROOT / "docs" / "metrics"
for d in (LOCAL, METRICS):
    d.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def player_text(rec: dict) -> str:
    return " ".join(m["text"] for m in rec["messages"] if m["role"] == "player").strip()


def admin_text(rec: dict) -> str:
    return " ".join(m["text"] for m in rec["messages"] if m["role"] == "admin")


# --- SILVER labels --------------------------------------------------------------------------
# Derived from the administrator's reply (rule code / appeal verdict). They describe what the admin
# did, not necessarily what the player wrote, and they exist for ~15% of complaints. They are NOT
# human annotation of the ticket and are only used as a weak agreement check, never as accuracy.
_GEN = re.compile(r"(\d+\.\d+(?:\.\d+)?)\s*(?:оп|op)(?![\w])", re.I)
_GZ = re.compile(r"(?:(?<![\d.])(\d+\.\d+)\s*(?:гз|gz|зз)|(?:гз|gz)\s*(\d+\.\d+))", re.I)
RULE_TO_CAT = {
    "2.1": "complaint.insult", "2.4": "complaint.insult", "2.6": "complaint.insult",
    "5.2": "complaint.cheat",
    "5.3": "complaint.dm", "5.4": "complaint.dm",
    "4.1": "complaint.scam", "4.2": "complaint.scam",
    "2.3": "complaint.rmt_ads", "4.4": "complaint.rmt_ads",
    "2.2": "complaint.other", "1.6": "complaint.other", "2.5": "complaint.other", "5.9": "complaint.other",
}


def silver_category(rec: dict) -> tuple[str | None, str | None]:
    """Return (category, rule_code_as_cite) or (None, None) when the admin reply gives no usable signal."""
    if rec["kind"] != "complaint":
        return None, None
    if rec.get("verdict") in {"appeal_approved", "appeal_rejected"}:
        return "complaint.admin", None
    blob = f"{rec.get('violation') or ''} {admin_text(rec)}"
    m = _GZ.search(blob)
    if m:
        return "complaint.dm", "GZ " + (m.group(1) or m.group(2))
    m = _GEN.search(blob)
    if m and m.group(1) in RULE_TO_CAT:
        return RULE_TO_CAT[m.group(1)], m.group(1)
    # RP violations named by their term (the rulebook's «RP термины»): DM, DB, TK, SK, MG (metagaming), PG, NonRP
    if re.search(r"(?<![\w])(?:dm|дм|db|дб|tk|тк|sk|ск|mg|мг|pg|пг|non\s*-?\s*rp|нон\s*-?\s*рп|metagaming|powergaming)(?![\w])",
                 rec.get("violation") or "", re.I):
        return "complaint.dm", None
    return None, None


def per_class(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict[str, dict]:
    """Precision / recall / F1 / support per class. Undefined ratios (no predictions or no true examples) are 0.0, as in
    scikit-learn with zero_division=0, so that a class we never predict (or predict only wrongly) lowers the average."""
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    out = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": tp + fn, "predicted": tp + fp}
    return out


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> float:
    """Macro-F1 over the classes that occur in the gold labels OR in the predictions (the scikit-learn convention).

    Why the union: averaging over the gold classes only (the old behaviour) ignores a class that the system predicts but the
    annotators never used, so a system that invents categories would score the same as one that does not. A class that appears
    in neither list is not counted. Pass `labels` to fix the class set explicitly (for example the 14 taxonomy codes)."""
    stats = per_class(y_true, y_pred, labels)
    return sum(s["f1"] for s in stats.values()) / len(stats) if stats else 0.0


def macro_f1_report(y_true: list[str], y_pred: list[str]) -> dict:
    """Both conventions side by side, plus the classes that only one side uses, so the number can be read honestly."""
    gold, pred = set(y_true), set(y_pred)
    return {
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "macro_f1_gold_classes_only": round(macro_f1(y_true, y_pred, sorted(gold)), 4),
        "classes_only_predicted": sorted(pred - gold),
        "classes_only_in_gold": sorted(gold - pred),
        "per_class": {k: {m: round(v, 3) if isinstance(v, float) else v for m, v in s.items()} for k, s in per_class(y_true, y_pred).items()},
    }
