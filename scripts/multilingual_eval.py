"""Adversarial multilingual suite -> docs/metrics/multilingual.json.

Cases are ARTIFICIAL, written by the authors (tests/fixtures/multilingual_cases.jsonl): pure languages, mixed messages, Kazakh look-alike
letters, Kazakh/Russian in Latin, obfuscated swearing, quoted vs own abuse, threats in and out of the game, jokes and false-positive traps.
They test the language and toxicity modules; they are NOT a measurement on real tickets. Cases marked `known_limitation` are expected
to fail and are reported as such. A second part counts how often the modules fire on the real VALID tickets (aggregates only).
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["LLM_ENABLED"] = "0"  # reads real VALID tickets: never an external model
sys.path.insert(0, str(ROOT))

from app.langseg import profile  # noqa: E402
from app.toxicity import analyze_messages  # noqa: E402

CASES = ROOT / "tests" / "fixtures" / "multilingual_cases.jsonl"


def check(case: dict) -> list[str]:
    """Problems of one case (empty list = pass)."""
    problems = []
    rep = analyze_messages([{"role": "player", "text": case["text"]}])
    if rep.has_abuse != case["abuse"]:
        problems.append(("false positive" if rep.has_abuse else "false negative") + f": found {[f.text for f in rep.findings]}")
    for t in case.get("types") or []:
        if t not in rep.types:
            problems.append(f"type {t} not found (got {rep.types})")
    if case.get("direction") and any(f.direction != case["direction"] for f in rep.findings):
        problems.append(f"direction {[f.direction for f in rep.findings]} != {case['direction']}")
    if case["group"] == "threat_real" and not rep.real_life_threat:
        problems.append("real-life threat not recognised")
    if case["group"] == "threat_game" and (rep.real_life_threat or rep.max_severity > 1):
        problems.append("in-game threat treated as serious")
    if case["group"] == "joke" and rep.tone != "joking":
        problems.append(f"tone {rep.tone} != joking")
    lp = profile(case["text"])
    if case.get("lang") and lp.primary != case["lang"]:
        problems.append(f"language {lp.primary} != {case['lang']}")
    if case.get("mixed") is not None and lp.mixed != case["mixed"]:
        problems.append(f"mixed {lp.mixed} != {case['mixed']} (segments {[(s.text, s.lang) for s in lp.segments]})")
    return problems


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def real_data_prevalence() -> dict | None:
    valid = ROOT / "data" / "handout" / "tickets_valid.jsonl"
    if not valid.is_file():
        return None
    recs = [json.loads(line) for line in valid.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    c: collections.Counter = collections.Counter()
    for r in recs:
        msgs = [m for m in r["messages"] if m["role"] != "admin"]
        rep = analyze_messages(msgs)
        c["tickets"] += 1
        c["with_findings"] += rep.has_abuse
        c["reported_abuse"] += rep.reported_abuse
        c["author_abusive"] += rep.author_abusive
        c["accused_abusive"] += rep.accused_abusive
        c["real_life_threat"] += rep.real_life_threat
        c["mixed_language"] += profile(" ".join(m["text"] for m in msgs if m["role"] == "player")).mixed
        for t in rep.types:
            c["type:" + t] += 1
    return dict(c)


def main() -> None:
    cases = load_cases()
    rows, groups = [], collections.Counter()
    tp = fp = fn = tn = 0
    for case in cases:
        problems = check(case)
        lim = case.get("known_limitation")
        status = "pass" if not problems else ("known_limitation" if lim else "FAIL")
        rows.append({"group": case["group"], "status": status, "problems": problems, **({"limitation": lim} if lim else {})})
        groups[status] += 1
        found = analyze_messages([{"role": "player", "text": case["text"]}]).has_abuse
        tp += found and case["abuse"]
        fp += found and not case["abuse"]
        fn += (not found) and case["abuse"]
        tn += (not found) and not case["abuse"]
    out = {
        "note": "artificial cases written by the authors; a regression and coverage check, NOT accuracy on real tickets",
        "cases": len(cases), "status": dict(groups),
        "abuse_detection_on_cases": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                     "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)},
        "false_positives": [r for r in rows if any(p.startswith("false positive") for p in r["problems"])],
        "false_negatives": [r for r in rows if any(p.startswith("false negative") for p in r["problems"])],
        "failures": [r for r in rows if r["status"] != "pass"],
        "real_valid_prevalence": real_data_prevalence(),
    }
    (ROOT / "docs" / "metrics" / "multilingual.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, ensure_ascii=False, indent=2))
    for r in out["failures"]:
        print(r["status"], r["group"], r["problems"])


if __name__ == "__main__":
    main()
