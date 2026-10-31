"""Run the artificial gold set -> docs/metrics/synthetic_gold.json (regression numbers, NOT accuracy)."""
from __future__ import annotations

import json

from _common import METRICS, ROOT, macro_f1
from app.pipeline import TriageService
from app.schemas import Message, TriageRequest


def main() -> None:
    gold = [json.loads(l) for l in (ROOT / "tests/fixtures/gold_synthetic.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    svc = TriageService()
    yc, pc, yp, pp = [], [], [], []
    for g in gold:
        r, _ = svc.triage(TriageRequest(kind=g["kind"], messages=[Message(text=g["text"], attachments=g["att"])], server_lang=g.get("server_lang")))
        yc.append(g["expect"]["category"]), pc.append(r.category), yp.append(g["expect"]["priority"]), pp.append(r.priority)
    out = {"n": len(gold), "category_accuracy": round(sum(a == b for a, b in zip(yc, pc)) / len(gold), 4),
           "priority_macro_f1": round(macro_f1(yp, pp), 4), "caveat": "authors' own artificial examples; regression guard, not an accuracy estimate"}
    (METRICS / "synthetic_gold.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
