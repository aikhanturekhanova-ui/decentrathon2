"""Real-API trial of the LLM mode on the ARTIFICIAL demo tickets (no real player data is sent).

Needs ANTHROPIC_API_KEY or GEMINI_API_KEY (+ LLM_PRICE_IN_PER_MTOK / LLM_PRICE_OUT_PER_MTOK for cost). Reports: how many
model drafts were accepted by the guard, why the others were rejected, latency p50/p95 of the whole
triage call, and the measured cost. Writes docs/metrics/llm_smoke.json (aggregates only).

    python scripts/llm_smoke.py [--repeat 5]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

import os

from _common import METRICS, ROOT

os.environ["LLM_ENABLED"] = "1"  # this script sends ONLY the artificial demo tickets (tests/fixtures/demo_tickets.jsonl)
from app.dedup import DedupIndex
from app.pipeline import TriageService
from app.schemas import TriageRequest


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    svc = TriageService(dedup=DedupIndex())
    if svc.llm is None:
        print("No ANTHROPIC_API_KEY or GEMINI_API_KEY: nothing to test. Set one in .env and run again.")
        return 2
    reqs = []
    for line in (ROOT / "tests/fixtures/demo_tickets.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        d.pop("reporter_id", None)  # no dedup in this trial
        reqs.append(TriageRequest.model_validate(d))
    lat, costs, modes, errors, unknown = [], [], collections.Counter(), collections.Counter(), 0
    for _ in range(a.repeat):
        for r in reqs:
            _, dbg = svc.triage(r)
            lat.append(dbg.latency_ms)
            modes[dbg.mode] += 1
            if dbg.llm_error:
                errors[dbg.llm_error.split(":")[0][:60]] += 1
            if dbg.llm_calls:
                if dbg.cost_usd is None:
                    unknown += 1
                else:
                    costs.append(dbg.cost_usd)
    n = len(lat)
    out = {
        "tickets": n, "provider": svc.settings.llm_provider, "model": svc.settings.llm_model, "answers_by_mode": dict(modes), "rejections_or_failures": dict(errors),
        "llm_draft_acceptance_rate": round(modes["llm"] / max(sum(1 for _ in range(n)), 1), 3),
        "latency_ms_total_triage": {"p50": round(pct(lat, .5), 1), "p95": round(pct(lat, .95), 1), "max": round(max(lat), 1)},
        "cost_usd_per_ticket_measured": round(sum(costs) / len(costs), 6) if costs else None,
        "calls_with_unknown_cost": unknown,
        "note": "artificial tickets only; includes network latency of this machine",
    }
    (METRICS / "llm_smoke.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
