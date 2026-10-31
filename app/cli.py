"""Batch mode:  python -m app.cli predict --in tickets.jsonl --out pred.jsonl   (see `make predict`).

The batch path calls exactly the same TriageService.triage() as POST /triage. It accepts both
compact /triage bodies and full dataset records (via app.adapter) and never feeds administrator
answers or verdict fields to the pipeline.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

from .adapter import to_request
from .dedup import parse_ts
from .pipeline import TriageService


def open_text(path: str, mode: str = "rt"):
    # reading uses utf-8-sig: files saved by Windows editors / PowerShell start with a BOM, which plain utf-8 turns into a crash
    enc = "utf-8-sig" if "r" in mode else "utf-8"
    nl = {} if "r" in mode else {"newline": "\n"}
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=enc, **nl)
    return open(path, mode, encoding=enc, **nl)


def safe_default(rec: dict, why: str) -> dict:
    """A VALID contract answer for a record we could not process (so that one bad line never breaks the file, and a
    downstream evaluator still finds every contract field): generic category, human required, nothing cited."""
    from .phrases import P

    kind = rec.get("kind") if rec.get("kind") in {"complaint", "report"} else "report"
    T = P["ru"]
    return {
        "lang": "und", "category": f"{kind}.other" if kind == "complaint" else "report.offtopic", "priority": "p2",
        "is_spam_or_duplicate": False,
        "extracted": {"nickname": None, "rule_refs": [], "has_evidence": False, "server": None, "attachments": 0, "evidence_verified": False, "rule_refs_unverified": []},
        "draft_reply": {"lang": "ru", "text": f"{T['hello']} {T['off_body']} {T['off_ask']}", "cited": [], "terms": []},
        "needs_human": True, "confidence": 0.0,
        "reasons": [f"Запись не удалось обработать автоматически ({why}); подставлен безопасный ответ, нужен человек"],
        # additive and explicit: this line is NOT a processed ticket (a downstream evaluator can filter it)
        "processing_error": why,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def predict(in_path: str, out_path: str, limit: int | None = None, service: TriageService | None = None) -> dict:
    svc = service or TriageService(hint_dirs=[Path(in_path).resolve().parent])  # rules.json may sit next to the tickets
    if svc.model is None and svc.settings.use_model:
        # not fatal (the lexicon is a complete classifier), but the measured quality of the hybrid needs the model: say so loudly
        print("[WARNING] no trained complaint model (data/models/complaint_clf.joblib): complaints are classified by the LEXICON ONLY. "
              "Put tickets_train.jsonl.gz into data/handout/ and run `make train` (or `make predict`, which trains it once).", file=sys.stderr)
    records: list[dict] = []
    with open_text(in_path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    records.append(obj if isinstance(obj, dict) else {"_bad": f"line {n} is not a JSON object"})
                except json.JSONDecodeError as exc:
                    records.append({"_bad": f"line {n} is not valid JSON ({exc.msg})"})
            if limit and len(records) >= limit:
                break

    # chronological order lets the dedup window work no matter how the file is sorted
    order = sorted(range(len(records)), key=lambda i: (parse_ts(records[i].get("created_at")) or parse_ts("1970-01-01T00:00:00"), i))
    results: dict[int, dict] = {}
    stats = {"n": 0, "latency_ms": [], "cost_usd": [], "llm_calls": 0, "errors": 0, "modes": {}}
    t_start = time.perf_counter()
    for i in order:
        rec = records[i]
        try:
            if "_bad" in rec:
                raise ValueError(rec["_bad"])
            req = to_request(rec)
            resp, dbg = svc.triage(req)
            out = resp.model_dump()
        except Exception as exc:  # one bad record must not kill the batch: it gets a valid, safe, human-required answer
            stats["errors"] += 1
            why = f"{type(exc).__name__}: {str(exc)[:120]}"
            print(f"[warn] record {i} ({rec.get('ticket_id')}): {why}", file=sys.stderr)
            out = safe_default(rec, why)
            dbg = None
        if rec.get("ticket_id") is not None:
            out = {"ticket_id": rec["ticket_id"], **out}
        results[i] = out
        stats["n"] += 1
        if dbg is not None:
            stats["latency_ms"].append(dbg.latency_ms)
            stats["llm_calls"] += dbg.llm_calls
            stats["llm_requests_sent"] = stats.get("llm_requests_sent", 0) + dbg.llm_requests_sent
            if dbg.cost_estimate_max_usd is not None:
                stats["cost_estimate_max_usd"] = stats.get("cost_estimate_max_usd", 0.0) + dbg.cost_estimate_max_usd
            if dbg.cost_usd is None:
                stats["cost_unknown"] = stats.get("cost_unknown", 0) + 1  # model called, prices not configured
            elif dbg.cost_usd is not None:
                stats["cost_usd"].append(dbg.cost_usd)
            stats["modes"][dbg.mode] = stats["modes"].get(dbg.mode, 0) + 1
    wall = time.perf_counter() - t_start

    with open_text(out_path, "wt") as f:
        for i in range(len(records)):
            f.write(json.dumps(results[i], ensure_ascii=False) + "\n")

    lat = stats["latency_ms"]
    summary = {
        "records": stats["n"], "errors": stats["errors"], "service_mode": svc.mode, "classifier": svc.classifier,
        "latency_ms": {"p50": round(percentile(lat, .5), 2), "p95": round(percentile(lat, .95), 2),
                       "mean": round(statistics.fmean(lat), 2) if lat else 0.0, "max": round(max(lat), 2) if lat else 0.0},
        "wall_seconds": round(wall, 2), "llm_calls": stats["llm_calls"], "llm_requests_sent": stats.get("llm_requests_sent", 0), "answer_modes": stats["modes"],
        # an ESTIMATE (worst case), never mixed with the measured cost below
        "cost_estimate_max_usd_total": round(stats["cost_estimate_max_usd"], 6) if "cost_estimate_max_usd" in stats else None,
    }
    unknown = stats.get("cost_unknown", 0)
    known = len(stats["cost_usd"])
    if stats.get("llm_requests_sent", 0) == 0 and unknown == 0:
        status = "no_external_calls"
    elif unknown == 0:
        status = "known"
    elif known == 0:
        status = "unknown"  # model was called but LLM_PRICE_* are not set: we do NOT report $0
    else:
        status = "partially_unknown"
    summary["cost_status"] = status
    summary["tickets_with_unknown_cost"] = unknown
    known_sum = round(sum(stats["cost_usd"]), 6)
    # the TOTAL is reported only when every ticket's cost is known; otherwise it stays null and the
    # known part is stored separately so that nobody reads a partial sum as the total
    summary["cost_usd_known_sum"] = known_sum if known else None
    if status in {"no_external_calls", "known"}:
        summary["cost_usd_total"] = known_sum
        summary["cost_usd_per_ticket"] = round(known_sum / max(stats["n"], 1), 6)
    else:
        summary["cost_usd_total"] = None
        summary["cost_usd_per_ticket"] = None
    Path(out_path + ".stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="triage a JSONL file")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "predict":
        summary = predict(args.inp, args.out, args.limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["errors"]:
            # the output file is complete and valid (bad records got the safe answer), so the run itself succeeded
            print(f"[warn] {summary['errors']} record(s) could not be processed and received the safe fallback answer; see messages above", file=sys.stderr)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
