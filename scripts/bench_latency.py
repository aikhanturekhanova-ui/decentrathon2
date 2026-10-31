"""End-to-end HTTP latency of POST /triage against a real uvicorn process (client-side wall clock).

Conditions are recorded in the output; a number without them is meaningless.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import httpx

from _common import DATA, METRICS, ROOT, load_jsonl
from app.adapter import record_to_request


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    # a throw-away database: the measured service writes its duplicate history to SQLite (the delivered configuration), but not into the dev database
    tmp = tempfile.mkdtemp(prefix="bench_")
    env = {**os.environ, "PERSIST_TRIAGE": "0", "DB_PATH": os.path.join(tmp, "bench.db"), "DEDUP_PERSIST": "1"}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(a.port), "--log-level", "warning"],
                           cwd=ROOT, env=env)
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(60):
            try:
                health = httpx.get(base + "/health", timeout=1).json()
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise SystemExit("server did not start")
        recs = load_jsonl(DATA / "tickets_valid.jsonl")[: a.n]
        bodies = [record_to_request(r).model_dump(exclude_none=True) for r in recs]
        lat = []
        with httpx.Client(timeout=30) as c:
            for b in bodies[:20]:  # warm-up, not counted
                c.post(base + "/triage", json=b)
            for b in bodies:
                t = time.perf_counter()
                r = c.post(base + "/triage", json=b)
                lat.append((time.perf_counter() - t) * 1000)
                r.raise_for_status()
        out = {
            "requests": len(lat), "sequential": True, "client": "httpx keep-alive on 127.0.0.1",
            "p50_ms": round(pct(lat, .5), 2), "p95_ms": round(pct(lat, .95), 2), "p99_ms": round(pct(lat, .99), 2), "max_ms": round(max(lat), 2),
            "service_mode": health["mode"], "dedup_history_in_sqlite": health.get("dedup_persistent"), "llm_calls": 0 if not health["llm_enabled"] else "enabled",
            "machine": {"os": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()},
            "note": "template mode: no network model call is included. LLM mode adds one model call per ticket (see docs/REPORT.md).",
        }
        (METRICS / "latency.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
