"""In-process metrics in the Prometheus text format (GET /metrics), without an extra dependency.

What an operations team needs to run the service: request counts by route and status, triage latency (histogram), the share of
tickets sent to a human, categories, priorities, languages, spam/duplicates, LLM calls and fallbacks. Counters reset on restart
(Prometheus handles that); with several replicas each one is scraped separately.
"""
from __future__ import annotations

import threading
from collections import Counter

_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
_lock = threading.Lock()
_requests: Counter = Counter()  # (route, status)
_triage: Counter = Counter()  # (label, value)
_lat_buckets: Counter = Counter()
_lat_sum = 0.0
_lat_count = 0


def record_request(route: str, status: int) -> None:
    with _lock:
        _requests[(route, status)] += 1


def record_triage(resp, dbg) -> None:
    global _lat_sum, _lat_count
    with _lock:
        _triage[("category", resp.category)] += 1
        _triage[("priority", resp.priority)] += 1
        _triage[("lang", resp.lang)] += 1
        _triage[("needs_human", str(resp.needs_human).lower())] += 1
        _triage[("spam_or_duplicate", str(resp.is_spam_or_duplicate).lower())] += 1
        _triage[("mode", dbg.mode)] += 1
        if dbg.llm_calls:
            _triage[("llm", "called")] += 1
        if dbg.llm_error:
            _triage[("llm", "fallback_to_template")] += 1
        ms = dbg.latency_ms
        _lat_sum += ms
        _lat_count += 1
        for b in _BUCKETS_MS:
            if ms <= b:
                _lat_buckets[b] += 1


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render() -> str:
    with _lock:
        lines = ["# HELP copilot_http_requests_total HTTP requests by route and status.", "# TYPE copilot_http_requests_total counter"]
        lines += [f'copilot_http_requests_total{{route="{_esc(r)}",status="{s}"}} {n}' for (r, s), n in sorted(_requests.items())]
        lines += ["# HELP copilot_triage_total Triaged tickets by outcome dimension.", "# TYPE copilot_triage_total counter"]
        lines += [f'copilot_triage_total{{dimension="{_esc(k)}",value="{_esc(v)}"}} {n}' for (k, v), n in sorted(_triage.items())]
        lines += ["# HELP copilot_triage_latency_ms Pipeline latency per ticket (milliseconds).", "# TYPE copilot_triage_latency_ms histogram"]
        lines += [f'copilot_triage_latency_ms_bucket{{le="{b}"}} {_lat_buckets[b]}' for b in _BUCKETS_MS]
        lines += [f'copilot_triage_latency_ms_bucket{{le="+Inf"}} {_lat_count}', f"copilot_triage_latency_ms_sum {round(_lat_sum, 3)}",
                  f"copilot_triage_latency_ms_count {_lat_count}"]
    return "\n".join(lines) + "\n"
