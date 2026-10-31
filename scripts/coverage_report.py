"""Coverage and safety counters over real tickets -> docs/metrics/coverage.json (aggregates only, no ticket text).

These numbers describe BEHAVIOUR, not correctness:
  * substantive_answer_share   drafts that carry a verified rule point or a defined game term (not just a request for details);
  * escalation_share           needs_human = true (a LOW share is not a goal: mandatory topics must always escalate);
  * citation_validity          every cited point exists, passes its anchor and is mentioned in the text (re-checked by the guard);
  * fallback_share             a draft replaced by the safe template because a check rejected it (template mode: expected 0);
  * kk / uz breakdown          the same counters for tickets detected as Kazakh / Uzbek;
  * sensitive_topic_errors     mandatory-handoff topics that did NOT reach a person, were not p0 where p0 is required, or were refused
                               as "invalid" - these must be zero, whatever the escalation share is.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from _common import DATA, METRICS, load_jsonl
from app.adapter import record_to_request
from app.classify import analyze
from app.dedup import DedupIndex, parse_ts
from app.guard import TRANSFER_CLAIM, check_draft
from app.phrases import P
from app.pipeline import TriageService
from app.rag import ANCHORS, context_ok

P0_FLAGS = {"payment", "hack", "access", "threat_outside_game", "mass_bug"}


def measure(records: list[dict], svc: TriageService) -> dict:
    order = sorted(range(len(records)), key=lambda i: (parse_ts(records[i].get("created_at")) or parse_ts("1970-01-01T00:00:00"), i))
    n = len(records)
    c = collections.Counter()
    by_lang: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    flags_seen = collections.Counter()
    for i in order:
        req = record_to_request(records[i])
        resp, dbg = svc.triage(req)
        a = analyze(req)
        for bucket in (c, by_lang[resp.lang if resp.lang in {"kk", "uz"} else "other"]):
            bucket["n"] += 1
            bucket["needs_human"] += resp.needs_human
            bucket["substantive"] += bool(resp.draft_reply.cited or resp.draft_reply.terms)
            bucket["with_citation"] += bool(resp.draft_reply.cited)
            bucket["spam_or_duplicate"] += resp.is_spam_or_duplicate
            bucket["draft_in_detected_lang"] += resp.draft_reply.lang == resp.lang
            bucket["template_only"] += dbg.mode == "template"
        for cite in resp.draft_reply.cited:
            c["cites_total"] += 1
            doc = svc.rules.resolve_cite(cite)
            anchor = ANCHORS.get(cite)
            # valid = the point exists, is citable, its real text contains the anchor words, and the draft names it
            c["cites_valid"] += bool(doc is not None and doc.citable and (not anchor or re.search(anchor, doc.body, flags=re.I)) and doc.ref in resp.draft_reply.text)
            if cite.startswith("GZ") and resp.category != "report.question":  # a question about a rule describes no event: no place/action facts are required
                c["gz_cites"] += 1
                c["gz_cites_with_place_fact"] += context_ok(cite, a.clean, svc.rules, a.text)
        c["fallback_to_safe_text"] += any("Автоматическая проверка отклонила" in r for r in resp.reasons)
        c["guard_rejected_final_drafts"] += not check_draft(resp.draft_reply.text, resp.draft_reply.cited, svc.rules, resp.needs_human, terms=resp.draft_reply.terms).ok
        c["claims_transfer"] += bool(TRANSFER_CLAIM.search(resp.draft_reply.text))
        c["taxonomy_adjusted"] += any("отсутствует в таксономии" in r for r in resp.reasons)
        c["possible_duplicate_suspected"] += any("Возможный повтор" in r for r in resp.reasons)
        for f in a.flags:
            flags_seen[f] += 1
        # sensitive-topic errors
        if a.sensitive and not resp.needs_human:
            c["sensitive_without_human"] += 1
        if (P0_FLAGS & a.flags) and resp.priority != "p0":
            c["critical_signal_not_p0"] += 1
        if a.sensitive and any(P[lg][k] in resp.draft_reply.text for lg in P for k in ("inv_noev", "inv_question", "inv_vague")):
            c["sensitive_refused_as_invalid"] += 1  # the draft REFUSES the ticket ("no evidence / this is a question / unclear")
        if a.sensitive:
            c["sensitive_total"] += 1

    def share(k: str, d=c) -> float:
        return round(d[k] / max(d["n"], 1), 4)

    out = {
        "tickets": n,
        "escalation_share": share("needs_human"),
        "answers_with_a_source_share": share("substantive"),  # a rule point or a defined term is attached: NOT a measure of usefulness
        "with_citation_share": share("with_citation"),
        "spam_or_duplicate_share": share("spam_or_duplicate"),
        "possible_duplicate_suspected": c["possible_duplicate_suspected"],
        # TECHNICAL validity only: the point exists, is citable, its real text has the anchor words, the draft names it.
        # It does NOT say that the rule fits the event (that needs a human judgement of relevance; not done).
        "technical_validity_of_citations": {"cited_total": c["cites_total"], "valid": c["cites_valid"],
                                            "share": round(c["cites_valid"] / max(c["cites_total"], 1), 4)},
        "place_bound_citations": {"total": c["gz_cites"], "with_place_fact_in_text": c["gz_cites_with_place_fact"]},
        "fallback_share": round(c["fallback_to_safe_text"] / max(n, 1), 4),
        "guard_rejected_final_drafts": c["guard_rejected_final_drafts"],
        "drafts_claiming_a_transfer": c["claims_transfer"],
        "taxonomy_adjusted": c["taxonomy_adjusted"],
        # (1) POLICY COMPLIANCE for signals the service itself recognised. It cannot see a topic the detector missed (see detection_on_expectation_set).
        "policy_compliance_for_recognised_sensitive_signals": {"tickets_with_sensitive_topic": c["sensitive_total"], "without_human": c["sensitive_without_human"],
                                   "p0_signal_not_p0": c["critical_signal_not_p0"], "refused_as_invalid": c["sensitive_refused_as_invalid"]},
        "flag_counts": dict(flags_seen.most_common()),
        "by_language_group": {k: {"n": v["n"], "escalation_share": round(v["needs_human"] / max(v["n"], 1), 4),
                                  "substantive_share": round(v["substantive"] / max(v["n"], 1), 4),
                                  "with_citation_share": round(v["with_citation"] / max(v["n"], 1), 4),
                                  "draft_in_detected_language_share": round(v["draft_in_detected_lang"] / max(v["n"], 1), 4)}
                              for k, v in sorted(by_lang.items())},
    }
    return out


def detection_on_expectation_set(svc: TriageService) -> dict:
    """(2) COMPLETENESS and FALSE ALARMS of the sensitive-topic detectors on a list with expectations written BEFORE looking at the output
    (tests/fixtures/sensitive_expectations.jsonl). It is the authors' list, not independent human labels, and it grows when a review finds a miss."""
    import pathlib

    rows = [json.loads(l) for l in pathlib.Path(__file__).resolve().parent.parent.joinpath("tests/fixtures/sensitive_expectations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    per = collections.defaultdict(lambda: [0, 0])
    false_alarm, escalated_ok = 0, 0
    for r in rows:
        req = record_to_request({"ticket_id": "x", "kind": r["kind"], "messages": [{"role": "player", "text": r["text"]}], "images_count": 1})
        a = analyze(req)
        resp, _ = svc.triage(req)
        f = a.flags
        got = {"account": bool(f & {"hack", "access"}), "payment": "payment" in f, "threat": "threat_outside_game" in f, "appeal": "appeal" in f}
        if r["expect"] == "none":
            false_alarm += any(got.values())
        else:
            per[r["expect"]][1] += 1
            per[r["expect"]][0] += got[r["expect"]]
            escalated_ok += resp.needs_human and (resp.priority == "p0" or r["expect"] == "appeal")
    n_pos = sum(v[1] for v in per.values())
    return {"rows": len(rows), "positives": n_pos, "negatives": len(rows) - n_pos,
            "recall_by_topic": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())},
            "false_alarms_on_negatives": false_alarm, "positives_escalated_with_right_priority": f"{escalated_ok}/{n_pos}",
            "caveat": "authors' own list; says nothing about phrasings nobody wrote down; real tickets contain almost no such cases"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sample", type=int, default=20000)
    a = ap.parse_args()
    svc_factory = lambda: TriageService(dedup=DedupIndex())  # noqa: E731  a fresh dedup history for every set
    valid = load_jsonl(DATA / "tickets_valid.jsonl")
    train = load_jsonl(DATA / "tickets_train.jsonl.gz")
    random.Random(7).shuffle(train)
    svc_exp = svc_factory()
    report = {"note": "behaviour counters on real tickets (aggregates only); NOT accuracy; the mode is template (no model was called)",
              "detection_on_expectation_set": detection_on_expectation_set(svc_exp),
              "valid": measure(valid, svc_factory()), f"train_sample_{a.train_sample}": measure(train[: a.train_sample], svc_factory())}
    (METRICS / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
