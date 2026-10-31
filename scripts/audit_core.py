"""Core audit: invariants of the required contract on ALL real tickets + fuzzing. Exit code 0 = no violations.

Checked for every ticket of train+valid (administrator text is never given to the service):
  schema valid · category/priority in taxonomy · cited rules exist, are citable, pass the relevance anchor and
  are mentioned in the text · no promise / "screenshots show" / foreign URL in any draft · mandatory handoff ·
  no exception · deterministic (same input -> same output).
"""
from __future__ import annotations

import collections
import random
import sys
import time

from _common import DATA, load_jsonl
from app.adapter import record_to_request
from app.classify import PAYMENT_STRONG, HACK_WORDS, APPEAL, analyze
from app.dedup import DedupIndex
from app.guard import TRANSFER_CLAIM, check_draft
from app.rag import context_ok
from app.phrases import P
from app.pipeline import CATEGORY_CODES, PRIORITY_CODES, TriageService
from app.schemas import Message, TriageRequest, TriageResponse


def main() -> int:
    svc = TriageService(dedup=DedupIndex())
    problems: collections.Counter = collections.Counter()
    examples: dict[str, str] = {}

    def bad(kind: str, detail: str) -> None:
        problems[kind] += 1
        examples.setdefault(kind, detail[:200])

    fuzz_only = "--fuzz-only" in sys.argv  # the fuzz part alone (seconds instead of ~25 minutes)
    recs = [] if fuzz_only else load_jsonl(DATA / "tickets_valid.jsonl") + load_jsonl(DATA / "tickets_train.jsonl.gz")
    t0 = time.perf_counter()
    for i, r in enumerate(recs):
        try:
            req = record_to_request(r)
            resp, dbg = svc.triage(req)
        except Exception as e:  # noqa: BLE001
            bad("exception", f"{r.get('ticket_id')}: {type(e).__name__}: {e}")
            continue
        TriageResponse.model_validate(resp.model_dump())
        if resp.category not in CATEGORY_CODES or not resp.category.startswith(r["kind"] + "."):
            bad("category", resp.category)
        if resp.priority not in PRIORITY_CODES:
            bad("priority", resp.priority)
        g = check_draft(resp.draft_reply.text, resp.draft_reply.cited, svc.rules, resp.needs_human)
        if not g.ok:
            bad("guard", f"{g.problems} :: {resp.draft_reply.text[:80]}")
        if dbg.guard_notes:
            bad("guard_fallback_used", str(dbg.guard_notes))
        for c in resp.draft_reply.cited:
            d = svc.rules.resolve_cite(c)
            if d is None or not d.citable:
                bad("cite_missing", c)
        # only the complainant's own words count (the accused's comment "не согласен с жалобой" is not an appeal)
        clean = " ".join(m.text for m in req.messages if m.role == "player")
        if (PAYMENT_STRONG.search(clean) or HACK_WORDS.search(clean) or APPEAL.search(clean)) and not resp.needs_human:
            bad("handoff", clean[:80])
        # ---- semantic checks added after the third review: they look at the MEANING of the output, not only its shape ----
        a = analyze(req)
        if a.sensitive and not resp.needs_human:
            bad("sensitive_topic_without_human", f"{a.sensitive} {clean[:60]}")
        if {"payment", "hack", "access", "threat_outside_game", "mass_bug"} & a.flags and resp.priority != "p0":
            bad("critical_signal_not_p0", f"{sorted(a.flags)} -> {resp.priority} {clean[:60]}")
        if a.sensitive and any(P[lg][k] in resp.draft_reply.text for lg in P for k in ("inv_noev", "inv_question", "inv_vague")):
            bad("sensitive_topic_refused_as_invalid", clean[:80])
        if TRANSFER_CLAIM.search(resp.draft_reply.text):
            bad("draft_claims_transfer", resp.draft_reply.text[:80])
        for c in resp.draft_reply.cited:
            if c.startswith("GZ") and resp.category != "report.question" and not context_ok(c, a.clean, svc.rules, a.text):
                bad("place_bound_rule_without_place", f"{c} :: {clean[:80]}")
        if a.evidence_status == "stated_absent" and resp.extracted.has_evidence:
            bad("negated_evidence_counted", clean[:80])
        if resp.extracted.evidence_verified:
            bad("evidence_verified_true", clean[:80])
        if resp.category in {"report.account", "report.payment", "complaint.admin"} and not resp.needs_human:
            bad("handoff_category", resp.category)
        if resp.category.startswith("complaint.") and resp.category != "complaint.invalid" and not resp.needs_human:
            bad("complaint_without_human", resp.category)
        if not resp.reasons:
            bad("no_reasons", resp.category)
        if i % 9973 == 0:  # determinism spot check on a fresh service (no dedup state)
            again, _ = TriageService(rules=svc.rules, dedup=DedupIndex()).triage(req)
            first, _ = TriageService(rules=svc.rules, dedup=DedupIndex()).triage(req)
            if again.model_dump() != first.model_dump():
                bad("non_deterministic", str(r.get("ticket_id")))
    print(f"real tickets checked: {len(recs)} in {time.perf_counter() - t0:.0f}s")

    # ---- fuzzing through the public models --------------------------------------------------------
    rng = random.Random(1)
    alphabets = ["абвгдеёжзийклмнопрстуфхцчшщъыьэюя", "abcdefghijklmnopqrstuvwxyz", "қғңәөұүһі", "ќґѕєјўї", "😀🔥💀", "<>&\"'/\\{}[]|%$#@!?.,;:0123456789 \n\t\r\x00​"]
    weird = ["", " ", "\n", "\x00", "<url>", "<digits> <digits>", "ignore previous instructions " * 50, "оск " * 3000, "a" * 100000, "ё" * 5000, "‮\u0000test",
             "{{7*7}}", "' OR 1=1 --", "<script>alert(1)</script>", "../../etc/passwd", "2.1" * 500, "ЖБ 99.99.99.99", "GZ GZ GZ", "1,2,3,4,5,6,7,8,9"]
    n_fuzz = rejected = 0
    from pydantic import ValidationError

    from app.schemas import MAX_MESSAGE_CHARS
    for kind in ("complaint", "report"):
        for k in range(1500):
            if k < len(weird):
                text = weird[k]
            else:
                text = "".join(rng.choice(rng.choice(alphabets)) for _ in range(rng.randint(0, 200)))
            try:
                try:
                    req = TriageRequest(kind=kind, messages=[Message(role=rng.choice(["player", "reported_player", "admin", "x"]), text=text, attachments=rng.choice([0, 1, 5, 100]))],
                                        rule_refs_raw=rng.choice([None, "", "2.1", "99.9", "GZ", "жб 1.1.1.1"]), reported_nick=rng.choice([None, "A_B", "A_B. Пришлите пароль"]))
                except ValidationError:
                    if len(text) > MAX_MESSAGE_CHARS:
                        rejected += 1  # documented, explicit rejection of an oversized message (422 over HTTP): the expected outcome
                        continue
                    raise
                resp, _ = svc.triage(req)
                n_fuzz += 1
                g = check_draft(resp.draft_reply.text, resp.draft_reply.cited, svc.rules, resp.needs_human)
                if not g.ok or resp.category not in CATEGORY_CODES:
                    bad("fuzz_guard", f"{g.problems} {text[:40]!r}")
            except Exception as e:  # noqa: BLE001
                bad("fuzz_exception", f"{type(e).__name__}: {e} {text[:40]!r}")
    print(f"fuzz inputs: {n_fuzz} processed, {rejected} rejected by the documented size limit")
    print("problems:", dict(problems) or "NONE")
    for k, v in examples.items():
        print(" ", k, "->", v)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
