"""Build CANDIDATES for the local knowledge base from recurring questions and the administrators' replies in TRAIN.

    python scripts/build_faq_candidates.py [--min-reporters 5] [--min-replies 3]

What it does (reproducible, local, from the permitted files only):
  1. takes `report` tickets of data/handout/tickets_train.jsonl.gz that the service itself classifies as report.question
     (VALID is never read: it is the held-out set, and nothing here may peek at the outcomes of new tickets);
  2. groups them by a normalised "question signature" (the content words left after removing stop words and names);
  3. keeps a group only if it was asked by several DIFFERENT reporters and several administrator replies exist;
  4. cleans every reply: removes nicknames, numbers, links, commands, contacts; REJECTS a reply that contains a promise or a
     completed action (the same guard as for drafts), a personal outcome ("ваш бан", "вам выдан"), an instruction we cannot
     verify (slash commands, links, "нажмите"), or a time/amount;
  5. writes data/local/faq_candidates.json: every candidate with its provenance (ticket ids, number of reporters/replies, how
     many replies had to be rejected and why) and `reviewed: false`.

What it does NOT do: it never writes data/kb/faq.json. A candidate is a SUGGESTION for a human. The service uses a FAQ entry to
remove the human handoff only when a person has copied it into data/kb/faq.json, written a `source`, and set `reviewed: true`
(app/faq.py). An administrator's reply is evidence of what one administrator said once, not a universal truth, and game facts
change; therefore nothing is copied automatically and the output file stays in git-ignored data/local/.
"""
from __future__ import annotations

import argparse
import collections
import json
import re

from _common import DATA, LOCAL, load_jsonl, player_text, admin_text
from app.adapter import record_to_request
from app.dedup import DedupIndex
from app.guard import find_promises, TRANSFER_CLAIM, check_free_text
from app.pipeline import TriageService
from app.classify import QUESTION
from app.textutil import clean_for_matching

STOP = set("как что где когда можно ли это в на и я не по за с к у из от до для или а но то ну да нет мне мой моя меня вас вам вы ты тебе есть бы же уже еще ещё "
           "the is a an to of in on and or how what where can i do you my me".split())
PERSONAL = re.compile(r"(?<![\w])(?:ваш\w*|вам|вас|вами|тебе|твой|твоя|your|you)(?![\w])", re.I)
UNVERIFIED = re.compile(r"/[a-z]{2,}|https?://|www\.|t\.me|@\w{3,}|нажм\w+|перейд\w+|откро\w+|введ\w+|напиш\w+\s+(?:в|на)|обрат\w+\s+(?:в|к)", re.I)
NUMBER = re.compile(r"\d")
# the administrator describes what HE is doing or asks to wait: that is an action of one person, not knowledge about the game
ACKNOWLEDGEMENT = re.compile(r"слеж|спеш|смотр|провер|разбер|иду\b|выдвига|ожида|подожд|приним|принят|рассмотр|в\s+процессе|занимаюсь|уже\s+на\s+мест", re.I)


def signature(clean: str) -> str:
    words = [w for w in clean.split() if w not in STOP and len(w) > 2 and not w.isdigit()]
    return " ".join(sorted(set(words))[:6])


def reply_problem(text: str) -> str | None:
    """Why an administrator reply may NOT become a candidate answer (None = acceptable as a SUGGESTION)."""
    if not text.strip() or len(text) > 600:
        return "empty or too long"
    if find_promises(text):
        return "promise or completed action"
    if TRANSFER_CLAIM.search(text):
        return "claims a transfer"
    if UNVERIFIED.search(text):
        return "link, command or instruction that cannot be verified"
    if PERSONAL.search(text):
        return "personal outcome (addressed to one player)"
    if NUMBER.search(text):
        return "contains a number (amount, time, level, rule)"
    if check_free_text(text):
        return "sanction/refund topic without negation, deadline or procedure"
    if ACKNOWLEDGEMENT.search(text):
        return "an administrator's acknowledgement of his own action, not knowledge"
    if len(text.split()) < 6:
        return "too short to be an explanation"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reporters", type=int, default=5)
    ap.add_argument("--min-replies", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="read only the first N train tickets (for a quick run)")
    a = ap.parse_args()
    svc = TriageService(dedup=DedupIndex())
    train = load_jsonl(DATA / "tickets_train.jsonl.gz", a.limit or None)  # VALID IS NOT READ
    groups: dict[str, dict] = collections.defaultdict(lambda: {"tickets": [], "reporters": set(), "questions": collections.Counter(), "replies": []})
    n_q = 0
    for r in train:
        if r["kind"] != "report":
            continue
        resp, _ = svc.triage(record_to_request(r))
        if resp.category != "report.question" or resp.draft_reply.cited or resp.draft_reply.terms:
            continue  # already answered from verified rules/terms
        text = player_text(r)
        if "?" not in text and not QUESTION.search(clean_for_matching(text)):
            continue  # a bare keyword ("277 спам", "помеха") is not a question about the game
        n_q += 1
        sig = signature(clean_for_matching(text))
        if not sig:
            continue
        g = groups[sig]
        g["tickets"].append(r["ticket_id"])
        g["reporters"].add(r["reporter_hash"])
        g["questions"][player_text(r)[:160]] += 1
        reply = admin_text(r).strip()
        if reply:
            g["replies"].append(reply)
    out = []
    rejected = collections.Counter()
    for sig, g in groups.items():
        if len(g["reporters"]) < a.min_reporters:
            continue
        ok, why = [], collections.Counter()
        for rep in g["replies"]:
            p = reply_problem(rep)
            (why.update([p]) if p else ok.append(rep))
        rejected.update(why)
        if len(ok) < a.min_replies:
            continue
        common = collections.Counter(clean_for_matching(x) for x in ok).most_common(1)[0]
        out.append({
            "id": "cand-" + re.sub(r"\W+", "-", sig)[:40],
            "question_signature": sig,
            "example_questions": [q for q, _ in g["questions"].most_common(3)],
            "candidate_answer_ru": next(x for x in ok if clean_for_matching(x) == common[0]),
            "same_reply_count": common[1],
            "provenance": {"source": "administrator replies in tickets_train.jsonl.gz", "tickets": len(g["tickets"]),
                           "distinct_reporters": len(g["reporters"]), "usable_replies": len(ok), "rejected_replies": dict(why),
                           "example_ticket_ids": g["tickets"][:5]},
            "reviewed": False,
            "note": "SUGGESTION ONLY. A person must verify it against the game, write `source`, translate it into the other languages "
                    "and set reviewed:true in data/kb/faq.json before the service may use it without a human.",
        })
    out.sort(key=lambda c: -c["provenance"]["distinct_reporters"])
    (LOCAL / "faq_candidates.json").write_text(json.dumps({"entries": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"train_tickets_read": len(train), "question_tickets_without_rule_answer": n_q, "signatures": len(groups),
               "candidates": len(out), "replies_rejected_by_reason": dict(rejected)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (DATA.parent.parent / "docs" / "metrics" / "faq_candidates_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
