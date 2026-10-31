"""Related complaints: other authors' complaints about the same accused player (24 h). Shown to the admin, never a duplicate."""
from __future__ import annotations

from app.classify import analyze
from app.dedup import DedupIndex, related_reason
from tests.conftest import make_req


def _req(reporter: str, ts: str, text: str = "Arman_Petrov оскорбил меня в чате", target: str = "t1"):
    return make_req(text, reporter_id=reporter, reported_id=target, created_at=ts)


def _run(idx: DedupIndex, req):
    return idx.check_and_add(req, analyze(req))


def test_other_authors_are_related_not_duplicates():
    idx = DedupIndex()
    assert _run(idx, _req("a", "2026-10-01T10:00:00")).related is None
    _run(idx, _req("b", "2026-10-01T11:00:00", "Arman_Petrov убил меня без причины у банка"))
    r = _run(idx, _req("c", "2026-10-01T12:00:00"))
    assert r.is_duplicate is False
    assert r.related == {"complaints": 2, "other_reporters": 2, "categories": {"complaint.insult": 1, "complaint.dm": 1}, "hours": 24}
    assert "2 других" in related_reason(r.related) and "не дубликат" in related_reason(r.related)


def test_own_tickets_other_targets_and_old_ones_are_not_related():
    idx = DedupIndex()
    _run(idx, _req("a", "2026-10-01T10:00:00"))
    assert _run(idx, _req("a", "2026-10-01T10:30:00", "Arman_Petrov снова оскорбил в чате у банка")).related is None  # same author
    assert _run(idx, _req("b", "2026-10-01T10:40:00", target="t2")).related is None  # another accused player
    assert _run(idx, _req("c", "2026-10-03T10:00:00")).related is None  # older than 24 h
    assert related_reason(None) is None


def test_reports_and_anonymous_tickets_are_not_counted():
    idx = DedupIndex()
    rep = make_req("меня дмят у спавна", kind="report", reporter_id="a", reported_id="t1", created_at="2026-10-01T10:00:00")
    _run(idx, rep)
    _run(idx, make_req("Arman_Petrov оскорбил", reported_id="t1", created_at="2026-10-01T10:05:00"))  # no author
    assert _run(idx, _req("b", "2026-10-01T10:10:00")).related is None


def test_related_history_survives_restart(tmp_path):
    db = tmp_path / "d.db"
    _run(DedupIndex(db), _req("a", "2026-10-01T10:00:00"))
    r = _run(DedupIndex(db), _req("b", "2026-10-01T10:20:00"))
    assert r.related and r.related["other_reporters"] == 1


def test_pipeline_lists_related_complaints_and_similarity(service):
    service.triage(_req("a", "2026-10-01T10:00:00"))
    resp, dbg = service.triage(_req("b", "2026-10-01T10:05:00"))
    assert resp.is_spam_or_duplicate is False
    assert any(x.startswith("Связанные жалобы") for x in resp.reasons)
    assert dbg.signals["related_complaints"]["other_reporters"] == 1
    resp, dbg = service.triage(_req("b", "2026-10-01T10:15:00"))  # the same author again: a duplicate with a similarity score
    assert resp.is_spam_or_duplicate is True and dbg.signals["duplicate_similarity"] == 1.0
    assert any("сходство" in x for x in resp.reasons)
