"""SQLite storage: admin queue + edits kept as feedback. Local only (the file is git-ignored).

Nothing here talks to the game. 'approve' means "approved locally": the text is stored and marked,
and the UI says explicitly that nothing was delivered to the player.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  external_id TEXT,
  source TEXT NOT NULL DEFAULT 'api',
  kind TEXT NOT NULL,
  category TEXT NOT NULL,
  priority TEXT NOT NULL,
  lang TEXT NOT NULL,
  needs_human INTEGER NOT NULL,
  is_dup INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'new',
  request_json TEXT NOT NULL,
  response_json TEXT NOT NULL,
  final_text TEXT,
  note TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_filters ON tickets(category, priority, lang, needs_human, status);
CREATE TABLE IF NOT EXISTS edits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticket_id INTEGER NOT NULL REFERENCES tickets(id),
  action TEXT NOT NULL,
  original_text TEXT NOT NULL,
  edited_text TEXT NOT NULL,
  category TEXT,
  lang TEXT,
  created_at REAL NOT NULL
);
-- the administrator's FINAL decision on a ticket (from the game's admin tool via POST /api/feedback, or from this panel):
-- these are human labels, the only trustworthy source of accuracy, and training data for `make train`
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticket_ref TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  player_text TEXT NOT NULL,
  predicted_category TEXT,
  predicted_priority TEXT,
  final_category TEXT NOT NULL,
  final_priority TEXT NOT NULL,
  final_reply TEXT,
  draft_rating INTEGER,
  lang TEXT,
  source TEXT NOT NULL,
  created_at REAL NOT NULL
);
"""

STATUSES = {"new", "edited", "approved_local", "escalated"}


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------------------------------
    def add(self, request: dict, response: dict, source: str = "api", external_id: Optional[str] = None) -> int:
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO tickets (external_id, source, kind, category, priority, lang, needs_human, is_dup, request_json,"
                " response_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (external_id, source, request["kind"], response["category"], response["priority"], response["lang"],
                 int(response["needs_human"]), int(response["is_spam_or_duplicate"]), json.dumps(request, ensure_ascii=False),
                 json.dumps(response, ensure_ascii=False), now, now),
            )
            return int(cur.lastrowid)

    @staticmethod
    def _row(r: sqlite3.Row, full: bool = False) -> dict[str, Any]:
        resp = json.loads(r["response_json"])
        d = {
            "id": r["id"], "external_id": r["external_id"], "source": r["source"], "kind": r["kind"], "category": r["category"],
            "priority": r["priority"], "lang": r["lang"], "needs_human": bool(r["needs_human"]), "is_dup": bool(r["is_dup"]),
            "status": r["status"], "created_at": r["created_at"], "updated_at": r["updated_at"],
            "preview": next((m.get("text", "") for m in json.loads(r["request_json"]).get("messages", []) if m.get("role") == "player"), "")[:140],
            "confidence": resp.get("confidence"),
        }
        if full:
            d.update(request=json.loads(r["request_json"]), response=resp, final_text=r["final_text"], note=r["note"])
        return d

    def list(self, category: str = "", lang: str = "", priority: str = "", needs_human: str = "", status: str = "",
             q: str = "", limit: int = 50, offset: int = 0) -> dict[str, Any]:
        where, args = [], []
        for col, val in (("category", category), ("lang", lang), ("priority", priority), ("status", status)):
            if val:
                where.append(f"{col} = ?")
                args.append(val)
        if needs_human in {"0", "1"}:
            where.append("needs_human = ?")
            args.append(int(needs_human))
        if q:
            where.append("request_json LIKE ?")
            args.append(f"%{q}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM tickets {clause}", args).fetchone()[0]
            rows = c.execute(
                f"SELECT * FROM tickets {clause} ORDER BY priority ASC, needs_human DESC, created_at DESC LIMIT ? OFFSET ?",
                [*args, max(1, min(limit, 200)), max(0, offset)],
            ).fetchall()
        return {"total": total, "items": [self._row(r) for r in rows]}

    def find_by_external_id(self, external_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM tickets WHERE external_id = ? ORDER BY id DESC LIMIT 1", (external_id,)).fetchone()
        return self._row(r, full=True) if r else None

    def get(self, ticket_id: int) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return self._row(r, full=True) if r else None

    def update_status(self, ticket_id: int, status: str, text: Optional[str] = None, note: Optional[str] = None) -> Optional[dict]:
        if status not in STATUSES:
            raise ValueError(f"bad status {status}")
        cur = self.get(ticket_id)
        if cur is None:
            return None
        original = cur["response"]["draft_reply"]["text"]
        now = time.time()
        with self._conn() as c:
            c.execute("UPDATE tickets SET status=?, final_text=COALESCE(?, final_text), note=COALESCE(?, note), updated_at=? WHERE id=?",
                      (status, text, note, now, ticket_id))
            last = c.execute("SELECT edited_text FROM edits WHERE ticket_id=? ORDER BY id DESC LIMIT 1", (ticket_id,)).fetchone()
            if text is not None and text != original and not (last and last["edited_text"] == text):
                # the edit is a training signal: original draft vs what the admin wanted to say
                # (saving the same text again, e.g. "save" then "confirm", is not a second signal)
                c.execute(
                    "INSERT INTO edits (ticket_id, action, original_text, edited_text, category, lang, created_at) VALUES (?,?,?,?,?,?,?)",
                    (ticket_id, status, original, text, cur["category"], cur["response"]["draft_reply"]["lang"], now),
                )
        return self.get(ticket_id)

    def stats(self) -> dict[str, Any]:
        with self._conn() as c:
            def grp(col: str) -> dict[str, int]:
                return {r[0]: r[1] for r in c.execute(f"SELECT {col}, COUNT(*) FROM tickets GROUP BY {col}")}
            return {
                "total": c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0],
                "by_category": grp("category"), "by_priority": grp("priority"), "by_lang": grp("lang"), "by_status": grp("status"),
                "needs_human": c.execute("SELECT COUNT(*) FROM tickets WHERE needs_human=1").fetchone()[0],
                "edits": c.execute("SELECT COUNT(*) FROM edits").fetchone()[0],
                "spam_or_duplicate": c.execute("SELECT COUNT(*) FROM tickets WHERE is_dup=1").fetchone()[0],
                "low_confidence": c.execute("SELECT COUNT(*) FROM tickets WHERE json_extract(response_json, '$.confidence') < 0.5").fetchone()[0],
                "open_critical": [
                    {"id": r[0], "category": r[1], "preview": (json.loads(r[2])["messages"][0].get("text") or "")[:90]}
                    for r in c.execute("SELECT id, category, request_json FROM tickets WHERE priority='p0' AND status IN ('new','edited') "
                                       "ORDER BY created_at DESC LIMIT 8")
                ],
            }

    def all_full(self, limit: int = 5000) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r, full=True) for r in rows]

    # ---------------------------------------------------------------------------------------
    def add_decision(self, d: dict[str, Any]) -> None:
        """Insert or replace the final decision for one ticket (the latest decision wins)."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO decisions (ticket_ref, kind, player_text, predicted_category, predicted_priority, final_category, final_priority,"
                " final_reply, draft_rating, lang, source, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(ticket_ref) DO UPDATE SET kind=excluded.kind, player_text=excluded.player_text,"
                " predicted_category=COALESCE(excluded.predicted_category, predicted_category),"
                " predicted_priority=COALESCE(excluded.predicted_priority, predicted_priority), final_category=excluded.final_category,"
                " final_priority=excluded.final_priority, final_reply=excluded.final_reply, draft_rating=excluded.draft_rating,"
                " lang=excluded.lang, source=excluded.source, created_at=excluded.created_at",
                (d["ticket_ref"], d["kind"], d["player_text"], d.get("predicted_category"), d.get("predicted_priority"), d["final_category"],
                 d["final_priority"], d.get("final_reply"), d.get("draft_rating"), d.get("lang"), d.get("source", "api"), time.time()),
            )

    def decisions(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM decisions ORDER BY id")]

    def decisions_summary(self) -> dict[str, Any]:
        """Live quality against the administrators' own decisions: category accuracy, priority macro-F1, draft ratings."""
        rows = [r for r in self.decisions() if r["predicted_category"]]
        out: dict[str, Any] = {"decisions": len(self.decisions()), "with_prediction": len(rows)}
        if not rows:
            return out
        cat_ok = sum(r["predicted_category"] == r["final_category"] for r in rows)
        out["category_accuracy"] = round(cat_ok / len(rows), 4)
        pri = [(r["final_priority"], r["predicted_priority"]) for r in rows if r["predicted_priority"]]
        if pri:
            labels = sorted({a for a, _ in pri} | {b for _, b in pri})
            f1s = []
            for lab in labels:
                tp = sum(a == lab and b == lab for a, b in pri)
                fp = sum(a != lab and b == lab for a, b in pri)
                fn = sum(a == lab and b != lab for a, b in pri)
                prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
                f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
            out["priority_macro_f1"] = round(sum(f1s) / len(f1s), 4)
            out["priority_accuracy"] = round(sum(a == b for a, b in pri) / len(pri), 4)
        confusion: dict[str, int] = {}
        for r in rows:
            if r["predicted_category"] != r["final_category"]:
                k = f"{r['predicted_category']} -> {r['final_category']}"
                confusion[k] = confusion.get(k, 0) + 1
        out["top_category_corrections"] = sorted(confusion.items(), key=lambda kv: -kv[1])[:10]
        ratings = [r["draft_rating"] for r in self.decisions() if r["draft_rating"]]
        if ratings:
            out["draft_rating_mean"] = round(sum(ratings) / len(ratings), 2)
            out["draft_ratings"] = len(ratings)
        by_lang: dict[str, list[int]] = {}
        for r in rows:
            by_lang.setdefault(r["lang"] or "und", []).append(int(r["predicted_category"] == r["final_category"]))
        out["category_accuracy_by_lang"] = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)} for k, v in by_lang.items()}
        return out

    def export_edits(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM edits ORDER BY id")]
