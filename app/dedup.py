"""Spam and duplicate detection, as far as the input data allows.

Dedup needs WHO sent the ticket (reporter id) and WHEN (timestamp). The /triage contract has neither, so a plain POST
/triage without them never reports duplicates; the batch pipeline and the admin queue pass `reporter_id`/`created_at`
when they exist. Texts of different people are never merged: the author is part of the key, and so is the accused player.

What makes two tickets "the same problem" (policy, documented in docs/ARCHITECTURE.md):
  * same author, same accused player, within the time window (60 min);
  * and NOT contradicting event details: two different places (bank / hospital), two servers or two clock times mean two events;
  * confident duplicate: almost the same text (Jaccard >= 0.8), or the same category with a shared detail (same place / server /
    time) and some text overlap, or the same category with a clearly paraphrased text (Jaccard >= 0.3);
  * "possible duplicate": the same category and no contradiction, but not enough shared content: reported in `reasons` only,
    `is_spam_or_duplicate` stays false (the category alone does not prove one problem).

Related complaints (a different thing): complaints about the same accused player from OTHER authors in the last 24 hours. They are never
duplicates (several victims of one offender are several complaints), but an admin should see them together: they are listed in `reasons`.

History survives a restart when the index is given a SQLite path (the HTTP service does that); it is bounded in memory
(per author and in the number of authors) and in time (entries older than twice the window are dropped).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .classify import Analysis, EVIDENCE_WORDS
from .schemas import TriageRequest

WINDOW_MINUTES = 60
KEEP_MINUTES = 2 * WINDOW_MINUTES
JACCARD_MIN = 0.8
PARAPHRASE_MIN = 0.3
SHARED_DETAIL_MIN = 0.12
MAX_PER_REPORTER = 40
MAX_REPORTERS = 50_000
# related complaints: other authors' complaints about the SAME accused player (never merged, only shown to the admin)
RELATED_HOURS = 24
MAX_PER_TARGET = 200
MAX_TARGETS = 100_000

# --- spam ---------------------------------------------------------------------------------------------------------------
# A promotion sent BY the player (links, "заходи на ...", giveaways). A bare word such as "казино" is not spam: it is the topic
# of an advertising complaint ("здесь рекламируют казино, проверьте").
_AD_PROMO = re.compile(
    r"t\.me/|discord\.gg|подписывайся|подпишись|подпис\w+\s+на|заходи(?:те)?\s+(?:на|в)|переходи(?:те)?\s+по|бесплатн\w+\s+(?:донат|вирты|коины)|"
    r"раздач\w+|заработ\w+\s+(?:деньги|в\s+интернете)|1xbet|промокод|ставки\s+на\s+спорт|играй(?:те)?\s+в\s+казино|казино\s+(?:онлайн|бонус)", re.I)
# the sender DESCRIBES or asks about advertising rather than doing it (a report about an ad, a quotation, a question)
_AD_ABOUT = re.compile(
    r"рекламир\w+|рекламу|реклама\s+(?:в|на|от)|пиарит\w*|пиарят|спамит\w*|спамят|проверьте|жалоб\w*|жалуюсь|кто[- ]то|можно\s+ли|запрещен\w*|разрешен\w*|"
    r"пишет|написал\w*|предлага\w+|присылает|скидывает|рассылает|\?|ads?\s+(?:in|on)|reporting|advertising\s+(?:in|on)", re.I)
_FLOOD_CHAR = re.compile(r"(.)\1{11,}")


def parse_ts(value) -> Optional[datetime]:
    """An ISO-8601 string -> naive datetime; anything else (a number, an object, a list, a broken string) -> None. Never raises: a bad
    date in one record must not stop a whole batch (the record itself is validated separately)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _epoch(dt: Optional[datetime]) -> Optional[float]:
    return None if dt is None else (dt - datetime(1970, 1, 1)).total_seconds()


def spam_reason(a: Analysis) -> Optional[str]:
    """Heuristic spam only (floods, repeated words, promotions in the report channel); a COMPLAINT about an ad is not spam."""
    t = a.raw_text
    if len(t) >= 12 and _FLOOD_CHAR.search(t):
        return "Спам: сообщение состоит из повторяющихся символов"
    words = a.clean.split()
    if len(words) >= 8:
        top = max((words.count(w) for w in set(words)), default=0)
        if top / len(words) >= 0.7:
            return "Спам: одно и то же слово повторяется в сообщении"
    if a.kind == "report" and _AD_PROMO.search(t) and not _AD_ABOUT.search(t):
        return "Спам: реклама или раздача в канале репортов"
    return None


# --- event details ------------------------------------------------------------------------------------------------------
_PLACE = re.compile(
    r"(?<![\w])(банк\w*|больниц\w*|госпитал\w*|полиц\w*|мвд|фсб|мэри\w*|рын[о]?к\w*|автосалон\w*|аэропорт\w*|казино|клуб\w*|завод\w*|шахт\w*|"
    r"ферм\w*|гараж\w*|заправк\w*|вокзал\w*|школ\w*|мотел\w*|отел\w*|магазин\w*|склад\w*|пляж\w*|лес(?:у|ом|а|е)?|тюрьм\w*|порт(?:у|а|е)?|"
    r"зз|гз|зелен\w*\s+зон\w*|bank|hospital|police|market|airport|school|casino|garage|club|park|beach|forest|prison|mall)(?![\w])", re.I)
_TIME = re.compile(r"(?<![\d.])([01]?\d|2[0-3])[:.]([0-5]\d)(?![\d])")


def details(a: Analysis) -> frozenset[str]:
    """Event details that distinguish one event from another: places, server, clock times."""
    out: set[str] = set()
    for m in _PLACE.finditer(a.clean):
        w = m.group(1).lower().replace("ё", "е")
        out.add("p:" + ("зз" if w in {"зз", "гз"} or w.startswith("зелен") else w[:4]))
    if a.server:
        out.add(f"s:{a.server}")
    for m in _TIME.finditer(a.text):
        out.add(f"t:{int(m.group(1)):02d}:{m.group(2)}")
    return frozenset(out)


def _kind(d: frozenset[str], prefix: str) -> frozenset[str]:
    return frozenset(x for x in d if x.startswith(prefix))


def contradict(d1: frozenset[str], d2: frozenset[str]) -> bool:
    """Two tickets name different places / servers / times: they describe different events."""
    for prefix in ("p:", "s:", "t:"):
        a, b = _kind(d1, prefix), _kind(d2, prefix)
        if a and b and not (a & b):
            return True
    return False


@dataclass
class DupResult:
    is_duplicate: bool
    reason: Optional[str] = None
    suspected: bool = False  # same author/target/category in the window, but not enough evidence of ONE event
    similarity: Optional[float] = None  # token Jaccard similarity with the matched earlier ticket
    related: Optional[dict] = None  # other authors' complaints about the same accused player in the last RELATED_HOURS


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Russian plural form: 1 жалоба, 2 жалобы, 5 жалоб (11-14 take the «many» form)."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def related_reason(rel: Optional[dict]) -> Optional[str]:
    if not rel or not rel.get("other_reporters"):
        return None
    cats = ", ".join(f"{c.split('.')[-1]}×{n}" for c, n in rel["categories"].items())
    n, m = rel["complaints"], rel["other_reporters"]
    return (f"Связанные жалобы: на этого игрока за {RELATED_HOURS} ч уже есть {n} {_plural(n, 'жалоба', 'жалобы', 'жалоб')} от {m} "
            f"{_plural(m, 'другого автора', 'других авторов', 'других авторов')} ({cats}). Это не дубликат: каждую жалобу рассматривают отдельно, "
            "но стоит посмотреть их вместе")


def _tokens(clean: str) -> frozenset[str]:
    return frozenset(w for w in clean.split() if len(w) > 1 or w.isdigit())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class _Seen:
    ts: Optional[float]
    target: str
    toks: frozenset[str]
    cat: str
    det: frozenset[str]


class DedupIndex:
    """Bounded in-memory index, optionally mirrored to SQLite (`path`) so that the history survives a restart."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._seen: "OrderedDict[str, list[_Seen]]" = OrderedDict()
        self._targets: "OrderedDict[str, list[tuple[float, str, str]]]" = OrderedDict()  # target -> [(ts, reporter, category)]
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool
        self._path = Path(path) if path else None
        self._conn: Optional[sqlite3.Connection] = None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._db() as c:
                c.execute("CREATE TABLE IF NOT EXISTS dedup_seen (id INTEGER PRIMARY KEY AUTOINCREMENT, reporter TEXT NOT NULL, ts REAL,"
                          " target TEXT NOT NULL, category TEXT NOT NULL, tokens TEXT NOT NULL, details TEXT NOT NULL)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_dedup_reporter ON dedup_seen(reporter)")
                c.execute("CREATE TABLE IF NOT EXISTS target_seen (id INTEGER PRIMARY KEY AUTOINCREMENT, target TEXT NOT NULL, ts REAL NOT NULL,"
                          " reporter TEXT NOT NULL, category TEXT NOT NULL)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_target_seen ON target_seen(target)")
            self._load()

    # ---- persistence -------------------------------------------------------------------------------------------------
    def _db(self):
        """A long-lived connection: opening one per ticket cost ~15 ms of the p95. Every use is under `self._lock` (or in __init__)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._path, timeout=10, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        conn = self._conn

        class _Ctx:
            def __enter__(s):
                return conn

            def __exit__(s, *exc):
                conn.commit()

        return _Ctx()

    def _load(self) -> None:
        with self._db() as c:
            newest = c.execute("SELECT MAX(ts) FROM target_seen").fetchone()[0]
            if newest is not None:
                c.execute("DELETE FROM target_seen WHERE ts < ?", (newest - RELATED_HOURS * 3600,))
                for target, ts, reporter, cat in c.execute("SELECT target, ts, reporter, category FROM target_seen ORDER BY id").fetchall():
                    self._targets.setdefault(target, []).append((ts, reporter, cat))
            rows = c.execute("SELECT reporter, ts, target, category, tokens, details FROM dedup_seen ORDER BY id").fetchall()
            for reporter, ts, target, cat, toks, det in rows:
                self._seen.setdefault(reporter, []).append(_Seen(ts, target, frozenset(json.loads(toks)), cat, frozenset(json.loads(det))))
            for reporter in list(self._seen):
                self._prune(reporter)
            while len(self._seen) > MAX_REPORTERS:
                old, _ = self._seen.popitem(last=False)
                c.execute("DELETE FROM dedup_seen WHERE reporter = ?", (old,))
            # keep the table in step with the pruned memory
            keep = sum(len(v) for v in self._seen.values())
            total = c.execute("SELECT COUNT(*) FROM dedup_seen").fetchone()[0]
            if keep != total:
                c.execute("DELETE FROM dedup_seen")
                c.executemany("INSERT INTO dedup_seen (reporter, ts, target, category, tokens, details) VALUES (?,?,?,?,?,?)",
                              [(r, s.ts, s.target, s.cat, json.dumps(sorted(s.toks), ensure_ascii=False), json.dumps(sorted(s.det), ensure_ascii=False))
                               for r, lst in self._seen.items() for s in lst])

    def _prune(self, reporter: str) -> None:
        """Bounded memory: drop entries more than KEEP_MINUTES older than the newest one, and cap the list."""
        lst = self._seen.get(reporter, [])
        stamped = [s.ts for s in lst if s.ts is not None]
        if stamped:
            newest = max(stamped)
            lst = [s for s in lst if s.ts is None or newest - s.ts <= KEEP_MINUTES * 60]
        self._seen[reporter] = lst[-MAX_PER_REPORTER:]

    def _persist(self, reporter: str, s: _Seen, pruned: bool) -> None:
        if not self._path:
            return
        with self._db() as c:
            c.execute("INSERT INTO dedup_seen (reporter, ts, target, category, tokens, details) VALUES (?,?,?,?,?,?)",
                      (reporter, s.ts, s.target, s.cat, json.dumps(sorted(s.toks), ensure_ascii=False), json.dumps(sorted(s.det), ensure_ascii=False)))
            if pruned:  # keep this author's rows in step with the pruned memory (oldest rows go first)
                ids = c.execute("SELECT id FROM dedup_seen WHERE reporter = ? ORDER BY id", (reporter,)).fetchall()
                extra = len(ids) - len(self._seen[reporter])
                if extra > 0:
                    c.executemany("DELETE FROM dedup_seen WHERE id = ?", ids[:extra])

    # ---- checking ----------------------------------------------------------------------------------------------------
    def check_and_add(self, req: TriageRequest, a: Analysis) -> DupResult:
        with self._lock:
            return self._check_and_add(req, a)

    def _related(self, target: str, reporter: str, ts: Optional[float], category: str) -> Optional[dict]:
        """Other authors' complaints about this accused player in the last RELATED_HOURS; then remember this one."""
        if not target or ts is None:
            return None
        lst = [x for x in self._targets.get(target, []) if ts - x[0] <= RELATED_HOURS * 3600]
        before = [x for x in lst if x[0] <= ts and x[1] != reporter]
        out = None
        if before:
            cats: dict[str, int] = {}
            for _, _, c in before:
                cats[c] = cats.get(c, 0) + 1
            out = {"complaints": len(before), "other_reporters": len({x[1] for x in before}),
                   "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])), "hours": RELATED_HOURS}
        lst.append((ts, reporter, category))
        self._targets[target] = lst[-MAX_PER_TARGET:]
        self._targets.move_to_end(target)
        while len(self._targets) > MAX_TARGETS:
            self._targets.popitem(last=False)
        if self._path:
            with self._db() as c:
                c.execute("INSERT INTO target_seen (target, ts, reporter, category) VALUES (?,?,?,?)", (target, ts, reporter, category))
        return out

    def _check_and_add(self, req: TriageRequest, a: Analysis) -> DupResult:
        best = self._check_dup(req, a)
        if req.kind == "complaint" and req.reporter_id:
            target = req.reported_id or (req.reported_nick or a.nickname or "").lower()
            best.related = self._related(target, req.reporter_id, _epoch(parse_ts(req.created_at)), a.category)
        return best

    def _check_dup(self, req: TriageRequest, a: Analysis) -> DupResult:
        reporter = req.reporter_id
        if not reporter:
            return DupResult(False)  # no author -> cannot tell a duplicate from two people saying the same thing
        target = req.reported_id or (req.reported_nick or a.nickname or "")
        ts = _epoch(parse_ts(req.created_at))
        toks = _tokens(a.clean)
        det = details(a)
        best = DupResult(False)
        for old in self._seen.get(reporter, []):
            if old.target != target:
                continue  # never merge tickets about different players
            sim = _jaccard(toks, old.toks)
            if ts is None or old.ts is None:
                if toks and toks == old.toks:
                    best = DupResult(True, "Повтор: тот же автор, адресат и текст; время подачи неизвестно, поэтому окно в 60 минут не проверено",
                                     similarity=1.0)
                    break
                continue
            minutes = abs(ts - old.ts) / 60.0
            if minutes > WINDOW_MINUTES or contradict(det, old.det):
                continue  # outside the window, or a different place/server/time: another event
            shared = bool(det & old.det)
            if sim >= JACCARD_MIN:
                best = DupResult(True, f"Дубликат: тот же автор и адресат, почти тот же текст (сходство {sim:.2f}), интервал {minutes:.0f} мин",
                                 similarity=round(sim, 3))
                break
            if target and old.cat == a.category:
                if sim >= PARAPHRASE_MIN or (shared and sim >= SHARED_DETAIL_MIN):
                    why = "общие детали события (место, сервер или время)" if shared else "схожий пересказ"
                    best = DupResult(True, f"Дубликат: тот же автор, адресат и категория ({a.category}), {why}, сходство текста {sim:.2f}, "
                                           f"интервал {minutes:.0f} мин", similarity=round(sim, 3))
                    break
                if not best.suspected:
                    best = DupResult(False, f"Возможный повтор (не подтверждён): тот же автор, адресат и категория ({a.category}) за {minutes:.0f} мин, "
                                            f"но общего содержания мало (сходство {sim:.2f}); категория сама по себе не доказывает одну и ту же проблему",
                                     suspected=True, similarity=round(sim, 3))
        entry = _Seen(ts, target, toks, a.category, det)
        self._seen.setdefault(reporter, []).append(entry)
        self._seen.move_to_end(reporter)
        n_before = len(self._seen[reporter])
        self._prune(reporter)
        pruned = len(self._seen[reporter]) < n_before
        while len(self._seen) > MAX_REPORTERS:
            old_reporter, _ = self._seen.popitem(last=False)
            if self._path:
                with self._db() as c:
                    c.execute("DELETE FROM dedup_seen WHERE reporter = ?", (old_reporter,))
        self._persist(reporter, entry, pruned)
        return best

    def size(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._seen.values())


def has_evidence_words(text: str) -> bool:
    return bool(EVIDENCE_WORDS.search(text))
