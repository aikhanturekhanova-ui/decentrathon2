"""Optional Telegram interface (bonus). Off unless TELEGRAM_ENABLED=1 and TELEGRAM_BOT_TOKEN is set.

It is only another front end to the same TriageService + SQLite queue. It is a DEMO bot: it says so,
never claims that a Grand Mobile employee received the message, and stores the ticket in the local
admin queue. Use invented messages for public demos. Run:  python -m app.telegram_bot
"""
from __future__ import annotations

import sys
import time

import httpx

from .config import load_settings
from .pipeline import TriageService
from .schemas import Message, TriageRequest, TriageResponse
from .store import Store

DISCLAIMER = "Это демонстрационный бот хакатона, а не служба поддержки Grand Mobile: сотрудник игры ваше сообщение не получил."


def build_reply(resp: TriageResponse, queue_id: int) -> str:
    status = (
        "Статус: нужна проверка человеком. Сообщение сохранено в очереди администратора (локально)."
        if resp.needs_human
        else "Статус: человек не требуется для этого уточняющего ответа. Сообщение сохранено в локальной очереди."
    )
    return f"{DISCLAIMER}\n\nЧерновик ответа:\n{resp.draft_reply.text}\n\n{status}\nКатегория: {resp.category}, приоритет: {resp.priority}. Номер в очереди: {queue_id}."


def handle_text(svc: TriageService, store: Store, text: str, chat_id: int | str) -> str:
    req = TriageRequest(kind="report", messages=[Message(role="player", text=text[:2000])], reporter_id=f"tg:{chat_id}")
    resp, _ = svc.triage(req)
    qid = store.add(req.model_dump(), resp.model_dump(), source="telegram")
    return build_reply(resp, qid)


def run() -> int:
    s = load_settings()
    if not (s.telegram_enabled and s.telegram_token):
        print("Telegram bot is disabled: set TELEGRAM_ENABLED=1 and TELEGRAM_BOT_TOKEN in .env", file=sys.stderr)
        return 2
    svc, store = TriageService(s), Store(s.db_path)
    api = f"https://api.telegram.org/bot{s.telegram_token}"
    offset = 0
    with httpx.Client(timeout=40) as c:
        print("bot started (long polling). Ctrl+C to stop.")
        while True:
            try:
                updates = c.get(f"{api}/getUpdates", params={"timeout": 30, "offset": offset}).json().get("result", [])
            except httpx.HTTPError as e:
                print("poll error:", type(e).__name__, file=sys.stderr)
                time.sleep(3)
                continue
            for u in updates:
                offset = u["update_id"] + 1
                m = u.get("message") or {}
                if not m.get("text"):
                    continue
                if m["text"].startswith("/start"):
                    reply = DISCLAIMER + "\nНапишите выдуманное обращение, и я покажу, как его разберёт сервис."
                else:
                    reply = handle_text(svc, store, m["text"], m["chat"]["id"])
                c.post(f"{api}/sendMessage", json={"chat_id": m["chat"]["id"], "text": reply})


if __name__ == "__main__":
    raise SystemExit(run())
