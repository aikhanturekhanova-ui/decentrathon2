from app.store import Store
from app.telegram_bot import DISCLAIMER, handle_text, run


def test_bot_reply_is_honest_and_queued(service, tmp_path):
    st = Store(tmp_path / "tg.db")
    reply = handle_text(service, st, "взломали аккаунт, пропала машина", 42)
    assert DISCLAIMER in reply and "нужна проверка человеком" in reply
    assert st.list(status="new")["items"][0]["source"] == "telegram"


def test_bot_is_off_by_default(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ENABLED", "0")
    assert run() == 2
