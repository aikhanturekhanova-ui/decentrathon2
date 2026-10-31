"""Hermetic test setup: artificial rulebook, temp database, no network, no API key."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"

# Must be set before app modules read settings.
os.environ["RULES_PATH"] = str(FIXTURES / "rules_synthetic.json")
os.environ["TAXONOMY_PATH"] = str(FIXTURES / "does_not_exist.json")  # -> built-in taxonomy
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""  # hermetic: a key in a local .env must never make tests call a real model
os.environ["LLM_PROVIDER"] = ""
os.environ["LLM_MODEL"] = ""
os.environ["PERSIST_TRIAGE"] = "0"
os.environ["FAQ_PATH"] = str(FIXTURES / "no_faq.json")  # hermetic: no knowledge-base file
os.environ["TELEGRAM_ENABLED"] = "0"
# hermetic: a locally trained model (data/models/, built from the confidential tickets) must not change test outcomes;
# the hybrid path is tested with a tiny model trained on artificial texts (tests/test_ml.py)
os.environ["USE_MODEL"] = os.environ.get("TEST_USE_MODEL", "0")

from app.config import load_settings  # noqa: E402
from app.knowledge import RuleBook  # noqa: E402
from app.pipeline import TriageService  # noqa: E402
from app.schemas import Message, TriageRequest  # noqa: E402


@pytest.fixture(scope="session")
def rulebook() -> RuleBook:
    return RuleBook.load(FIXTURES / "rules_synthetic.json")


@pytest.fixture()
def service(rulebook, tmp_path) -> TriageService:
    os.environ["DB_PATH"] = str(tmp_path / "t.db")
    return TriageService(load_settings(), rules=rulebook)


def make_req(text: str, kind: str = "complaint", att: int = 1, **extra) -> TriageRequest:
    return TriageRequest(kind=kind, messages=[Message(role="player", text=text, attachments=att)], **extra)


@pytest.fixture()
def mk():
    return make_req


REAL_DATA = ROOT / "data" / "handout"
requires_real_data = pytest.mark.skipif(
    not (REAL_DATA / "rules.json").is_file(), reason="organizers' files are not in data/handout (they are not published)"
)
