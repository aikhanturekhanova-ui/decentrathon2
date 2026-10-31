"""Settings read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader: KEY=VALUE lines, existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    """Paths from the environment/.env are resolved against the PROJECT root, never against the current directory
    (running a script from scripts/ or another folder used to break relative values such as DATA_DIR=data/handout)."""
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else (ROOT / p)


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    rules_path: Path
    taxonomy_path: Path
    db_path: Path
    faq_path: Path
    # learned complaint model built by `make train` (optional; absent = lexicon only)
    model_path: Path
    use_model: bool
    # access control and limits of the HTTP service (app/security.py); empty API_TOKENS = demo mode, authentication off
    api_tokens_raw: str
    rate_limit_per_min: int
    # needs_human thresholds on the classification confidence (below the first: always a human; below the second: a human unless off-topic)
    human_conf_min: float
    human_ambiguous_min: float
    # LLM (optional). Without a key the service runs in explicit "template" mode.
    llm_provider: str  # anthropic | gemini (chosen by LLM_PROVIDER, otherwise by which API key is set)
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    llm_timeout_s: float
    llm_max_output_tokens: int
    llm_effort: str
    llm_disable_thinking: bool
    llm_price_in_per_mtok: float
    llm_price_out_per_mtok: float
    llm_budget_per_ticket_usd: float
    persist_triage: bool
    # the HTTP service keeps its duplicate-detection history in SQLite so that a restart does not forget it (batch mode never does)
    dedup_persist: bool
    # total wall-clock budget for the optional LLM call of ONE ticket (the p95 target for the whole ticket is 5 s)
    llm_total_budget_s: float
    config_warnings: tuple
    # Kazakh drafts: normal Kazakh letters (default, readable for reviewers) or the game's cp1251 look-alikes (ќ ґ ѕ є ј ў ї)
    kk_output_game_symbols: bool
    # restrict the top-level `lang` to the 5 values of the PDF contract (ru kk uz en und): lat -> ru, es/pt -> und
    lang_strict_contract: bool
    # Telegram (optional, off unless explicitly enabled)
    telegram_enabled: bool
    telegram_token: str

    # the key may be kept while the model stays off (LLM_ENABLED=0): evaluation scripts on real tickets force it off
    llm_switch: bool = True

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key) and self.llm_switch


def _clamped(name: str, value: float, lo: float, hi: float, warnings: list[str]) -> float:
    """A nonsensical setting (negative price, 10-minute timeout, 100k output tokens) is replaced by the nearest sane value and reported."""
    if value < lo or value > hi:
        fixed = min(max(value, lo), hi)
        warnings.append(f"{name}={value} is outside [{lo}, {hi}]; using {fixed}")
        return fixed
    return value


DEFAULT_MODEL = {"anthropic": "claude-haiku-5-5", "gemini": "gemini-3.5-flash-lite"}
DEFAULT_BASE_URL = {"anthropic": "https://api.anthropic.com", "gemini": "https://generativelanguage.googleapis.com"}


def load_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    data_dir = _path("DATA_DIR", ROOT / "data" / "handout")
    warnings: list[str] = []
    timeout = _clamped("LLM_TIMEOUT_S", _float("LLM_TIMEOUT_S", 3.5), 0.5, 20.0, warnings)
    total = _clamped("LLM_TOTAL_BUDGET_S", _float("LLM_TOTAL_BUDGET_S", 4.0), 0.5, 30.0, warnings)
    max_out = int(_clamped("LLM_MAX_OUTPUT_TOKENS", _float("LLM_MAX_OUTPUT_TOKENS", 450), 64, 2000, warnings))
    price_in = _clamped("LLM_PRICE_IN_PER_MTOK", _float("LLM_PRICE_IN_PER_MTOK", 0.0), 0.0, 1000.0, warnings)
    price_out = _clamped("LLM_PRICE_OUT_PER_MTOK", _float("LLM_PRICE_OUT_PER_MTOK", 0.0), 0.0, 1000.0, warnings)
    budget = _clamped("LLM_BUDGET_PER_TICKET_USD", _float("LLM_BUDGET_PER_TICKET_USD", 0.002), 0.0, 1.0, warnings)
    effort = os.environ.get("LLM_EFFORT", "low").strip() or "low"
    if effort not in {"low", "medium", "high"}:
        warnings.append(f"LLM_EFFORT={effort!r} is not low|medium|high; using 'low'")
        effort = "low"
    keys = {"anthropic": os.environ.get("ANTHROPIC_API_KEY", "").strip(), "gemini": os.environ.get("GEMINI_API_KEY", "").strip()}
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider not in keys:
        if provider:
            warnings.append(f"LLM_PROVIDER={provider!r} is not anthropic|gemini; chosen by the API key instead")
        provider = "anthropic" if keys["anthropic"] or not keys["gemini"] else "gemini"
    api_key = keys[provider]
    if api_key and not (price_in and price_out):
        warnings.append("LLM is enabled but LLM_PRICE_IN_PER_MTOK / LLM_PRICE_OUT_PER_MTOK are not set: the per-ticket budget cannot be "
                        "enforced and the cost is reported as UNKNOWN (never as $0)")
    return Settings(
        data_dir=data_dir,
        rules_path=_path("RULES_PATH", data_dir / "rules.json"),
        taxonomy_path=_path("TAXONOMY_PATH", data_dir / "taxonomy.json"),
        db_path=_path("DB_PATH", ROOT / "data" / "local" / "copilot.db"),
        faq_path=_path("FAQ_PATH", ROOT / "data" / "kb" / "faq.json"),
        model_path=_path("MODEL_PATH", ROOT / "data" / "models" / "complaint_clf.joblib"),
        use_model=_bool("USE_MODEL", True),
        api_tokens_raw=os.environ.get("API_TOKENS", "").strip(),
        # unset: OFF in the demo mode (a local run or a load test of /triage must not get 429), 1200/min when API_TOKENS is configured
        rate_limit_per_min=int(_clamped("RATE_LIMIT_PER_MIN", _float("RATE_LIMIT_PER_MIN", 1200 if os.environ.get("API_TOKENS", "").strip() else 0),
                                        0, 1_000_000, warnings)),
        human_conf_min=_clamped("HUMAN_CONF_MIN", _float("HUMAN_CONF_MIN", 0.4), 0.0, 1.0, warnings),
        human_ambiguous_min=_clamped("HUMAN_AMBIGUOUS_MIN", _float("HUMAN_AMBIGUOUS_MIN", 0.5), 0.0, 1.0, warnings),
        llm_switch=_bool("LLM_ENABLED", True),
        llm_provider=provider,
        llm_api_key=api_key,
        llm_model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL[provider],
        llm_base_url=os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL[provider],
        llm_timeout_s=timeout,
        llm_max_output_tokens=max_out,
        llm_effort=effort,
        llm_disable_thinking=_bool("LLM_DISABLE_THINKING", True),
        # Prices are NOT hard-coded facts: set them from the provider's price page.
        llm_price_in_per_mtok=price_in,
        llm_price_out_per_mtok=price_out,
        llm_budget_per_ticket_usd=budget,
        persist_triage=_bool("PERSIST_TRIAGE", False),
        dedup_persist=_bool("DEDUP_PERSIST", True),
        llm_total_budget_s=total,
        config_warnings=tuple(warnings),
        kk_output_game_symbols=_bool("KK_OUTPUT_GAME_SYMBOLS", False),
        lang_strict_contract=_bool("LANG_STRICT_CONTRACT", False),
        telegram_enabled=_bool("TELEGRAM_ENABLED", False),
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
    )
