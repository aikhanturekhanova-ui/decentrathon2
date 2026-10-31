"""Pydantic models for the POST /triage contract.

The required request/response fields follow the case statement exactly. Everything we add
is OPTIONAL on input (ignored if absent) and never changes the shape of the required output.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Kind = Literal["complaint", "report"]
Lang = Literal["ru", "kk", "uz", "en", "es", "pt", "lat", "und"]
Priority = Literal["p0", "p1", "p2", "p3"]
Category = Literal[
    "complaint.insult",
    "complaint.cheat",
    "complaint.dm",
    "complaint.scam",
    "complaint.rmt_ads",
    "complaint.admin",
    "complaint.other",
    "complaint.invalid",
    "report.question",
    "report.help",
    "report.complaint",
    "report.account",
    "report.payment",
    "report.offtopic",
]

# Request limits (the longest player text in all 119,337 real tickets is 1,630 characters). A request above them is REJECTED with 422,
# never silently cut: the safety flags read the whole dialogue (app/classify.player_text_full), the classifier/LLM a bounded context.
MAX_MESSAGE_CHARS = 20_000
MAX_DIALOGUE_CHARS = 60_000
MAX_TEXT_CHARS = 2000  # classification context (head + tail), see app/classify.player_text


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "player"
    text: str = Field(default="", max_length=MAX_MESSAGE_CHARS)
    attachments: int = Field(default=0, ge=0)  # no upper bound: real data has tickets with 145 images
    ts: Optional[str] = None

    @field_validator("text", "role", mode="before")
    @classmethod
    def _none_to_default(cls, v, info):
        """null text/role in a payload must not crash the service (an empty text is a valid, answerable ticket)."""
        return v if v is not None else ("player" if info.field_name == "role" else "")

    @field_validator("attachments", mode="before")
    @classmethod
    def _none_attachments(cls, v):
        return 0 if v is None else v


class TriageRequest(BaseModel):
    """Required: kind + messages. The rest is optional context that improves dedup/extraction."""

    model_config = ConfigDict(extra="ignore")

    kind: Kind
    messages: list[Message] = Field(min_length=1, max_length=50)
    # optional context (present in the dataset, may be absent in production)
    ticket_id: Optional[str] = None
    created_at: Optional[str] = None
    reporter_id: Optional[str] = None
    reported_id: Optional[str] = None
    reported_nick: Optional[str] = None
    rule_refs_raw: Optional[str] = None
    server_lang: Optional[str] = None
    region: Optional[str] = None

    @model_validator(mode="after")
    def _dialogue_size(self):
        total = sum(len(m.text) for m in self.messages)
        if total > MAX_DIALOGUE_CHARS:
            raise ValueError(f"the dialogue has {total} characters, the limit is {MAX_DIALOGUE_CHARS}: split it or shorten it")
        return self


class Extracted(BaseModel):
    nickname: Optional[str] = None
    rule_refs: list[str] = Field(default_factory=list)
    has_evidence: bool = False
    server: Optional[str] = None
    # Extensions (additive): what we could NOT verify.
    attachments: int = 0
    evidence_verified: bool = False
    rule_refs_unverified: list[str] = Field(default_factory=list)
    # additive: attached | claimed_only | stated_absent | asked_about | none ("has_evidence" is true for attached and claimed_only)
    evidence_status: str = "none"


class DraftReply(BaseModel):
    lang: str
    text: str
    cited: list[str] = Field(default_factory=list)  # numbered rule points (e.g. "2.1", "GZ 1.1"); verified to exist
    terms: list[str] = Field(default_factory=list)  # additive: game terms defined in «RP термины» (DM, DB, SK, ...), which have no numbers


class TriageResponse(BaseModel):
    lang: Lang
    category: str  # one of taxonomy.json's codes (see Category for the draft list); enforced at runtime by Taxonomy.fit()
    priority: str  # p0..p3 per taxonomy.json
    is_spam_or_duplicate: bool
    extracted: Extracted
    draft_reply: DraftReply
    needs_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str]


class TriageDebug(BaseModel):
    """Internal diagnostics, returned only with ?debug=1 and written to the stats file."""

    mode: str = "template"
    latency_ms: float = 0.0
    cost_usd: Optional[float] = 0.0  # MEASURED; None = a request was sent but its cost is unknown (no prices or no usage record)
    cost_estimate_max_usd: Optional[float] = None  # worst-case ESTIMATE for the sent requests, kept apart from the measured cost
    llm_calls: int = 0
    llm_requests_sent: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_error: Optional[str] = None
    signals: dict = Field(default_factory=dict)
    retrieved: list[dict] = Field(default_factory=list)
    guard_notes: list[str] = Field(default_factory=list)
