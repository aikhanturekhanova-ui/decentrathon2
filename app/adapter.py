"""Adapter: full dataset record  ->  /triage request.

The dataset record also contains the administrator's answer and fields extracted from it
(verdict, punishment, violation). Those reveal the outcome of the case, so they MUST NOT reach
the pipeline. This module is the only place that converts a record, and it drops them.
"""
from __future__ import annotations

from typing import Any

from .schemas import Message, TriageRequest

# Roles whose text is information about the new case. "admin" is the answer = leakage.
PLAYER_ROLES = {"player", "reported_player"}

LEAKY_FIELDS = ("verdict", "punishment", "violation", "category", "priority", "status")


def record_to_request(record: dict[str, Any]) -> TriageRequest:
    images = max(0, int(record.get("images_count") or 0))
    out: list[Message] = []
    first_player = True
    for m in record.get("messages", []):
        role = m.get("role", "player")
        if role not in PLAYER_ROLES:
            continue  # drop admin replies entirely
        att = 0
        if role == "player" and first_player:
            att = int(images)  # the attachment count belongs to the complainant's message
            first_player = False
        out.append(Message(role=role, text=m.get("text") or "", attachments=att, ts=m.get("ts")))
    if not out:  # a record without any player text still has to be answerable
        out.append(Message(role="player", text="", attachments=int(images)))
    return TriageRequest(
        kind=record["kind"],
        messages=out,
        ticket_id=record.get("ticket_id"),
        created_at=record.get("created_at"),
        reporter_id=record.get("reporter_hash"),
        reported_id=record.get("reported_hash"),
        reported_nick=record.get("reported_nick"),
        rule_refs_raw=record.get("rule_refs_raw"),
        server_lang=record.get("server_lang"),
        region=record.get("region"),
    )


def to_request(obj: dict[str, Any]) -> TriageRequest:
    """Accept both the compact /triage body and a full dataset record."""
    if "ticket_id" in obj and ("reporter_hash" in obj or "images_count" in obj or "verdict" in obj):
        return record_to_request(obj)
    cleaned = {k: v for k, v in obj.items() if k not in LEAKY_FIELDS}
    # a compact body may still carry admin messages; they are ignored downstream, strip anyway
    cleaned["messages"] = [m for m in (cleaned.get("messages") or []) if (m.get("role") or "player") in PLAYER_ROLES]
    if not cleaned["messages"]:
        cleaned["messages"] = [{"role": "player", "text": ""}]
    return TriageRequest.model_validate(cleaned)
