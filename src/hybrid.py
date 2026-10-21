"""Hybrid flags = rules flags ∪ verified LLM flags, deduplicated by overlapping quoted span.

Two flags are duplicates when they have the same concern_type, the same segment, and their
quotes overlap in that segment's text. The rules flag is kept (its quote is the narrow matched
phrase) and records that the LLM agreed. Flags of different types on the same text are both
kept (multi-label). Labels and wording are unchanged: every flag keeps the classifier's label
and regulatory note.
"""

from __future__ import annotations

from typing import Any


def _span(flag: dict, text_by_segment: dict[str, str]) -> tuple[int, int] | None:
    text = text_by_segment.get(flag["segment_id"], "")
    i = text.find(flag["quote"]) if flag.get("quote") else -1
    return (i, i + len(flag["quote"])) if i >= 0 else None


def merge_flags(rule_flags: list[dict[str, Any]], llm_flags: list[dict[str, Any]],
                segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_by_segment = {s["segment_id"]: s.get("text", "") for s in segments}
    merged: list[dict[str, Any]] = []
    for f in rule_flags:
        merged.append({**f, "source": "rules"})
    for f in llm_flags:
        span = _span(f, text_by_segment)
        dup = None
        for m in merged:
            if m["concern_type"] != f["concern_type"] or m["segment_id"] != f["segment_id"]:
                continue
            ms = _span(m, text_by_segment)
            if span and ms and span[0] < ms[1] and ms[0] < span[1]:
                dup = m
                break
        if dup is not None:
            if dup["source"] == "rules":
                dup["source"] = "rules+llm"
            continue
        merged.append({**f, "source": "llm"})
    return merged
