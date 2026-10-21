"""Rules-based conduct-concern classifier for recovery call transcripts.

Usage
-----
    from src.classifier import classify

    segments = [
        {"segment_id": "seg00", "text": "...", "start_s": 0.0, "end_s": 12.0},
        ...
    ]
    flags = classify(segments, language="hi-IN")

Each flag in the returned list:
    {
        "concern_type": str,       # one of the four category constants
        "quote":        str,       # narrowest substring of the *original* text that matched
        "segment_id":   str,
        "start_s":      float,
        "end_s":        float,
        "label":        str,       # per-category label string
        "regulatory_note": str,    # always "to verify against the RBI circular"
    }

Language codes
--------------
    "hi-IN"  → Devanagari rules
    "kn-IN"  → Kannada rules
    "en-IN"  → Latin rules (code-mixed Hinglish)

The classifier uses the same Unicode normalization as src/metrics.normalize so that nukta
variants, ZWNJ, punctuation differences and case do not break rule matching.

Quote extraction
----------------
The returned "quote" is the matched phrase as it appears in the *original* segment text: the
normalized match is mapped back character by character (_normalize_with_map), so the quote is
always a verbatim substring of the segment. The full segment text is used only if alignment
fails (a non-NFC segment whose composed and decomposed spellings differ).
"""

from __future__ import annotations

import unicodedata
from typing import Any

from src.metrics import EXTRA_PUNCT, normalize
from src.concern_rules import (
    ALL_CATEGORIES,
    AUTHORITY,
    CATEGORY_LABELS,
    REGULATORY_NOTE,
    RULES_DEVANAGARI,
    RULES_KANNADA,
    RULES_LATIN,
)


def _dominant_script(text: str) -> str:
    """Return 'DEVANAGARI', 'KANNADA', or 'LATIN' based on letter majority."""
    counts: dict[str, int] = {}
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        try:
            script = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "LATIN"
    return max(counts, key=lambda s: counts[s])


def _rules_for_language(language: str) -> list[dict]:
    """Select rule list by language code."""
    if language == "kn-IN":
        return RULES_KANNADA
    if language == "hi-IN":
        return RULES_DEVANAGARI
    # en-IN (Hinglish) and any other code → Latin rules
    return RULES_LATIN


def _rules_for_text(text: str, language: str) -> list[dict]:
    """Select rule list by text script when language is ambiguous, otherwise by language."""
    if language in ("hi-IN", "kn-IN", "en-IN"):
        return _rules_for_language(language)
    script = _dominant_script(text)
    if script == "KANNADA":
        return RULES_KANNADA
    if script == "DEVANAGARI":
        return RULES_DEVANAGARI
    return RULES_LATIN


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Apply src.metrics.normalize to text in one pass, recording each output char's source index.

    Mirrors normalize(): drop format chars (Cf), lowercase, punctuation (P*, । ॥) and whitespace
    to single spaces, strip. NFC is applied by the caller; positions refer to the string passed in.
    """
    out: list[str] = []
    src: list[int] = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch) == "Cf":
            continue
        for c in ch.lower():
            if c.isspace() or unicodedata.category(c).startswith("P") or c in EXTRA_PUNCT:
                if out and out[-1] != " ":
                    out.append(" ")
                    src.append(i)
            else:
                out.append(c)
                src.append(i)
    while out and out[-1] == " ":
        out.pop()
        src.pop()
    return "".join(out), src


def _recover_original_quote(original: str, norm_phrase: str, whole_word: bool = False) -> str:
    """Return the shortest slice of `original` whose normalization is `norm_phrase`.

    The slice runs from the first to the last character that produced the match, so it is
    always a verbatim substring of the original segment text. Falls back to the full segment
    text only if the phrase cannot be located (e.g. the original is not in NFC form and the
    composed and decomposed spellings differ).
    """
    base = original if unicodedata.is_normalized("NFC", original) else None
    if base is None:
        return original
    norm, src = _normalize_with_map(base)
    target = f" {norm_phrase} " if whole_word else norm_phrase
    hay = f" {norm} " if whole_word else norm
    pos = hay.find(target)
    if pos < 0:
        return original
    start = pos  # in `norm` coordinates (padding offsets cancel for whole_word)
    end = start + len(norm_phrase) - 1
    return base[src[start]: src[end] + 1]


def verify_quote(quote: str, segment_text: str) -> bool:
    """A flag's quote is valid only if it occurs verbatim (non-empty) in its segment's text."""
    return bool(quote and quote.strip()) and quote in segment_text


def classify(
    transcript_segments: list[dict[str, Any]],
    language: str,
) -> list[dict[str, Any]]:
    """Classify transcript segments for conduct concerns.

    Parameters
    ----------
    transcript_segments:
        List of dicts with keys: segment_id (str), text (str), start_s (float),
        end_s (float). Additional keys are ignored.
    language:
        STT language code: 'hi-IN', 'kn-IN', or 'en-IN'.

    Returns
    -------
    List of flag dicts. Empty list if no concerns are found or input is empty.
    Each flag has: concern_type, quote, segment_id, start_s, end_s, label,
    regulatory_note.
    """
    if not transcript_segments:
        return []

    rules = _rules_for_text(
        " ".join(seg.get("text", "") for seg in transcript_segments),
        language,
    )

    flags: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()  # (segment_id, concern_type, phrase) dedupe key

    for seg in transcript_segments:
        text: str = seg.get("text", "")
        if not text.strip():
            continue

        norm_text = normalize(text)
        padded = f" {norm_text} "

        for rule in rules:
            phrase: str = rule["phrase"]  # already lowercase; normalize for safety
            norm_phrase = normalize(phrase)
            match_mode: str = rule["match"]
            concern: str = rule["concern_type"]

            if match_mode == "word_boundary":
                matched = f" {norm_phrase} " in padded
            else:  # substring
                matched = norm_phrase in norm_text

            if not matched:
                continue

            dedup_key = (seg["segment_id"], concern, norm_phrase)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Recover original-text quote
            quote = _recover_original_quote(text, norm_phrase, whole_word=match_mode == "word_boundary")

            flags.append({
                "concern_type": concern,
                "quote": quote,
                "segment_id": seg["segment_id"],
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "label": CATEGORY_LABELS[concern],
                "regulatory_note": REGULATORY_NOTE,
            })

    # Deduplicate across segments: if the same concern fires on the same segment
    # from multiple rules, keep only the first (shortest) quote per (segment, concern).
    deduped: list[dict] = []
    seen_seg_concern: set[tuple[str, str]] = set()
    for flag in flags:
        key = (flag["segment_id"], flag["concern_type"])
        if key not in seen_seg_concern:
            seen_seg_concern.add(key)
            deduped.append(flag)

    return deduped
