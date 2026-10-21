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
The returned "quote" is the shortest matching run in the *original* (un-normalized) text.
For "substring" rules this is the original text slice aligned to the normalized match.
For "word_boundary" rules (single tokens) the original word is recovered the same way.
If alignment fails (shouldn't happen given correct rules), the full segment text is used.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from src.metrics import normalize
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


def _recover_original_quote(original: str, norm_original: str, norm_phrase: str) -> str:
    """Find the original-text span corresponding to a normalized phrase match.

    Strategy: walk character-by-character through the original text, maintaining
    a parallel position in the normalized string. When the normalized position
    reaches the start of norm_phrase, record the original start. When the
    normalized position passes the end, record the original end.

    Falls back to the full segment text if alignment fails.
    """
    # Build a mapping: norm_pos -> list of (orig_start, orig_end) char ranges.
    # We map each normalized character back to its originating original position.
    norm_chars: list[tuple[int, int]] = []  # list of (orig_start, orig_end) per norm char
    orig_i = 0
    orig_buf = ""  # accumulates normalized output for current orig character window

    # We re-normalize character-by-character to build the alignment.
    # This is approximate but sufficient for multi-word phrase alignment.
    norm_text = ""
    char_map: list[int] = []  # norm_text[i] came from original[char_map[i]]
    for oi, och in enumerate(original):
        n = normalize(och)
        for nc in n:
            norm_text += nc
            char_map.append(oi)

    # Find norm_phrase in norm_text
    try:
        start_norm = norm_text.index(norm_phrase)
    except ValueError:
        return original  # fallback

    end_norm = start_norm + len(norm_phrase) - 1
    orig_start = char_map[start_norm]
    orig_end = char_map[end_norm] + 1  # exclusive

    return original[orig_start:orig_end].strip()


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
            quote = _recover_original_quote(text, norm_text, norm_phrase)

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
