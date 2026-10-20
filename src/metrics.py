"""Text normalization, WER/CER with edit breakdown, script check, and critical-span matching."""

from __future__ import annotations

import unicodedata
import warnings
from collections import Counter
from difflib import SequenceMatcher

EXTRA_PUNCT = {"।", "॥"}  # । danda, ॥ double danda (already Po, listed for clarity)
NEAR_THRESHOLD = 0.8


def normalize(text: str) -> str:
    """NFC -> drop format chars (Cf, incl. ZWJ/ZWNJ) -> lowercase -> punctuation to space -> collapse spaces.

    Punctuation (Unicode category P*) is replaced by a space rather than deleted, so
    "ji,bataiye" splits into two words in both reference and hypothesis. Numbers and
    number words are left alone: "500" and "paanch sau" are different tokens on purpose.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.lower()
    text = "".join(" " if (unicodedata.category(ch).startswith("P") or ch in EXTRA_PUNCT) else ch for ch in text)
    return " ".join(text.split())


def edit_ops(ref: list, hyp: list) -> dict:
    """Levenshtein alignment between token lists. Returns counts of substitutions, deletions, insertions."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    sub = dele = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {"sub": sub, "del": dele, "ins": ins, "ref_len": n, "distance": dp[n][m]}


def error_rates(reference: str, hypothesis: str) -> dict:
    """WER on whitespace tokens and CER on code points (spaces included), both after normalize()."""
    ref_n, hyp_n = normalize(reference), normalize(hypothesis)
    w = edit_ops(ref_n.split(), hyp_n.split())
    c = edit_ops(list(ref_n), list(hyp_n))
    return {
        "wer": w["distance"] / w["ref_len"] if w["ref_len"] else None,
        "cer": c["distance"] / c["ref_len"] if c["ref_len"] else None,
        "sub": w["sub"], "del": w["del"], "ins": w["ins"], "ref_words": w["ref_len"],
        "char_sub": c["sub"], "char_del": c["del"], "char_ins": c["ins"], "ref_chars": c["ref_len"],
    }


def script_of(ch: str) -> str | None:
    if not unicodedata.category(ch).startswith("L"):
        return None
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return None


def script_profile(text: str) -> Counter:
    return Counter(s for s in map(script_of, text) if s)


def script_check(reference: str, hypothesis: str) -> dict:
    """script_mismatch is True when most hypothesis letters are not in the reference's dominant script."""
    ref_prof, hyp_prof = script_profile(reference), script_profile(hypothesis)
    # Guard: reference may contain no letters (e.g. purely numeric reference).
    ref_script = ref_prof.most_common(1)[0][0] if ref_prof else None
    hyp_script = hyp_prof.most_common(1)[0][0] if hyp_prof else None
    total = sum(hyp_prof.values())
    if ref_script is None:
        # No letters in reference: cannot determine script mismatch.
        share = None
    else:
        share = hyp_prof.get(ref_script, 0) / total if total else None
    return {
        "reference_script": ref_script,
        "hypothesis_script": hyp_script,
        "hypothesis_share_in_reference_script": round(share, 3) if share is not None else None,
        "script_mismatch": bool(ref_script is not None and total and share is not None and share < 0.5),
    }


def match_span(span: str, transcript: str) -> dict:
    """Fuzzy-locate span in transcript using word windows of len(span)-1..len(span)+1 and SequenceMatcher."""
    span_n, hyp_n = normalize(span), normalize(transcript)
    span_toks, hyp_toks = span_n.split(), hyp_n.split()
    if not span_toks:
        raise ValueError("empty span")
    if f" {span_n} " in f" {hyp_n} ":
        return {"status": "exact", "ratio": 1.0, "best_match": span_n}
    best_ratio, best_text = 0.0, ""
    for size in {max(1, len(span_toks) - 1), len(span_toks), len(span_toks) + 1}:
        for start in range(0, max(1, len(hyp_toks) - size + 1)):
            window = " ".join(hyp_toks[start:start + size])
            ratio = SequenceMatcher(None, span_n, window, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best_text = ratio, window
    status = "near" if best_ratio >= NEAR_THRESHOLD else "garbled"
    return {"status": status, "ratio": round(best_ratio, 3), "best_match": best_text}


def evaluate_spans(spans: list[dict], transcript: str) -> list[dict]:
    """Match each critical span; for negation spans, flag when no negation marker survives in the best match.

    Negation detection priority:
      1. If the span has a "negation_token" field, check for that token by whole-word
         match on the normalized best_match (most precise; use for single words / suffixes
         that are unambiguous when space-delimited).
      2. Otherwise fall back to substring matching on each "negation_markers" entry and
         emit a warning so the caller knows the less-precise path was taken.
    """
    out = []
    for sp in spans:
        res = {"type": sp["type"], "span": sp["text"], **match_span(sp["text"], transcript)}
        if sp["type"] == "negation":
            token = sp.get("negation_token")
            markers = sp.get("negation_markers", [])
            best_norm = res["best_match"]
            if token is not None:
                # Whole-token match: pad with spaces so partial-word hits are avoided.
                tok_n = normalize(token)
                present = bool(tok_n and f" {tok_n} " in f" {best_norm} ")
                res["negation_token"] = token
                res["negation_detection"] = "token"
            else:
                # Fallback: substring match on each marker.
                norm_markers = [normalize(m) for m in markers]
                present = any(m and m in best_norm for m in norm_markers)
                warnings.warn(
                    f"negation span {sp['text']!r} has no negation_token; "
                    "using substring fallback for negation detection",
                    stacklevel=2,
                )
                res["negation_detection"] = "substring_fallback"
            res["negation_markers"] = markers
            res["negation_missing"] = not present
        out.append(res)
    return out
