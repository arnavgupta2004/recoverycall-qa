import json
from pathlib import Path

import numpy as np
import pytest

from src import audio
from src.metrics import edit_ops, error_rates, evaluate_spans, match_span, normalize, script_check

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- normalization

def test_normalize_nfc_composes():
    decomposed = "क़"  # क + nukta
    assert normalize(decomposed) == normalize("क़")  # precomposed क़ also NFC-normalizes to the same


def test_normalize_removes_zwj_zwnj():
    assert normalize("ಸನ್‌ರೈಸ್") == "ಸನ್ರೈಸ್"
    assert normalize("क्‍ष") == "क्ष"


def test_normalize_lowercase_and_punct():
    assert normalize("Haan ji, main Rohan bol raha hoon.") == "haan ji main rohan bol raha hoon"
    assert normalize("जी, बताइए।") == "जी बताइए"
    assert normalize("एक॥दो") == "एक दो"


def test_normalize_collapses_whitespace_and_keeps_numbers():
    assert normalize("  ₹500 \n\t paanch   sau ") == "₹500 paanch sau"


def test_normalize_does_not_strip_combining_marks():
    assert normalize("नहीं") == "नहीं"  # anusvara / vowel signs are Mn/Mc, not punctuation


# ---------------------------------------------------------------- WER / CER

def test_edit_ops_counts():
    ops = edit_ops("a b c d".split(), "a x c d e".split())
    assert (ops["sub"], ops["del"], ops["ins"], ops["distance"]) == (1, 0, 1, 2)
    ops = edit_ops("a b c".split(), "a c".split())
    assert (ops["sub"], ops["del"], ops["ins"]) == (0, 1, 0)


def test_error_rates_identity_after_normalization():
    r = error_rates("Haan ji, main Rohan.", "haan ji main rohan")
    assert r["wer"] == 0 and r["cer"] == 0


def test_error_rates_values():
    r = error_rates("one two three four", "one too three")
    assert r["wer"] == pytest.approx(2 / 4)
    assert (r["sub"], r["del"], r["ins"]) == (1, 1, 0)
    assert r["cer"] > 0


def test_error_rates_empty_hypothesis():
    r = error_rates("one two", "")
    assert r["wer"] == 1.0 and r["del"] == 2


# ---------------------------------------------------------------- script check

def test_script_mismatch_devanagari_for_roman():
    s = script_check("barah hazaar paanch sau rupaye", "बारह हज़ार पाँच सौ रुपये")
    assert s["script_mismatch"] is True
    assert s["reference_script"] == "LATIN" and s["hypothesis_script"] == "DEVANAGARI"


def test_script_ok_with_some_english_words():
    s = script_check("आपकी ईएमआई बकाया है", "आपकी EMI बकाया है")
    assert s["script_mismatch"] is False


def test_script_empty_hypothesis_not_mismatch():
    assert script_check("ನಮಸ್ಕಾರ", "")["script_mismatch"] is False


# ---------------------------------------------------------------- critical spans

def test_span_exact():
    m = match_span("पाँच सौ रुपये", "आपके पाँच सौ रुपये बाकी हैं")
    assert m["status"] == "exact"


def test_span_near_and_best_match():
    m = match_span("barah hazaar paanch sau rupaye", "ki barah hazar paanch sau rupaye due")
    assert m["status"] == "near" and "hazar" in m["best_match"]


def test_span_garbled():
    m = match_span("barah hazaar paanch sau rupaye", "kuch aur hi baat ho rahi thi")
    assert m["status"] == "garbled"


def test_negation_missing_flag():
    spans = [{"type": "negation", "text": "पूरी रकम नहीं दे पाऊँगी", "negation_markers": ["नहीं", "नही"]}]
    kept = evaluate_spans(spans, "मैं पूरी रकम नही दे पाऊँगी")[0]
    assert kept["negation_missing"] is False and kept["status"] == "near"
    lost = evaluate_spans(spans, "मैं पूरी रकम दे पाऊँगी")[0]
    assert lost["negation_missing"] is True


def test_kannada_negation_suffix_marker():
    spans = [{"type": "negation", "text": "ಪೂರ್ತಿ ಹಣ ಕಟ್ಟಲು ಸಾಧ್ಯವಿಲ್ಲ", "negation_markers": ["ಲ್ಲ"]}]
    assert evaluate_spans(spans, "ಪೂರ್ತಿ ಹಣ ಕಟ್ಟಲು ಸಾಧ್ಯ ಇಲ್ಲ")[0]["negation_missing"] is False
    assert evaluate_spans(spans, "ಪೂರ್ತಿ ಹಣ ಕಟ್ಟಲು ಸಾಧ್ಯ")[0]["negation_missing"] is True


# ---------------------------------------------------------------- data + audio helpers

def test_scripts_json_spans_occur_in_turn_text():
    data = json.loads((ROOT / "data" / "scripts.json").read_text(encoding="utf-8"))
    types = set()
    for sc in data["scenarios"]:
        for turn in sc["turns"]:
            for sp in turn.get("critical_spans", []):
                types.add(sp["type"])
                assert match_span(sp["text"], turn["text"])["status"] == "exact", sp["text"]
                if sp["type"] == "negation":
                    assert any(normalize(m) in normalize(sp["text"]) for m in sp["negation_markers"])
    assert types == {"amount", "negation", "threat", "family"}


def test_plan_chunks_respects_limit_and_turn_boundaries():
    offsets = [(i * 6.0, i * 6.0 + 5.5) for i in range(8)]  # 8 turns x 5.5 s, 0.5 s gaps
    chunks = audio.plan_chunks(offsets, total_s=47.5, max_s=28.0)
    assert [t for c in chunks for t in c["turns"]] == list(range(8))
    assert all(c["end_s"] - c["start_s"] <= 28.0 for c in chunks)
    assert len(chunks) == 2


def test_plan_chunks_rejects_overlong_turn():
    with pytest.raises(ValueError):
        audio.plan_chunks([(0.0, 40.0)], total_s=40.0)


def test_mix_at_snr_hits_target():
    sr = 8000
    t = np.arange(sr * 4) / sr
    sig = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    sig[sr: sr * 2] = 0  # a silent gap that must not count as signal
    offsets = [(0.0, 1.0), (2.0, 4.0)]
    _, info = audio.mix_at_snr(sig, sr, offsets, 10.0, seed=3)
    assert info["measured_snr_db"] == pytest.approx(10.0, abs=0.3)


# ---------------------------------------------------------------- negation_token (Fix 3)

def test_negation_token_whole_word_match():
    """negation_token present: whole-word hit prevents negation_missing."""
    spans = [{"type": "negation", "text": "पूरी रकम नहीं दे पाऊँगी",
              "negation_token": "नहीं", "negation_markers": ["नहीं"]}]
    kept = evaluate_spans(spans, "पूरी रकम नहीं दे पाऊंगी")[0]
    assert kept["negation_missing"] is False
    assert kept["negation_detection"] == "token"


def test_negation_token_not_present_sets_missing():
    """negation_token present but absent from best_match -> negation_missing=True."""
    spans = [{"type": "negation", "text": "पूरी रकम नहीं दे पाऊँगी",
              "negation_token": "नहीं", "negation_markers": ["नहीं"]}]
    lost = evaluate_spans(spans, "पूरी रकम दे पाऊंगी")[0]
    assert lost["negation_missing"] is True
    assert lost["negation_detection"] == "token"


def test_negation_token_no_partial_word_match():
    """negation_token whole-word check must NOT fire on a substring inside a different word.

    'नहीं' must not match inside 'अनहीं' (contrived word). Token check pads with spaces,
    so ' नहीं ' does not appear in ' अनहीं '.
    """
    spans = [{"type": "negation", "text": "पूरी रकम नहीं दे पाऊँगी",
              "negation_token": "नहीं", "negation_markers": ["नहीं"]}]
    result = evaluate_spans(spans, "पूरी रकम अनहीं दे पाऊंगी")[0]
    # 'नहीं' is embedded in 'अनहीं' but not as a standalone token.
    assert result["negation_missing"] is True


def test_negation_fallback_emits_warning():
    """No negation_token -> substring fallback used and a UserWarning is emitted."""
    spans = [{"type": "negation", "text": "पूरी रकम नहीं दे पाऊँगी",
              "negation_markers": ["नहीं"]}]
    with pytest.warns(UserWarning, match="substring fallback"):
        result = evaluate_spans(spans, "पूरी रकम नहीं दे पाऊंगी")[0]
    assert result["negation_detection"] == "substring_fallback"
    assert result["negation_missing"] is False


def test_negation_token_kannada_whole_word():
    """Kannada negation_token 'ಸಾಧ್ಯವಿಲ್ಲ' (a whole word) matched by token check."""
    spans = [{"type": "negation", "text": "ಪೂರ್ತಿ ಹಣ ಕಟ್ಟಲು ಸಾಧ್ಯವಿಲ್ಲ",
              "negation_token": "ಸಾಧ್ಯವಿಲ್ಲ", "negation_markers": ["ಲ್ಲ"]}]
    kept = evaluate_spans(spans, "ಪೂರ್ತಿ ಹಣ ಕಟ್ಟಲು ಸಾಧ್ಯವಿಲ್ಲ")[0]
    assert kept["negation_missing"] is False and kept["negation_detection"] == "token"
    lost = evaluate_spans(spans, "ಪೂರ್ತಿ ಹಣ ಕಟ್ಟಲು ಸಾಧ್ಯ")[0]
    assert lost["negation_missing"] is True


# ---------------------------------------------------------------- script_check empty ref (Fix 6b)

def test_script_check_empty_reference():
    """Pure-numeric reference has no letters: script_mismatch must be False, not crash."""
    result = script_check("500 1000", "five hundred one thousand")
    assert result["script_mismatch"] is False
    assert result["reference_script"] is None


def test_script_check_empty_both():
    result = script_check("", "")
    assert result["script_mismatch"] is False


# ---------------------------------------------------------------- char edit counts in records (Fix 6d)

def test_error_rates_exposes_char_edit_counts():
    """error_rates returns char_sub, char_del, char_ins."""
    r = error_rates("one two three", "one too")
    assert "char_sub" in r and "char_del" in r and "char_ins" in r
    assert r["char_sub"] + r["char_del"] + r["char_ins"] > 0


def test_scripts_json_negation_tokens_present_and_in_span():
    """Every negation span in scripts.json must have negation_token and it must occur in the span text."""
    data = json.loads((ROOT / "data" / "scripts.json").read_text(encoding="utf-8"))
    for sc in data["scenarios"]:
        for turn in sc["turns"]:
            for sp in turn.get("critical_spans", []):
                if sp["type"] == "negation":
                    assert "negation_token" in sp, f"missing negation_token on span: {sp['text']}"
                    tok_n = normalize(sp["negation_token"])
                    span_n = normalize(sp["text"])
                    assert f" {tok_n} " in f" {span_n} " or span_n == tok_n, (
                        f"negation_token {sp['negation_token']!r} not a whole token in span {sp['text']!r}"
                    )
