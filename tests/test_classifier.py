"""Tests for src/classifier.py and src/concern_rules.py."""

from __future__ import annotations

import pytest

from src.classifier import classify
from src.concern_rules import ABUSIVE, AUTHORITY, INTIMIDATION, THIRD_PARTY


def _seg(text: str, sid: str = "seg00", lang: str = "hi-IN") -> tuple[list, str]:
    return [{"segment_id": sid, "text": text, "start_s": 0.0, "end_s": 10.0}], lang


# ── T2-a: each concern type fires on a synthetic one-turn transcript ──────────

class TestEachCategoryFires:
    def test_intimidation_hindi(self):
        segs, lang = _seg("आज शाम तक पैसे नहीं आए तो हमारे लोग आएँगे और घर पर आएँगे।")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert INTIMIDATION in types

    def test_intimidation_hinglish(self):
        segs, lang = _seg("Aaj shaam tak hamare log aapke ghar aayenge.", lang="en-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert INTIMIDATION in types

    def test_intimidation_kannada(self):
        segs, lang = _seg("ಇಂದು ಸಂಜೆಯೊಳಗೆ ಹಣ ಬರದಿದ್ದರೆ ನಮ್ಮ ಹುಡುಗರು ನಿಮ್ಮ ಮನೆಗೆ ಬರುತ್ತಾರೆ.", lang="kn-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert INTIMIDATION in types

    def test_abusive_hindi(self):
        segs, lang = _seg("आप बिल्कुल बेशर्म हैं।")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert ABUSIVE in types

    def test_abusive_hinglish(self):
        segs, lang = _seg("Aap bilkul bewakoof hain.", lang="en-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert ABUSIVE in types

    def test_abusive_kannada(self):
        segs, lang = _seg("ನೀವು ನಾಚಿಕೆಗೇಡು ಮಾಡಿಕೊಳ್ಳುತ್ತಿದ್ದೀರಿ.", lang="kn-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert ABUSIVE in types

    def test_third_party_hindi(self):
        segs, lang = _seg("हम आपके दफ़्तर फ़ोन करेंगे और आपके मैनेजर को बताएँगे।")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert THIRD_PARTY in types

    def test_third_party_hinglish(self):
        segs, lang = _seg("Hum aapki company ko call karenge aur boss ko batayenge.", lang="en-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert THIRD_PARTY in types

    def test_third_party_kannada(self):
        segs, lang = _seg("ನಾವು ನಿಮ್ಮ ಹೆಂಡತಿ ಮತ್ತು ನಿಮ್ಮ ತಂದೆಗೆ ಫೋನ್ ಮಾಡುತ್ತೇವೆ.", lang="kn-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert THIRD_PARTY in types

    def test_authority_hindi(self):
        segs, lang = _seg("हम पुलिस को सूचना देंगे और एफ़आईआर दर्ज होगी।")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert AUTHORITY in types

    def test_authority_hinglish(self):
        segs, lang = _seg("Agar aaj nahi diya toh police complaint karenge aur arrest karwayenge.", lang="en-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert AUTHORITY in types

    def test_authority_kannada(self):
        segs, lang = _seg("ನಾವು ಪೊಲೀಸ್ ಕಂಪ್ಲೇಂಟ್ ಮಾಡುತ್ತೇವೆ.", lang="kn-IN")
        flags = classify(segs, lang)
        types = {f["concern_type"] for f in flags}
        assert AUTHORITY in types


# ── T2-b: hard-negative phrases must not fire ─────────────────────────────────

class TestHardNegativesDoNotFire:
    """At least three hard-negative phrases must produce no flags."""

    def test_cibil_mention_hindi(self):
        segs, lang = _seg(
            "Aapki CIBIL rating prabhavit ho sakti hai agar payment nahi hui.",
            lang="en-IN",
        )
        # 'nahi' appears but there should be no concern flag (CIBIL mention is lawful)
        flags = classify(segs, lang)
        assert flags == [], f"Unexpected flags: {flags}"

    def test_escalation_mention_hindi(self):
        segs, lang = _seg(
            "अगर अगले हफ़्ते तक भुगतान नहीं हुआ तो हमें मामला आगे बढ़ाना होगा।"
        )
        flags = classify(segs, lang)
        assert flags == [], f"Unexpected flags: {flags}"

    def test_escalation_mention_hinglish(self):
        segs, lang = _seg(
            "Agar agle hafte tak payment nahi aaya toh humein aapka case escalate karna hoga.",
            lang="en-IN",
        )
        flags = classify(segs, lang)
        assert flags == [], f"Unexpected flags: {flags}"

    def test_review_queue_mention(self):
        segs, lang = _seg(
            "Agar payment nahi hui toh aapka account review mein bhejana hoga.",
            lang="en-IN",
        )
        flags = classify(segs, lang)
        assert flags == [], f"Unexpected flags: {flags}"

    def test_kannada_credit_score_mention(self):
        segs, lang = _seg(
            "ತಡವಾದರೆ ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್ ಪ್ರಭಾವಿತವಾಗಬಹುದು.",
            lang="kn-IN",
        )
        flags = classify(segs, lang)
        assert flags == [], f"Unexpected flags: {flags}"

    def test_kannada_escalation_mention(self):
        segs, lang = _seg(
            "ಮುಂದಿನ ವಾರದೊಳಗೆ ಕಟ್ಟದಿದ್ದರೆ ನಾವು ನಿಮ್ಮ ಕೇಸನ್ನು ಮೇಲ್ ಹಂತಕ್ಕೆ ಕಳುಹಿಸಬೇಕಾಗುತ್ತದೆ.",
            lang="kn-IN",
        )
        flags = classify(segs, lang)
        assert flags == [], f"Unexpected flags: {flags}"


# ── T2-c: every flag's quote is a substring of the input text ─────────────────

class TestQuoteIsSubstring:
    def test_hindi_flag_quote_in_text(self):
        text = "अगर आज शाम तक पैसे नहीं आए, तो हमारे लोग आपके घर आएँगे।"
        segs = [{"segment_id": "seg00", "text": text, "start_s": 0.0, "end_s": 10.0}]
        flags = classify(segs, "hi-IN")
        for flag in flags:
            assert flag["quote"] in text, (
                f"Quote {flag['quote']!r} not a substring of:\n{text!r}"
            )

    def test_hinglish_flag_quote_in_text(self):
        text = "Humne court mein case file kar diya hai, warrant aayega."
        segs = [{"segment_id": "seg00", "text": text, "start_s": 0.0, "end_s": 8.0}]
        flags = classify(segs, "en-IN")
        for flag in flags:
            assert flag["quote"] in text, (
                f"Quote {flag['quote']!r} not a substring of:\n{text!r}"
            )

    def test_kannada_flag_quote_in_text(self):
        text = "ನಾವು ಪೊಲೀಸ್ ಕಂಪ್ಲೇಂಟ್ ಮಾಡುತ್ತೇವೆ ಮತ್ತು ನಿಮ್ಮನ್ನು ಬಂಧಿಸಲಾಗುತ್ತದೆ."
        segs = [{"segment_id": "seg00", "text": text, "start_s": 0.0, "end_s": 10.0}]
        flags = classify(segs, "kn-IN")
        for flag in flags:
            assert flag["quote"] in text, (
                f"Quote {flag['quote']!r} not a substring of:\n{text!r}"
            )

    def test_multiple_segments_all_quotes_valid(self):
        segments = [
            {"segment_id": "seg00",
             "text": "Main aapki company ko call karenge aur boss ko batayenge.",
             "start_s": 0.0, "end_s": 5.0},
            {"segment_id": "seg01",
             "text": "Police complaint karenge aur arrest karwayenge.",
             "start_s": 5.0, "end_s": 10.0},
        ]
        flags = classify(segments, "en-IN")
        for flag in flags:
            seg_text = next(s["text"] for s in segments if s["segment_id"] == flag["segment_id"])
            assert flag["quote"] in seg_text, (
                f"Quote {flag['quote']!r} not in segment text:\n{seg_text!r}"
            )


# ── T2-d: empty transcript returns empty list ─────────────────────────────────

class TestEmptyInput:
    def test_empty_list(self):
        assert classify([], "hi-IN") == []

    def test_single_empty_text(self):
        segs = [{"segment_id": "seg00", "text": "", "start_s": 0.0, "end_s": 5.0}]
        assert classify(segs, "hi-IN") == []

    def test_whitespace_only_text(self):
        segs = [{"segment_id": "seg00", "text": "   ", "start_s": 0.0, "end_s": 5.0}]
        assert classify(segs, "hi-IN") == []


# ── Label and regulatory_note correctness ─────────────────────────────────────

class TestOutputShape:
    def test_authority_flag_has_distinct_label(self):
        segs, lang = _seg("Agar aaj nahi diya toh police complaint karenge.", lang="en-IN")
        flags = classify(segs, lang)
        auth_flags = [f for f in flags if f["concern_type"] == AUTHORITY]
        assert auth_flags, "Expected authority flag"
        for f in auth_flags:
            assert f["label"] == "claim of police, arrest or court action — verify"

    def test_non_authority_flag_has_default_label(self):
        segs, lang = _seg("Hamare log aapke ghar aayenge.", lang="en-IN")
        flags = classify(segs, lang)
        intim_flags = [f for f in flags if f["concern_type"] == INTIMIDATION]
        assert intim_flags, "Expected intimidation flag"
        for f in intim_flags:
            assert f["label"] == "potential conduct concern — human review required"

    def test_regulatory_note_always_present(self):
        segs, lang = _seg("Hamare log aapke ghar aayenge.", lang="en-IN")
        flags = classify(segs, lang)
        for f in flags:
            assert f["regulatory_note"] == "to verify against the RBI circular"

    def test_required_keys_present(self):
        segs, lang = _seg("Police complaint karenge.", lang="en-IN")
        flags = classify(segs, lang)
        for f in flags:
            for key in ("concern_type", "quote", "segment_id", "start_s", "end_s",
                        "label", "regulatory_note"):
                assert key in f, f"Missing key {key!r} in flag: {f}"


# ── Quote extraction: a flag quotes the matched phrase, not the whole turn ──────

import json as _json
import unicodedata as _ud
from pathlib import Path as _Path

from src.classifier import _normalize_with_map, _recover_original_quote, verify_quote
from src.metrics import normalize as _normalize


class TestQuoteExtraction:
    def test_map_matches_normalize_on_every_dev_turn(self):
        dev = _json.loads((_Path(__file__).resolve().parents[1] / "data" / "scripts_dev.json").read_text(encoding="utf-8"))
        for sc in dev["scenarios"]:
            for t in sc["turns"]:
                text = _ud.normalize("NFC", t["text"])
                norm, src = _normalize_with_map(text)
                assert norm == _normalize(text)
                assert len(src) == len(norm)

    def test_multiword_phrase_across_punctuation_is_narrow(self):
        text = "नौकरी गई या क्या — यह हमारी समस्या नहीं है। अगर पैसा नहीं आया, तो हमारे लोग आपके घर आएँगे और सब देखेंगे।"
        assert _recover_original_quote(text, _normalize("हमारे लोग आपके घर")) == "हमारे लोग आपके घर"

    def test_case_and_inner_punctuation_preserved_from_original(self):
        text = "Uske saath, hum aapke EMPLOYER ko, formal letter bhejenge."
        assert _recover_original_quote(text, _normalize("employer ko formal letter")) == "EMPLOYER ko, formal letter"

    def test_zwnj_inside_phrase(self):
        text = "ನಾನು ಸನ್‌ರೈಸ್ ಫೈನಾನ್ಸ್‌ನಿಂದ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ"
        quote = _recover_original_quote(text, _normalize("ಸನ್ರೈಸ್"))
        assert quote == "ಸನ್‌ರೈಸ್" and quote in text

    def test_word_boundary_does_not_match_inside_longer_word(self):
        text = "लड़कियों को मत भेजो, हम लड़के भेजेंगे"
        assert _recover_original_quote(text, _normalize("लड़के"), whole_word=True) == "लड़के"

    def test_unmatched_phrase_falls_back_to_full_text(self):
        text = "कोई मेल नहीं"
        assert _recover_original_quote(text, _normalize("पुलिस")) == text

    def test_classifier_quotes_are_short_and_verbatim(self):
        text = ("Genuine problem hai toh nikalo paise. Nahi diya toh hamare log kal subah aapke ghar ke bahar "
                "khade honge — neighbours ko dikhayenge kin logo ne hamare saath fraud kiya.")
        flags = classify([{"segment_id": "s0", "text": text, "start_s": 0.0, "end_s": 6.0}], "en-IN")
        assert flags
        for f in flags:
            assert verify_quote(f["quote"], text)
            assert len(f["quote"]) < len(text)

    def test_verify_quote(self):
        assert verify_quote("घर आएँगे", "हमारे लोग घर आएँगे")
        assert not verify_quote("घर आयेंगे", "हमारे लोग घर आएँगे")   # paraphrase / respelling rejected
        assert not verify_quote("", "anything")
        assert not verify_quote("   ", "a   b")
