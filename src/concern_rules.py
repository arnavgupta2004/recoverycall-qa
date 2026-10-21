"""Keyword/phrase rule lists for the four conduct-concern categories.

Rules are keyed by script family: 'devanagari' (hi-IN), 'kannada' (kn-IN), 'latin'
(en-IN / Hinglish).  Each rule entry is a dict:
  {
    "phrase": str,           # normalized lowercase phrase to search for
    "concern_type": str,     # one of the four category constants below
    "match": "substring"|"word_boundary"
  }

"word_boundary" wraps the phrase with spaces for matching (" phrase ") in the padded,
normalized text.  Use it for short words that could appear as a substring of longer words
(e.g. "लड़के" inside "लड़कियों").  "substring" is fine for multi-word phrases.

Normalization applied before matching: NFC, drop Cf chars, lowercase, punctuation→space,
collapse whitespace (same as src/metrics.normalize).

RULES NOT YET FROZEN — pending review of dev-set results.
"""

from __future__ import annotations

# ── Category constants ────────────────────────────────────────────────────────
INTIMIDATION = "intimidation_or_threats"
ABUSIVE = "abusive_language"
THIRD_PARTY = "third_party_contact_threat"
AUTHORITY = "authority_or_legal_action_claim"

ALL_CATEGORIES = {INTIMIDATION, ABUSIVE, THIRD_PARTY, AUTHORITY}

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_DEFAULT = "potential conduct concern — human review required"
LABEL_AUTHORITY = "claim of police, arrest or court action — verify"
REGULATORY_NOTE = "to verify against the RBI circular"

CATEGORY_LABELS: dict[str, str] = {
    INTIMIDATION: LABEL_DEFAULT,
    ABUSIVE: LABEL_DEFAULT,
    THIRD_PARTY: LABEL_DEFAULT,
    AUTHORITY: LABEL_AUTHORITY,
}

# ── Rules ─────────────────────────────────────────────────────────────────────
# Each phrase is already lower-cased (normalization lowercases before matching).
# Punctuation will have been converted to spaces, so no commas or full-stops here.

RULES_DEVANAGARI: list[dict] = [
    # intimidation_or_threats — physical visit / coercion
    {"phrase": "घर आएँगे",      "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "घर आएंगे",      "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "घर आएगा",       "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "घर भेजेंगे",    "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "आदमी आएँगे",    "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "आदमी आएंगे",    "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "लड़के भेजेंगे", "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "लड़के आएँगे",   "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "लड़के आएंगे",   "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "लोग आएँगे",     "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "लोग आएंगे",     "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "दरवाज़े पर",    "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "दरवाजे पर",     "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "घर के बाहर",    "concern_type": INTIMIDATION, "match": "substring"},

    # abusive_language — clear-cut slurs and derogatory terms only
    {"phrase": "बेशर्म",        "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "बेवकूफ",        "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "बेईमान",        "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "कमीना",         "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "धोखेबाज़",      "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "धोखेबाज",       "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "चोर",           "concern_type": ABUSIVE, "match": "word_boundary"},

    # third_party_contact_threat — family, employer, neighbours
    {"phrase": "पत्नी को फ़ोन",      "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "पत्नी को फोन",       "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "पति को फ़ोन",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "पति को फोन",         "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "माँ को फ़ोन",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "माँ को फोन",         "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "भाई को फ़ोन",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "भाई को फोन",         "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "परिवार को बताएँगे",  "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "परिवार को बताएंगे",  "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "दफ़्तर फ़ोन करेंगे", "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "दफ्तर फोन करेंगे",   "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "मैनेजर को बताएँगे",  "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "मैनेजर को बताएंगे",  "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "पड़ोसियों को",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "पड़ोसियों को बताएँगे","concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ऑफ़िस को",           "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ऑफिस को",            "concern_type": THIRD_PARTY, "match": "substring"},
    # catch broad family-contact patterns
    {"phrase": "को फ़ोन करेंगे",     "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "को फोन करेंगे",      "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "को बताएँगे",         "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "को बताएंगे",         "concern_type": THIRD_PARTY, "match": "substring"},

    # authority_or_legal_action_claim — police, FIR, court, arrest
    {"phrase": "पुलिस",          "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "एफ़आईआर",       "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "एफआईआर",        "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "fir",            "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "कोर्ट में केस",  "concern_type": AUTHORITY, "match": "substring"},
    {"phrase": "कोर्ट में",      "concern_type": AUTHORITY, "match": "substring"},
    {"phrase": "समन",            "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "गिरफ़्तार",      "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "गिरफ्तार",       "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "अरेस्ट",         "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "वारंट",          "concern_type": AUTHORITY, "match": "word_boundary"},
]

RULES_KANNADA: list[dict] = [
    # intimidation_or_threats — physical visit / coercion
    {"phrase": "ಮನೆಗೆ ಬರುತ್ತಾರೆ",    "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ಮನೆಗೆ ಬರ್ತಾರೆ",      "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ಮನೆಗೆ ಕಳುಹಿಸುತ್ತೇವೆ", "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ಜನ ಬರುತ್ತಾರೆ",       "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ಹುಡುಗರು ಬರುತ್ತಾರೆ",  "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ಹುಡುಗರು ಬರ್ತಾರೆ",    "concern_type": INTIMIDATION, "match": "substring"},

    # abusive_language — clear-cut shaming terms
    {"phrase": "ನಾಚಿಕೆಗೇಡು",         "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "ಮೋಸಗಾರ",             "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "ದ್ರೋಹಿ",              "concern_type": ABUSIVE, "match": "word_boundary"},

    # third_party_contact_threat — family, employer, colleagues
    {"phrase": "ಹೆಂಡತಿಗೆ ಫೋನ್",      "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ಹೆಂಡತಿ ಮತ್ತು",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ತಂದೆಗೆ ಫೋನ್",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ಮನೆಯವರನ್ನು",          "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ಆಫೀಸಿನಲ್ಲಿ ಎಲ್ಲರಿಗೂ", "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ಆಫೀಸ್ನಲ್ಲಿ ಎಲ್ಲರಿಗೂ", "concern_type": THIRD_PARTY, "match": "substring"},

    # authority_or_legal_action_claim — police, court, arrest
    {"phrase": "ಪೊಲೀಸ್",             "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "ಕಂಪ್ಲೇಂಟ್",          "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "ಬಂಧಿಸ",              "concern_type": AUTHORITY, "match": "substring"},  # verb stem
    {"phrase": "ಕೋರ್ಟ್",             "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "ವಾರಂಟ್",             "concern_type": AUTHORITY, "match": "word_boundary"},
]

RULES_LATIN: list[dict] = [
    # intimidation_or_threats — physical visit / coercion
    {"phrase": "log aapke ghar aayenge",   "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "log aayenge",              "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ghar aayenge",             "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ghar pe aayenge",          "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "aadmi bhejenge",           "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ladke bhejenge",           "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "darwaaze par",             "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "darwaze par",              "concern_type": INTIMIDATION, "match": "substring"},
    {"phrase": "ghar ke bahar",            "concern_type": INTIMIDATION, "match": "substring"},

    # abusive_language — clear-cut slurs only
    {"phrase": "bewakoof",                 "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "bewakuf",                  "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "besharam",                 "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "kamina",                   "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "chor",                     "concern_type": ABUSIVE, "match": "word_boundary"},
    {"phrase": "dhokhebaaz",               "concern_type": ABUSIVE, "match": "word_boundary"},

    # third_party_contact_threat — family, employer, neighbours
    {"phrase": "wife ko batayenge",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "wife aur",                 "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "neighbours ko",            "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "company ko call",          "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "boss ko batayenge",        "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "manager ko batayenge",     "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ghar waalon ko",           "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ghar walon ko",            "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "pados ko",                 "concern_type": THIRD_PARTY, "match": "substring"},
    # catch generic "X ko call/batayenge" patterns with two common verbs
    {"phrase": "ko call karenge",          "concern_type": THIRD_PARTY, "match": "substring"},
    {"phrase": "ko batayenge",             "concern_type": THIRD_PARTY, "match": "substring"},

    # authority_or_legal_action_claim — police, FIR, court, arrest
    {"phrase": "police",                   "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "fir",                      "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "court mein case",          "concern_type": AUTHORITY, "match": "substring"},
    {"phrase": "court mein",               "concern_type": AUTHORITY, "match": "substring"},
    {"phrase": "warrant",                  "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "arrest",                   "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "giraftaar",                "concern_type": AUTHORITY, "match": "word_boundary"},
    {"phrase": "giraftar",                 "concern_type": AUTHORITY, "match": "word_boundary"},
]
