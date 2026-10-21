#!/usr/bin/env python3
"""Merge the spent held-out v1 set into dev (split=dev_v2) and apply the 2026-10-21 label rules.

    python prepare_labels.py

Inputs:  data/held_out_source/held_out_v1_external.json (kept byte-for-byte as delivered)
         data/scripts_dev.json, data/dataset_manifest.json (original dev entries)
Outputs: data/scripts_dev.json (original dev scenarios + the 20 converted v1 scenarios)
         data/dataset_manifest.json (dev entries gain authority fields; dev_v2 entries rebuilt)

Held-out v1 was evaluated once against rules v1 (docs/baseline_rules_v1.md) and is now
development data. A future held-out v2 will be written externally to data/scripts_held_out.json.

Label rules (applied by reading the scripts, never by running the classifier):
  1. hard_negative is true only for COMPLIANT calls containing firm-but-lawful language
     (credit-bureau / late-fee consequences, lawful legal notice, guarantor or co-borrower
     liability, consent-based visit offers). It is false on every concerning call.
  2. authority_mention is true, with one authority_or_legal_action_claim span per mention,
     whenever the AGENT mentions police, arrest, FIR, court or a legal notice, including in
     compliant calls. Borrower utterances (e.g. "Court?") are not agent claims and get no span.
     It is scored as a verify flag, separately from the three conduct categories.
  3. concern_types is the set of span types (a compliant call's lawful authority mention is
     a verify flag only and stays out of concern_types).
  4. (dev_v2 definition, applied to all dev scripts) telling or showing family, employer,
     neighbours or contacts = third_party_contact_threat; visiting, harming or shaming the
     borrower in person = intimidation_or_threats; one passage may carry both labels. A
     payment deadline or an arrest/police claim alone is not intimidation (the latter is the
     authority verify flag). The original 36 dev scripts already satisfy this definition.

Every decision is written out per call below so it can be reviewed line by line.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "held_out_source" / "held_out_v1_external.json"
DEV_SCRIPTS = ROOT / "data" / "scripts_dev.json"
SPLIT = "dev_v2"
MANIFEST = ROOT / "data" / "dataset_manifest.json"

AUTHORITY = "authority_or_legal_action_claim"
CONDUCT = ["intimidation_or_threats", "abusive_language", "third_party_contact_threat"]

# Same per-language conventions as data/scripts_dev.json.
LANG = {
    "hi-IN": {"tts_language": "hi-IN", "stt_language_codes": ["hi-IN"], "voices": {"agent": "Hemraj", "customer": "Chitra"},
              "name": "Hindi"},
    "kn-IN": {"tts_language": "kn-IN", "stt_language_codes": ["kn-IN"], "voices": {"agent": "Kavin", "customer": "Saanvi"},
              "name": "Kannada"},
    "en-IN": {"tts_language": "hi-en", "stt_language_codes": ["hi-IN", "en-IN"], "primary_stt_language_code": "en-IN",
              "voices": {"agent": "Poorvi", "customer": "Pranav"}, "name": "Hinglish"},
}

# Rule 1: firm-but-lawful content of each compliant held-out call (all ten qualify).
HARD_NEGATIVE_NOTES = {
    "script_001": "Credit bureau report consequence of continued default — lawful consequence",
    "script_002": "Late fee per loan agreement and credit score impact — lawful consequence",
    "script_005": "Notice to the named guarantor under the loan agreement — lawful third-party contact",
    "script_007": "Formal legal notice under the loan agreement, avoidable by payment — lawful process",
    "script_008": "Late fee and CIBIL report impact — lawful consequence",
    "script_011": "Legal notice process per policy, avoidable by payment; credit score impact — lawful process",
    "script_013": "Co-borrower joint liability under the agreement; credit score impact — lawful consequence",
    "script_014": "Late fee and CIBIL report impact — lawful consequence",
    "script_016": "Optional property visit offered only with consent — lawful, not a threat",
    "script_018": "Credit bureau status and NOC process — lawful consequence",
}

# Rule 2: agent authority mentions not already covered by an authority span in the source.
# (turn index, exact substring of that turn's text)
ADDED_AUTHORITY_SPANS = {
    "script_007": [(4, "हमने कानूनी नोटिस भेजने की तैयारी शुरू की है।")],
    "script_011": [(2, "ನಾವು ಕಾನೂನು ನೋಟಿಸ್ ಕಳಿಸೋ ಪ್ರಕ್ರಿಯೆ ಶುರು ಮಾಡಿದ್ದೀವಿ")],
    "script_019": [(8, "Tab tak arrest ki preparation karo.")],
}

# Rule 4 (dev_v2 definition). Spans removed (type, exact text) and added (type, turn, exact text).
DEFINITION_SPAN_REMOVALS = {
    "script_003": [("intimidation_or_threats", "हमारे लोग आपके घर आएँगे और आपके पड़ोसियों और मोहल्ले वालों को बता देंगे कि आप कर्ज़दार हैं और भुगतान से भाग रहे हैं।")],
    "script_004": [("intimidation_or_threats", "आज रात तक ट्रांसफर करें।")],
    "script_019": [("intimidation_or_threats", "Tab tak arrest ki preparation karo.")],
}
DEFINITION_SPAN_ADDITIONS = {
    "script_003": [("intimidation_or_threats", 4, "हमारे लोग आपके घर आएँगे"),
                   ("third_party_contact_threat", 4, "आपके पड़ोसियों और मोहल्ले वालों को बता देंगे कि आप कर्ज़दार हैं और भुगतान से भाग रहे हैं।")],
    "script_010": [("intimidation_or_threats", 4, "ನಿನ್ನ ಮಗನ ಶಾಲೆಗೂ ಹೋಗ್ತೀನಿ."),
                   ("third_party_contact_threat", 4, "ನಿನ್ನ ಮಗನ ಶಾಲೆಗೂ ಹೋಗ್ತೀನಿ.")],
    "script_012": [("third_party_contact_threat", 4, "ಅಕ್ಕ-ಪಕ್ಕದ ಮನೆಯವರ ಮುಂದೆ ಎಲ್ಲ ಬಯಲು ಮಾಡ್ತಾರೆ.")],
    "script_017": [("third_party_contact_threat", 6, "neighbours ko dikhayenge kin logo ne hamare saath fraud kiya.")],
    "script_020": [("third_party_contact_threat", 2, "building society ke log sab dekhenge.")],
}
DEFINITION_REASONS = {
    "script_003": "split: house visit = intimidation; telling neighbours = third-party (consistent with 017)",
    "script_004": "a payment deadline is not visiting/harming/shaming in person; the call stays concerning via its FIR/police claims",
    "script_010": "going to the son's school = in-person visit (intimidation) and contact with a third party",
    "script_012": "exposing the borrower before neighbours = third-party, in addition to the doorstep visit",
    "script_017": "showing neighbours = third-party, in addition to the house visit (consistent with 003)",
    "script_019": "an arrest claim alone is the authority verify flag, not intimidation",
    "script_020": "building society people seeing = showing neighbours (third-party); the house visit stays intimidation",
}

# Dev: every agent mention of police / arrest / FIR / court / legal notice (all already in
# authority-labelled concerning calls; no compliant dev call mentions them).
DEV_AUTHORITY_SPANS = {
    "dev_hi_concern_authority_01": [(4, "आज नहीं दिए तो हम पुलिस को सूचना देंगे और आपके ख़िलाफ़ एफ़आईआर दर्ज होगी।")],
    "dev_hi_concern_authority_02": [(4, "हमने आपके ख़िलाफ़ कोर्ट में केस दायर कर दिया है और समन आएगा।")],
    "dev_en_concern_authority_01": [(4, "Agar aaj nahi diya toh police complaint karenge aur aapko arrest karwayenge.")],
    "dev_en_concern_mixed_01": [(4, "Humne court mein case file kar diya hai, warrant aayega.")],
    "dev_kn_concern_authority_01": [(4, "ಹಾಗಾದರೆ ನಾವು ಪೊಲೀಸ್ ಕಂಪ್ಲೇಂಟ್ ಮಾಡುತ್ತೇವೆ ಮತ್ತು ನಿಮ್ಮನ್ನು ಬಂಧಿಸಲಾಗುತ್ತದೆ.")],
}

MANIFEST_NOTE = (
    "Ground-truth labels for the RecoveryCall QA dataset. 'split' is 'dev' (original dev set; rules v1 were built "
    "on it), 'dev_v2' (held-out v1, spent after one evaluation on 2026-10-21 and merged into development; see "
    "docs/baseline_rules_v1.md) or 'held_out' (a future externally written set; none exists now). Labels follow the "
    "definition in prepare_labels.py: telling/showing family, employer, neighbours or contacts = "
    "third_party_contact_threat; visiting/harming/shaming in person = intimidation_or_threats; multi-label allowed. 'compliant' is false when "
    "the call contains a conduct concern or a threatening authority claim. 'concern_types' lists the categories present. "
    "'authority_mention' is true when the agent mentions police, arrest, FIR, court or a legal notice (including lawful "
    "mentions in compliant calls); it is scored as a verify flag, separately from the three conduct categories. "
    "'concern_spans' quote the labelled text (turn index + exact substring); dev entries carry authority spans only. "
    "'hard_negative' is true only for compliant calls containing firm-but-lawful language that must not be flagged. "
    "'needs_native_review' is true for Kannada scripts pending native speaker review. 'label_changes' logs every "
    "change from the delivered labels."
)


def turn_index_of(turns: list[dict], text: str) -> int:
    hits = [i for i, t in enumerate(turns) if text in t["text"]]
    if len(hits) != 1:
        raise SystemExit(f"span text found in {len(hits)} turns (need exactly 1): {text!r}")
    return hits[0]


def convert_held_out() -> tuple[list[dict], list[dict]]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    scenarios, entries = [], []
    for s in source:
        sid, lang = s["id"], s["language"]
        conv = LANG[lang]
        turns = [{"speaker": "customer" if t["speaker"] == "borrower" else t["speaker"], "text": t["text"]}
                 for t in s["turns"]]
        scenario = {"id": sid, "title": f"Dev v2 (ex held-out v1) {conv['name']} {s['label']}",
                    "tts_language": conv["tts_language"], "stt_language_codes": conv["stt_language_codes"]}
        if "primary_stt_language_code" in conv:
            scenario["primary_stt_language_code"] = conv["primary_stt_language_code"]
        scenario.update({"voices": conv["voices"], "needs_native_review": lang == "kn-IN", "turns": turns})
        scenarios.append(scenario)

        changes = []
        spans = [{"type": sp["type"], "turn": turn_index_of(turns, sp["text"]), "text": sp["text"]}
                 for sp in s["concern_spans"]]
        for turn, text in ADDED_AUTHORITY_SPANS.get(sid, []):
            if text not in turns[turn]["text"] or turns[turn]["speaker"] != "agent":
                raise SystemExit(f"{sid}: added span is not in agent turn {turn}: {text!r}")
            spans.append({"type": AUTHORITY, "turn": turn, "text": text})
            changes.append(f"added {AUTHORITY} span (turn {turn})")

        for typ, text in DEFINITION_SPAN_REMOVALS.get(sid, []):
            before = len(spans)
            spans = [sp for sp in spans if not (sp["type"] == typ and sp["text"] == text)]
            if len(spans) != before - 1:
                raise SystemExit(f"{sid}: span to remove not found: {typ} {text!r}")
            changes.append(f"dev_v2 definition: removed {typ} span “{text}”")
        for typ, turn, text in DEFINITION_SPAN_ADDITIONS.get(sid, []):
            if text not in turns[turn]["text"] or turns[turn]["speaker"] != "agent":
                raise SystemExit(f"{sid}: definition span is not in agent turn {turn}: {text!r}")
            spans.append({"type": typ, "turn": turn, "text": text})
            changes.append(f"dev_v2 definition: added {typ} span (turn {turn}) “{text}”")
        if sid in DEFINITION_REASONS:
            changes.append(f"dev_v2 definition reason: {DEFINITION_REASONS[sid]}")

        compliant = s["label"] == "compliant"
        span_types = {sp["type"] for sp in spans} - ({AUTHORITY} if compliant else set())
        types = [t for t in s["concern_types"] if t in span_types] + sorted(span_types - set(s["concern_types"]))
        for t in sorted(set(s["concern_types"]) - span_types):
            changes.append(f"removed {t} from concern_types (no span of that type remains)")
        for t in sorted(span_types - set(s["concern_types"])):
            changes.append(f"added {t} to concern_types (a span of that type exists)")
        hard_negative = sid in HARD_NEGATIVE_NOTES
        if hard_negative != s["hard_negative"]:
            changes.append(f"hard_negative {s['hard_negative']} -> {hard_negative}"
                           + ("" if hard_negative else " (hard negatives are compliant calls only)"))
        if hard_negative and not compliant:
            raise SystemExit(f"{sid}: hard negative must be compliant")

        entry = {"id": sid, "split": SPLIT, "origin": "held_out_v1", "language": lang, "compliant": compliant,
                 "hard_negative": hard_negative, "concern_types": types,
                 "authority_mention": any(sp["type"] == AUTHORITY for sp in spans),
                 "concern_spans": spans, "needs_native_review": lang == "kn-IN"}
        if hard_negative:
            entry["hard_negative_note"] = HARD_NEGATIVE_NOTES[sid]
        if changes:
            entry["label_changes"] = changes
        entries.append(entry)
    return scenarios, entries


def update_dev(manifest: dict) -> None:
    dev_scripts = {s["id"]: s for s in json.loads(DEV_SCRIPTS.read_text(encoding="utf-8"))["scenarios"]}
    for c in manifest["calls"]:
        if c["split"] != "dev":
            continue
        spans = []
        for turn, text in DEV_AUTHORITY_SPANS.get(c["id"], []):
            t = dev_scripts[c["id"]]["turns"][turn]
            if text not in t["text"] or t["speaker"] != "agent":
                raise SystemExit(f"{c['id']}: dev authority span not in agent turn {turn}")
            spans.append({"type": AUTHORITY, "turn": turn, "text": text})
        c["authority_mention"] = bool(spans)
        c["concern_spans"] = spans
        if c["hard_negative"] and not c["compliant"]:
            raise SystemExit(f"{c['id']}: hard negative must be compliant")


def main() -> None:
    scenarios, v2_entries = convert_held_out()
    v2_ids = {e["id"] for e in v2_entries}
    dev = json.loads(DEV_SCRIPTS.read_text(encoding="utf-8"))
    dev["scenarios"] = [sc for sc in dev["scenarios"] if sc["id"] not in v2_ids] + scenarios
    dev["note"] = dev["note"].split(" DEV_V2:")[0] + (
        " DEV_V2: the last 20 scenarios (script_001..script_020) are held-out v1, spent after one evaluation on "
        "2026-10-21 (docs/baseline_rules_v1.md) and merged here by prepare_labels.py; STT critical_spans are not "
        "authored for them yet and their voices are per-language defaults.")
    DEV_SCRIPTS.write_text(json.dumps(dev, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["note"] = MANIFEST_NOTE
    manifest["calls"] = [c for c in manifest["calls"] if c["id"] not in v2_ids and c["split"] != "held_out"]
    update_dev(manifest)
    manifest["calls"] += v2_entries
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for e in v2_entries:
        for ch in e.get("label_changes", []):
            print(f"  {e['id']}: {ch}")
    print(f"merged {len(scenarios)} scenarios into {DEV_SCRIPTS.relative_to(ROOT)} (split={SPLIT}) and updated "
          f"{MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
