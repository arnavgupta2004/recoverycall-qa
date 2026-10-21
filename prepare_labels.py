#!/usr/bin/env python3
"""Convert the externally written held-out set to our schema and apply the 2026-10-21 label rules.

    python prepare_labels.py

Inputs:  data/held_out_source/scripts_held_out_external.json (kept byte-for-byte as delivered)
         data/dataset_manifest.json (dev entries)
Outputs: data/scripts_held_out.json (same schema as data/scripts_dev.json)
         data/dataset_manifest.json (dev entries gain authority fields; held-out entries rebuilt)

Label rules (applied by reading the scripts, never by running the classifier):
  1. hard_negative is true only for COMPLIANT calls containing firm-but-lawful language
     (credit-bureau / late-fee consequences, lawful legal notice, guarantor or co-borrower
     liability, consent-based visit offers). It is false on every concerning call.
  2. authority_mention is true, with one authority_or_legal_action_claim span per mention,
     whenever the AGENT mentions police, arrest, FIR, court or a legal notice, including in
     compliant calls. Borrower utterances (e.g. "Court?") are not agent claims and get no span.
     It is scored as a verify flag, separately from the three conduct categories.
  3. concern_types must include every type that has a concern span.

Every decision is written out per call below so it can be reviewed line by line.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "held_out_source" / "scripts_held_out_external.json"
OUT_SCRIPTS = ROOT / "data" / "scripts_held_out.json"
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
    "Ground-truth labels for the RecoveryCall QA dataset. 'split' is 'dev' (rules were built on this) or "
    "'held_out' (headline evaluation; labels fixed before the classifier was run on it). 'compliant' is false when "
    "the call contains a conduct concern or a threatening authority claim. 'concern_types' lists the categories present. "
    "'authority_mention' is true when the agent mentions police, arrest, FIR, court or a legal notice (including lawful "
    "mentions in compliant calls); it is scored as a verify flag, separately from the three conduct categories. "
    "'concern_spans' quote the labelled text (turn index + exact substring); dev entries carry authority spans only. "
    "'hard_negative' is true only for compliant calls containing firm-but-lawful language that must not be flagged. "
    "'needs_native_review' is true for Kannada scripts pending native speaker review."
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
        scenario = {"id": sid, "title": f"Held-out {conv['name']} {s['label']}",
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

        compliant = s["label"] == "compliant"
        types = list(s["concern_types"])
        for t in sorted({sp["type"] for sp in spans} - set(types)):
            if compliant and t == AUTHORITY:
                continue  # lawful authority mention in a compliant call: verify flag only
            types.append(t)
            changes.append(f"added {t} to concern_types (a span of that type exists)")
        hard_negative = sid in HARD_NEGATIVE_NOTES
        if hard_negative != s["hard_negative"]:
            changes.append(f"hard_negative {s['hard_negative']} -> {hard_negative}"
                           + ("" if hard_negative else " (hard negatives are compliant calls only)"))
        if hard_negative and not compliant:
            raise SystemExit(f"{sid}: hard negative must be compliant")

        entry = {"id": sid, "split": "held_out", "language": lang, "compliant": compliant,
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
    dev_scripts = {s["id"]: s for s in json.loads((ROOT / "data" / "scripts_dev.json").read_text(encoding="utf-8"))["scenarios"]}
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
    scenarios, held_entries = convert_held_out()
    OUT_SCRIPTS.write_text(json.dumps({
        "version": 1,
        "note": ("HELD-OUT set: written externally after the concern rules were frozen (2026-10-21), converted from "
                 "data/held_out_source/scripts_held_out_external.json by prepare_labels.py. Headline evaluation only; "
                 "never tune rules on it. All calls, people, companies and amounts are fictional. STT critical_spans "
                 "(amount/negation) are not authored yet; voices are per-language defaults (review speaker gender "
                 "before synthesis)."),
        "tts_model": "timbre-v2.5", "tts_speed": 1.0, "scenarios": scenarios,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["note"] = MANIFEST_NOTE
    update_dev(manifest)
    manifest["calls"] = [c for c in manifest["calls"] if c["split"] != "held_out"] + held_entries
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for e in held_entries:
        for ch in e.get("label_changes", []):
            print(f"  {e['id']}: {ch}")
    print(f"wrote {OUT_SCRIPTS.relative_to(ROOT)} ({len(scenarios)} scenarios) and updated {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
