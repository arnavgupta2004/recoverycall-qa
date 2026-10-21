#!/usr/bin/env python3
"""Run the classifier against the dev set scripts' TEXT (no audio) and report per-category hits and false flags.

Usage:
    python eval_classifier_text.py

Output: outputs/eval/classifier_text_report.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.classifier import classify

ROOT = Path(__file__).resolve().parent
SCRIPTS_DEV = ROOT / "data" / "scripts_dev.json"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
OUT_DIR = ROOT / "outputs" / "eval"
OUT_REPORT = OUT_DIR / "classifier_text_report.md"


def turns_to_segments(scenario: dict) -> list[dict]:
    """Convert a scenario's turns into classifier-compatible segments (one per turn, no audio)."""
    segs = []
    for i, turn in enumerate(scenario["turns"]):
        segs.append({
            "segment_id": f"turn{i:02d}",
            "text": turn["text"],
            "start_s": float(i * 6),
            "end_s": float((i + 1) * 6),
        })
    return segs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scripts = json.loads(SCRIPTS_DEV.read_text(encoding="utf-8"))
    manifest_raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest: dict[str, dict] = {c["id"]: c for c in manifest_raw["calls"]}

    # Per-category counters
    cats = ["intimidation_or_threats", "abusive_language",
            "third_party_contact_threat", "authority_or_legal_action_claim"]

    # tp[cat]: flagged correctly on a call labelled with that category
    # fp[cat]: flagged on a call NOT labelled with that category
    # fn[cat]: not flagged on a call labelled with that category
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    false_flags_on_compliant: list[dict] = []
    false_flags_on_hard_neg: list[dict] = []
    per_call_results: list[dict] = []

    for sc in scripts["scenarios"]:
        sid = sc["id"]
        meta = manifest.get(sid)
        if meta is None:
            print(f"[WARN] {sid} not in manifest, skipping")
            continue

        lang = meta["language"]
        segments = turns_to_segments(sc)
        flags = classify(segments, lang)

        flagged_types = {f["concern_type"] for f in flags}
        expected_types = set(meta["concern_types"])

        per_call_results.append({
            "id": sid,
            "language": lang,
            "compliant": meta["compliant"],
            "hard_negative": meta.get("hard_negative", False),
            "expected_types": sorted(expected_types),
            "flagged_types": sorted(flagged_types),
            "flags": flags,
            "needs_native_review": meta.get("needs_native_review", False),
        })

        for cat in cats:
            expected = cat in expected_types
            flagged = cat in flagged_types
            if expected and flagged:
                tp[cat] += 1
            elif expected and not flagged:
                fn[cat] += 1
            elif not expected and flagged:
                fp[cat] += 1

        if meta["compliant"] and flagged_types:
            entry = {"id": sid, "language": lang, "hard_negative": meta.get("hard_negative", False), "flagged_types": sorted(flagged_types), "flags": flags}
            false_flags_on_compliant.append(entry)
            if meta.get("hard_negative"):
                false_flags_on_hard_neg.append(entry)

    # Quote existence check
    bad_quotes = []
    for call in per_call_results:
        sc_obj = next(s for s in scripts["scenarios"] if s["id"] == call["id"])
        all_text = " ".join(t["text"] for t in sc_obj["turns"])
        for flag in call["flags"]:
            turn_text = next(
                t["text"] for i, t in enumerate(sc_obj["turns"])
                if f"turn{i:02d}" == flag["segment_id"]
            )
            if flag["quote"] not in turn_text:
                bad_quotes.append({
                    "id": call["id"],
                    "segment_id": flag["segment_id"],
                    "quote": flag["quote"],
                    "turn_text": turn_text,
                })

    # Write report
    lines = [
        "# Classifier text evaluation — dev set",
        "",
        "> **DEV SET ONLY** — rules were built on these scripts. Do not report these numbers as headline results.",
        "",
        "## Per-category precision and recall",
        "",
        "| Category | TP | FP | FN | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for cat in cats:
        t, f_, n = tp[cat], fp[cat], fn[cat]
        prec = t / (t + f_) if (t + f_) else None
        rec = t / (t + n) if (t + n) else None
        prec_s = f"{prec:.2f}" if prec is not None else "—"
        rec_s = f"{rec:.2f}" if rec is not None else "—"
        lines.append(f"| `{cat}` | {t} | {f_} | {n} | {prec_s} | {rec_s} |")

    total_compliant = sum(1 for c in per_call_results if c["compliant"])
    total_hard_neg = sum(1 for c in per_call_results if c["hard_negative"])
    ff_compliant_rate = len(false_flags_on_compliant) / total_compliant if total_compliant else None
    ff_hard_neg_rate = len(false_flags_on_hard_neg) / total_hard_neg if total_hard_neg else None

    lines += [
        "",
        "## False-flag rates on compliant calls",
        "",
        f"- Compliant calls: {total_compliant}",
        f"- Compliant calls with any false flag: {len(false_flags_on_compliant)} ({ff_compliant_rate:.0%} of compliant)" if ff_compliant_rate is not None else "- No compliant calls.",
        f"- Hard-negative calls: {total_hard_neg}",
        f"- Hard-negative calls with any false flag: {len(false_flags_on_hard_neg)} ({ff_hard_neg_rate:.0%} of hard-negative)" if ff_hard_neg_rate is not None else "- No hard-negative calls.",
        "",
    ]

    if false_flags_on_compliant:
        lines += ["### False-flag details (compliant calls)", ""]
        for entry in false_flags_on_compliant:
            hn = " [HARD NEGATIVE]" if entry["hard_negative"] else ""
            lines.append(f"**{entry['id']}** ({entry['language']}){hn}")
            for flag in entry["flags"]:
                lines.append(f"  - `{flag['concern_type']}` — quote: `{flag['quote']}`")
            lines.append("")

    lines += [
        "## Quote existence check",
        "",
        f"- Total flags: {sum(len(c['flags']) for c in per_call_results)}",
        f"- Flags whose quote is NOT a substring of the turn text: {len(bad_quotes)}",
        "",
    ]
    if bad_quotes:
        lines += ["### Bad quotes", ""]
        for bq in bad_quotes:
            lines.append(f"- `{bq['id']}` / `{bq['segment_id']}`: quote `{bq['quote']!r}` not in `{bq['turn_text']!r}`")
        lines.append("")

    lines += [
        "## Per-call results",
        "",
        "| id | lang | compliant | hard_neg | expected | flagged | correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for call in per_call_results:
        exp = ", ".join(call["expected_types"]) or "—"
        fla = ", ".join(call["flagged_types"]) or "—"
        correct = set(call["expected_types"]) == set(call["flagged_types"])
        mark = "✓" if correct else "✗"
        hn = "✓" if call["hard_negative"] else ""
        comp = "✓" if call["compliant"] else ""
        nr = " ¹" if call["needs_native_review"] else ""
        lines.append(f"| {call['id']}{nr} | {call['language']} | {comp} | {hn} | {exp} | {fla} | {mark} |")

    lines += [
        "",
        "¹ Kannada script — flagged for native speaker review.",
        "",
        "## Limitations (dev set)",
        "",
        "- Rules are NOT frozen. This report is an iteration aid, not a headline result.",
        "- All matching is done on plain text (no audio, no STT). Final evaluation runs on STT output.",
        "- Short turns (≤ 3 words) may not contain concern language; this is expected and not counted as FN.",
        "- Kannada results are provisional pending native speaker review.",
    ]

    report_text = "\n".join(lines) + "\n"
    OUT_REPORT.write_text(report_text, encoding="utf-8")
    print(f"Written: {OUT_REPORT}")
    print()

    # Print summary to stdout
    print("=== Dev-set classifier text evaluation ===")
    print(f"{'Category':<38} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Recall':>7}")
    print("-" * 70)
    for cat in cats:
        t, f_, n = tp[cat], fp[cat], fn[cat]
        prec = t / (t + f_) if (t + f_) else None
        rec = t / (t + n) if (t + n) else None
        prec_s = f"{prec:.2f}" if prec is not None else "  — "
        rec_s = f"{rec:.2f}" if rec is not None else "   — "
        print(f"{cat:<38} {t:>4} {f_:>4} {n:>4} {prec_s:>6} {rec_s:>7}")
    print("-" * 70)
    print(f"False flags on compliant calls: {len(false_flags_on_compliant)}/{total_compliant}")
    print(f"False flags on hard-negative calls: {len(false_flags_on_hard_neg)}/{total_hard_neg}")
    print(f"Bad quotes: {len(bad_quotes)}")


if __name__ == "__main__":
    main()
