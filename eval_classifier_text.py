#!/usr/bin/env python3
"""Run the rules classifier on script TEXT (no audio) for one split and report call-level results.

    python eval_classifier_text.py --split dev
    python eval_classifier_text.py --split held_out

Scoring
-------
* Conduct categories (intimidation_or_threats, abusive_language, third_party_contact_threat):
  per-call TP/FP/FN against the manifest's concern_types. The false-flag rate on compliant
  (and hard-negative) calls counts ONLY these three categories.
* authority_or_legal_action_claim is a VERIFY flag: scored separately against the manifest's
  authority_mention (any agent mention of police, arrest, FIR, court or legal notice, lawful
  or not), with its own precision and recall.

Output: outputs/eval/classifier_text_report_<split>.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.classifier import classify
from src.concern_rules import ABUSIVE, AUTHORITY, INTIMIDATION, THIRD_PARTY

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "dataset_manifest.json"
SCRIPTS = {"dev": ROOT / "data" / "scripts_dev.json", "held_out": ROOT / "data" / "scripts_held_out.json"}
OUT_DIR = ROOT / "outputs" / "eval"
CONDUCT = [INTIMIDATION, ABUSIVE, THIRD_PARTY]


def turns_to_segments(scenario: dict) -> list[dict]:
    """One classifier segment per turn (text only, synthetic 6 s timing)."""
    return [{"segment_id": f"turn{i:02d}", "text": t["text"], "start_s": float(i * 6), "end_s": float((i + 1) * 6)}
            for i, t in enumerate(scenario["turns"])]


def prf(tp: int, fp: int, fn: int) -> tuple[str, str]:
    p = f"{tp / (tp + fp):.2f}" if tp + fp else "—"
    r = f"{tp / (tp + fn):.2f}" if tp + fn else "—"
    return p, r


def evaluate(split: str) -> dict:
    scripts = json.loads(SCRIPTS[split].read_text(encoding="utf-8"))["scenarios"]
    manifest = {c["id"]: c for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["calls"] if c["split"] == split}
    missing = [s["id"] for s in scripts if s["id"] not in manifest]
    if missing:
        raise SystemExit(f"scripts missing from manifest ({split}): {missing}")
    calls = []
    for sc in scripts:
        meta = manifest[sc["id"]]
        flags = classify(turns_to_segments(sc), meta["language"])
        for f in flags:
            f["turn"] = int(f["segment_id"][4:])
            f["turn_text"] = sc["turns"][f["turn"]]["text"]
            f["speaker"] = sc["turns"][f["turn"]]["speaker"]
        calls.append({"id": sc["id"], "meta": meta, "scenario": sc, "flags": flags,
                      "flagged": {f["concern_type"] for f in flags}})
    return {"split": split, "calls": calls}


def report(res: dict) -> str:
    split, calls = res["split"], res["calls"]
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    misses, false_flags = [], []
    for c in calls:
        meta, flagged = c["meta"], c["flagged"]
        expected = set(meta["concern_types"]) & set(CONDUCT)
        for cat in CONDUCT:
            if cat in expected and cat in flagged:
                tp[cat] += 1
            elif cat in expected:
                fn[cat] += 1
                misses.append((c, cat))
            elif cat in flagged:
                fp[cat] += 1
                false_flags.append((c, cat))
        exp_auth, got_auth = bool(meta.get("authority_mention")), AUTHORITY in flagged
        if exp_auth and got_auth:
            tp[AUTHORITY] += 1
        elif exp_auth:
            fn[AUTHORITY] += 1
            misses.append((c, AUTHORITY))
        elif got_auth:
            fp[AUTHORITY] += 1
            false_flags.append((c, AUTHORITY))

    compliant = [c for c in calls if c["meta"]["compliant"]]
    hard_neg = [c for c in calls if c["meta"]["hard_negative"]]
    conduct_ff = lambda cs: [c for c in cs if c["flagged"] & set(CONDUCT)]

    title = {"dev": "dev set", "held_out": "HELD-OUT set"}[split]
    L = [f"# Classifier text evaluation — {title}", ""]
    if split == "dev":
        L += ["> **DEV SET ONLY** — rules were built on these scripts. Do not report these numbers as headline results.", ""]
    else:
        L += ["> **Held-out set.** Rules were frozen (src/concern_rules.py, 2026-10-21) before these scripts were "
              "written, and labels were fixed (prepare_labels.py) before the classifier was run on them. "
              f"{len(calls)} short synthetic scripts evaluated on text only, so these are small counts, not a "
              "benchmark: one call moves a rate by 5–10 points.", ""]
    langs = sorted({c["meta"]["language"] for c in calls})
    L += [f"Calls: {len(calls)} ({', '.join(f'{l}: {sum(c['meta']['language'] == l for c in calls)}' for l in langs)}); "
          f"compliant {len(compliant)}, hard negatives {len(hard_neg)}, concerning {len(calls) - len(compliant)}. "
          "Scoring is per call: a category counts as flagged if any turn produced a flag of that type.", ""]

    L += ["## Conduct categories", "", "| Category | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
    for cat in CONDUCT:
        p, r = prf(tp[cat], fp[cat], fn[cat])
        L.append(f"| `{cat}` | {tp[cat]} | {fp[cat]} | {fn[cat]} | {p} | {r} |")
    p, r = prf(tp[AUTHORITY], fp[AUTHORITY], fn[AUTHORITY])
    L += ["", "## Authority / legal-action claims (verify flag, scored separately)", "",
          "Ground truth: the agent mentions police, arrest, FIR, court or a legal notice (lawful mentions in compliant "
          "calls included). A hit means the call was routed to a human to verify the claim; it is not a conduct finding.",
          "", "| | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|",
          f"| `{AUTHORITY}` | {tp[AUTHORITY]} | {fp[AUTHORITY]} | {fn[AUTHORITY]} | {p} | {r} |", ""]

    ffc, ffh = conduct_ff(compliant), conduct_ff(hard_neg)
    rate = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "—"
    auth_on_compliant = [c for c in compliant if AUTHORITY in c["flagged"]]
    L += ["## False flags on compliant calls (conduct categories only)", "",
          f"- Compliant calls with a conduct flag: **{rate(len(ffc), len(compliant))}**",
          f"- Hard-negative calls with a conduct flag: **{rate(len(ffh), len(hard_neg))}**",
          f"- Compliant calls with an authority verify flag: {rate(len(auth_on_compliant), len(compliant))} "
          f"({sum(bool(c['meta'].get('authority_mention')) for c in auth_on_compliant)} of them correctly — the agent "
          "did mention a legal notice / police / court)", ""]

    def call_tag(c) -> str:
        m = c["meta"]
        tags = [m["language"], "compliant" if m["compliant"] else "concerning"]
        if m["hard_negative"]:
            tags.append("hard negative")
        if m.get("needs_native_review"):
            tags.append("needs native review")
        return ", ".join(tags)

    L += ["## Every miss (false negative)", ""]
    if not misses:
        L.append("None.")
    for c, cat in misses:
        spans = [s for s in c["meta"].get("concern_spans", []) if s["type"] == cat]
        L.append(f"- **{c['id']}** ({call_tag(c)}) — missed `{cat}`")
        if spans:
            for s in spans:
                L.append(f"  - labelled span, turn {s['turn']}: “{s['text']}”")
        else:
            agent = [f"t{i}: {t['text']}" for i, t in enumerate(c["scenario"]["turns"]) if t["speaker"] == "agent"]
            L.append("  - no span labelled for this category (dev labels are call-level); agent turns: "
                     + " | ".join(agent))
        other = [f for f in c["flags"]]
        if other:
            L.append("  - flags it did raise: " + "; ".join(f"`{f['concern_type']}` on “{f['quote']}”" for f in other))
    L += ["", "## Every false flag", ""]
    if not false_flags:
        L.append("None.")
    for c, cat in false_flags:
        for f in [f for f in c["flags"] if f["concern_type"] == cat]:
            L.append(f"- **{c['id']}** ({call_tag(c)}) — `{cat}` on turn {f['turn']} ({f['speaker']}): quote “{f['quote']}”")
            L.append(f"  - turn text: “{f['turn_text']}”")

    bad = [(c["id"], f) for c in calls for f in c["flags"] if f["quote"] not in f["turn_text"]]
    L += ["", "## Quote check", "", f"- Flags: {sum(len(c['flags']) for c in calls)}; quotes not found verbatim in "
          f"their turn: {len(bad)}", ""]
    for cid, f in bad:
        L.append(f"  - {cid} turn {f['turn']}: {f['quote']!r}")

    L += ["## Per-call results", "", "| id | lang | compliant | hard neg | expected conduct | flagged conduct | "
          "authority expected | authority flagged | conduct correct |", "|---|---|---|---|---|---|---|---|---|"]
    for c in calls:
        m = c["meta"]
        exp = set(m["concern_types"]) & set(CONDUCT)
        got = c["flagged"] & set(CONDUCT)
        L.append(f"| {c['id']}{' ¹' if m.get('needs_native_review') else ''} | {m['language']} | "
                 f"{'✓' if m['compliant'] else ''} | {'✓' if m['hard_negative'] else ''} | "
                 f"{', '.join(sorted(exp)) or '—'} | {', '.join(sorted(got)) or '—'} | "
                 f"{'✓' if m.get('authority_mention') else ''} | {'✓' if AUTHORITY in c['flagged'] else ''} | "
                 f"{'✓' if exp == got else '✗'} |")
    L += ["", "¹ Kannada script — pending native speaker review; Kannada results are provisional.", "",
          "## Limitations", "",
          "- Text only: no TTS, no STT. Results on transcribed audio will be lower (spelling variants, omissions).",
          "- Call-level scoring: a category flagged on the wrong turn of a call that has that category still counts as TP.",
          "- The classifier sees every turn, including the borrower's (no speaker separation at runtime).",
          "- Small synthetic sets: treat every rate as a count of a handful of calls."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "held_out"], required=True)
    args = ap.parse_args()
    res = evaluate(args.split)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"classifier_text_report_{args.split}.md"
    out.write_text(report(res), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
