#!/usr/bin/env python3
"""Compare rules-only, LLM-only and hybrid tagging on script TEXT (no audio) for one split.

    python eval_classifier_text.py --split dev --llm-mock     # offline, no key
    python eval_classifier_text.py --split dev                # real LLM (GEMINI_API_KEY), cached
    python eval_classifier_text.py --split dev --no-llm       # rules only

Splits: "dev" = original dev + dev_v2 (spent held-out v1) from data/scripts_dev.json; dev numbers
are context only, never headline results. "held_out" = data/scripts_held_out.json (a future,
externally written v2; none exists yet).

Scoring
-------
* Conduct categories (intimidation_or_threats, abusive_language, third_party_contact_threat):
  per-call TP/FP/FN against the manifest's concern_types. The false-flag rate on compliant
  (and hard-negative) calls counts ONLY these three categories.
* authority_or_legal_action_claim is a VERIFY flag, scored separately against the manifest's
  authority_mention (any agent mention of police, arrest, FIR, court or legal notice).
* LLM flags whose quote is not a verbatim substring of the named segment are rejected by the
  quote verifier before scoring; rejections are reported.

Output: outputs/eval/classifier_compare_<split>[_mock].md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.classifier import classify, verify_quote
from src.concern_rules import ABUSIVE, AUTHORITY, INTIMIDATION, THIRD_PARTY
from src.gnani_client import ConfigError
from src.hybrid import merge_flags
from src.llm_tagger import make_backend, tag

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "dataset_manifest.json"
SPLITS = {"dev": (ROOT / "data" / "scripts_dev.json", {"dev", "dev_v2"}),
          "held_out": (ROOT / "data" / "scripts_held_out.json", {"held_out"})}
OUT_DIR = ROOT / "outputs" / "eval"
CONDUCT = [INTIMIDATION, ABUSIVE, THIRD_PARTY]
CATS = CONDUCT + [AUTHORITY]


def turns_to_segments(scenario: dict) -> list[dict]:
    """One segment per turn (text only, synthetic 6 s timing)."""
    return [{"segment_id": f"turn{i:02d}", "text": t["text"], "start_s": float(i * 6), "end_s": float((i + 1) * 6)}
            for i, t in enumerate(scenario["turns"])]


def run(split: str, backend, cache_dir: Path) -> tuple[list[dict], list[str]]:
    path, splits = SPLITS[split]
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} does not exist (no {split} set has been written yet)")
    manifest = {c["id"]: c for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["calls"] if c["split"] in splits}
    scenarios = [s for s in json.loads(path.read_text(encoding="utf-8"))["scenarios"] if s["id"] in manifest]
    missing = set(manifest) - {s["id"] for s in scenarios}
    if missing:
        raise SystemExit(f"manifest calls without scripts: {sorted(missing)}")
    modes = ["rules"] + (["llm", "hybrid"] if backend is not None else [])
    calls = []
    for sc in scenarios:
        meta, segs = manifest[sc["id"]], turns_to_segments(sc)
        rules = classify(segs, meta["language"])
        out = {"id": sc["id"], "meta": meta, "scenario": sc, "segments": segs, "flags": {"rules": rules}}
        if backend is not None:
            res = tag(segs, meta["language"], backend, cache_dir)
            out.update(llm_rejected=res["rejected"], llm_error=res["error"], llm_from_cache=res["from_cache"])
            out["flags"]["llm"] = res["flags"]
            out["flags"]["hybrid"] = merge_flags(rules, res["flags"], segs)
        calls.append(out)
    return calls, modes


def score(calls: list[dict], mode: str) -> dict:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    misses, false_flags = [], []
    for c in calls:
        got = {f["concern_type"] for f in c["flags"][mode]}
        expected = (set(c["meta"]["concern_types"]) & set(CONDUCT)) | ({AUTHORITY} if c["meta"].get("authority_mention") else set())
        for cat in CATS:
            if cat in expected and cat in got:
                tp[cat] += 1
            elif cat in expected:
                fn[cat] += 1
                misses.append((c, cat))
            elif cat in got:
                fp[cat] += 1
                false_flags.append((c, cat))
    compliant = [c for c in calls if c["meta"]["compliant"]]
    hard = [c for c in calls if c["meta"]["hard_negative"]]
    conduct_hit = lambda c: any(f["concern_type"] in CONDUCT for f in c["flags"][mode])
    return {"tp": tp, "fp": fp, "fn": fn, "misses": misses, "false_flags": false_flags,
            "ff_compliant": [c for c in compliant if conduct_hit(c)], "n_compliant": len(compliant),
            "ff_hard": [c for c in hard if conduct_hit(c)], "n_hard": len(hard),
            "bad_quotes": [(c["id"], f) for c in calls for f in c["flags"][mode]
                           if not verify_quote(f["quote"], next(s["text"] for s in c["segments"]
                                                                if s["segment_id"] == f["segment_id"]))]}


def ratio(a: int, b: int) -> str:
    return f"{a / b:.2f}" if b else "—"


def tag_of(c: dict) -> str:
    m = c["meta"]
    t = [m["language"], m["split"], "compliant" if m["compliant"] else "concerning"]
    if m["hard_negative"]:
        t.append("hard negative")
    if m.get("needs_native_review"):
        t.append("needs native review")
    return ", ".join(t)


def report(split: str, calls: list[dict], modes: list[str], backend_name: str | None, mock: bool) -> str:
    S = {m: score(calls, m) for m in modes}
    L = [f"# Rules vs LLM vs hybrid — {split} set (text only)", ""]
    if split == "dev":
        L += ["> **DEV SET — context only.** Rules v1 were built on the original dev scripts, and dev_v2 is held-out v1, "
              "already spent (baseline in docs/baseline_rules_v1.md). Do not report these numbers as headline results.", ""]
    if mock:
        L += ["> **MOCK LLM.** The LLM column comes from a keyword mock that also respells some quotes on purpose; it tests "
              "the pipeline (verifier, cache, merge), not any model.", ""]
    langs = Counter(c["meta"]["language"] for c in calls)
    splits = Counter(c["meta"]["split"] for c in calls)
    L += [f"Calls: {len(calls)} ({', '.join(f'{k}: {v}' for k, v in sorted(splits.items()))}; "
          f"{', '.join(f'{k}: {v}' for k, v in sorted(langs.items()))}). LLM backend: {backend_name or 'not run'}. "
          "Per-call scoring: a category counts as flagged if any segment produced a flag of that type.", ""]

    L += ["## Per-category TP / FP / FN", "",
          "| category | " + " | ".join(f"{m} TP/FP/FN | {m} P | {m} R" for m in modes) + " |",
          "|---|" + "---|---|---|" * len(modes)]
    for cat in CATS:
        row = [f"`{cat}`" + (" (verify flag)" if cat == AUTHORITY else "")]
        for m in modes:
            t, f, n = S[m]["tp"][cat], S[m]["fp"][cat], S[m]["fn"][cat]
            row += [f"{t}/{f}/{n}", ratio(t, t + f), ratio(t, t + n)]
        L.append("| " + " | ".join(row) + " |")
    row = ["**conduct total**"]
    for m in modes:
        t, f, n = (sum(S[m][k][c] for c in CONDUCT) for k in ("tp", "fp", "fn"))
        row += [f"**{t}/{f}/{n}**", ratio(t, t + f), ratio(t, t + n)]
    L.append("| " + " | ".join(row) + " |")

    L += ["", "## False flags on compliant calls (conduct categories only)", "",
          "| | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes),
          "| compliant calls with a conduct flag | " + " | ".join(
              f"{len(S[m]['ff_compliant'])}/{S[m]['n_compliant']}" for m in modes) + " |",
          "| hard-negative calls with a conduct flag | " + " | ".join(
              f"{len(S[m]['ff_hard'])}/{S[m]['n_hard']}" for m in modes) + " |",
          "| flags whose quote is not verbatim (after verification) | " + " | ".join(
              str(len(S[m]["bad_quotes"])) for m in modes) + " |", ""]

    if "llm" in modes:
        rej = [(c, r) for c in calls for r in c["llm_rejected"]]
        errs = [(c["id"], c["llm_error"]) for c in calls if c["llm_error"]]
        n_out = sum(len(c["flags"]["llm"]) for c in calls) + len(rej)
        L += ["## Quote-verifier rejections (LLM)", "",
              f"{len(rej)} of {n_out} LLM outputs rejected "
              f"({', '.join(f'{k}: {v}' for k, v in Counter(r['reason'] for _, r in rej).items()) or 'none'}). "
              f"Responses from cache: {sum(c['llm_from_cache'] for c in calls)}/{len(calls)}.", ""]
        for c, r in rej:
            L.append(f"- {c['id']} {r.get('segment_id')} `{r.get('type')}` — {r['reason']}: “{r.get('quote')}”")
        if errs:
            L += ["", "**LLM call errors** (these calls have no LLM flags):", ""]
            L += [f"- {cid}: {e}" for cid, e in errs]
        hy = Counter(f["source"] for c in calls for f in c["flags"]["hybrid"])
        L += ["", f"Hybrid flag sources: {dict(hy)} (rules+llm = both found an overlapping span of the same type).", ""]

    for m in modes:
        L += [f"## {m}: every miss and false flag", ""]
        for c, cat in S[m]["misses"]:
            spans = [s for s in c["meta"].get("concern_spans", []) if s["type"] == cat]
            where = "; ".join(f"t{s['turn']} “{s['text']}”" for s in spans) or "no span labelled (call-level dev label)"
            L.append(f"- MISS **{c['id']}** ({tag_of(c)}) `{cat}` — {where}")
        for c, cat in S[m]["false_flags"]:
            for f in [f for f in c["flags"][m] if f["concern_type"] == cat]:
                L.append(f"- FALSE FLAG **{c['id']}** ({tag_of(c)}) `{cat}` on {f['segment_id']}: “{f['quote']}”")
        if not S[m]["misses"] and not S[m]["false_flags"]:
            L.append("None.")
        L.append("")

    L += ["## Limitations", "",
          "- Text only: no TTS or STT. Transcribed audio will be harder (spelling variants, omissions).",
          "- Call-level scoring: a category flagged on the wrong segment of a call that has it still counts as a TP.",
          "- The taggers see every turn, including the borrower's; there is no speaker separation at runtime.",
          "- Small synthetic set; Kannada scripts await native review. Treat every number as a handful of calls."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--llm-mock", action="store_true", help="use the offline mock LLM")
    g.add_argument("--no-llm", action="store_true", help="rules only")
    ap.add_argument("--llm-cache", type=Path, default=None,
                    help="LLM response cache dir (default outputs/llm_cache, git-ignored). Entries hold only the "
                         "model's JSON text, backend name, prompt version and language: no request ids, "
                         "signatures, headers or keys")
    args = ap.parse_args()
    backend = None
    if not args.no_llm:
        try:
            backend = make_backend(args.llm_mock, ROOT / ".env")
        except ConfigError as exc:
            sys.exit(f"error: {exc} For this script: --llm-mock (offline) or --no-llm (rules only).")
    cache_dir = args.llm_cache or ROOT / "outputs" / ("llm_cache_mock" if args.llm_mock else "llm_cache")
    t0 = time.monotonic()
    calls, modes = run(args.split, backend, cache_dir)
    elapsed = time.monotonic() - t0
    if backend is not None:
        api_calls = getattr(backend, "calls", 0)
        events = getattr(backend, "events", [])
        print(f"LLM: {api_calls} API request(s) incl. retries, {sum(c['llm_from_cache'] for c in calls)} call(s) "
              f"from cache, {len(events)} non-200 attempt(s){': ' + '; '.join(events) if events else ''}; "
              f"total run time {elapsed:.0f} s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"classifier_compare_{args.split}{'_mock' if args.llm_mock else ''}.md"
    out.write_text(report(args.split, calls, modes, backend.name if backend else None, args.llm_mock), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
