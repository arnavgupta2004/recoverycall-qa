#!/usr/bin/env python3
"""End-to-end (audio) evaluation on held-out v3: Timbre TTS -> phone degradation -> the app pipeline.

    python eval_e2e.py run    [--mock]   # TTS (cached) + degrade + src.pipeline.analyse per call and variant
    python eval_e2e.py report [--mock]   # score outputs/e2e_v3/results.json -> docs/held_out_v3_results.md

Per call: each turn is synthesized with Gnani Timbre v2.5 (per-language default voices), turns are
concatenated with 0.6 s gaps, then two phone-quality variants are made:
  phone_clean     300-3400 Hz band-limit, 8 kHz, G.711 mu-law encode/decode
  phone_noisy10   the same plus band-limited pink noise at 10 dB SNR (measured over speech) before mu-law
Each variant goes through src.pipeline.analyse exactly as the app runs it (energy VAD segments <= 12 s ->
Gnani Prisma STT -> coverage check + re-split recovery -> frozen rules, Gemini tagger with quote
verification, hybrid merge). No summary audio is made. The reference script is used only to build the
audio (turn offsets, noise level) and in `report` for scoring.

The run stops on HTTP 403 from any API, or when two requests in a row still fail with 429/5xx (or a
transport error) after the client's own retries. Held-out v3 is evaluated once: never tune rules, prompts
or labels on these results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from src import audio
from src.classifier import verify_quote
from src.concern_rules import ABUSIVE, AUTHORITY, INTIMIDATION, THIRD_PARTY
from src.gnani_client import GnaniClient, MockSession, load_api_key
from src.llm_tagger import make_backend
from src.metrics import turn_missing
from src.pipeline import Services, _gnani_transcriber, analyse, mock_services

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"
SCRIPTS = ROOT / "data" / "scripts_held_out_v3.json"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
DOC = ROOT / "docs" / "held_out_v3_results.md"
VARIANTS = ["phone_clean", "phone_noisy10"]
SNR_DB = 10.0
APP_LANGUAGE = {"hi-IN": "hi-IN", "kn-IN": "kn-IN", "hi-en": "en-IN"}   # TTS language -> app/STT language
CONDUCT = [INTIMIDATION, ABUSIVE, THIRD_PARTY]
CATS = CONDUCT + [AUTHORITY]
MODES = {"rules": {"rules", "both"}, "llm": {"LLM", "both"}, "hybrid": {"rules", "LLM", "both"}}
LANG_ORDER = ["hi-IN", "kn-IN", "en-IN"]
LANG_NAME = {"hi-IN": "Hindi", "kn-IN": "Kannada", "en-IN": "Hinglish"}
FAIL_STATUSES = {None, 429, 500, 502, 503, 504}


class Abort(RuntimeError):
    pass


class Guard:
    """Stop on 403, or on two consecutive requests that still fail with 429/5xx after client retries."""

    def __init__(self):
        self.streak = 0
        self.events: list[str] = []

    def check(self, api: str, status: int | None, ok: bool) -> None:
        if ok:
            self.streak = 0
            return
        self.events.append(f"{api} HTTP {status}")
        if status == 403:
            raise Abort(f"{api} returned 403")
        if status in FAIL_STATUSES:
            self.streak += 1
            if self.streak >= 2:
                raise Abort(f"{api}: two consecutive requests failed ({', '.join(self.events[-2:])})")


def out_dir(mock: bool) -> Path:
    return ROOT / "outputs" / ("mock" if mock else "") / "e2e_v3"


def load_set() -> tuple[list[dict], dict]:
    scenarios = json.loads(SCRIPTS.read_text(encoding="utf-8"))["scenarios"]
    meta = {c["id"]: c for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["calls"] if c["split"] == "held_out_v3"}
    if {s["id"] for s in scenarios} != set(meta):
        raise SystemExit("scripts_held_out_v3.json and the manifest's held_out_v3 entries disagree")
    return scenarios, meta


# --------------------------------------------------------------------------- run


def synthesize_call(sc: dict, client: GnaniClient, guard: Guard, tts_cache: Path, cdir: Path,
                    counts: Counter) -> list[tuple[float, float]]:
    wavs = []
    for i, turn in enumerate(sc["turns"]):
        voice, lang = sc["voices"][turn["speaker"]], sc["tts_language"]
        payload = {"text": turn["text"], "voice": voice, "language": lang, "model": "timbre-v2.5", "speed": 1.0}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
        raw = tts_cache / sc["id"] / f"turn{i:02d}_{digest}.wav"
        if raw.exists():
            counts["tts_cached"] += 1
        else:
            res = client.synthesize(turn["text"], voice=voice, language=lang)
            counts["tts_calls"] += 1
            guard.check("TTS", res.http_status, res.ok)
            if not res.ok:
                raise Abort(f"TTS failed for {sc['id']} turn {i}: HTTP {res.http_status} {(res.error or {}).get('type')}")
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(res.audio)
        norm = cdir / "turns" / f"turn{i:02d}.wav"
        audio.normalize_turn(raw, norm)
        wavs.append(norm)
    return audio.concatenate(wavs, cdir / "clean.wav")


def degrade(cdir: Path, offsets: list[tuple[float, float]], seed: int) -> dict:
    audio.bandlimit_8k(cdir / "clean.wav", cdir / "pre.wav")
    audio.encode_ulaw(cdir / "pre.wav", cdir / "phone_clean_ulaw.wav")
    audio.decode_to_pcm(cdir / "phone_clean_ulaw.wav", cdir / "phone_clean.wav")
    pre, sr = audio.read_wav(cdir / "pre.wav")
    noisy, snr = audio.mix_at_snr(pre, sr, offsets, SNR_DB, seed=seed)
    audio.write_wav(cdir / "noisy_pre.wav", noisy, sr)
    audio.encode_ulaw(cdir / "noisy_pre.wav", cdir / "phone_noisy10_ulaw.wav")
    audio.decode_to_pcm(cdir / "phone_noisy10_ulaw.wav", cdir / "phone_noisy10.wav")
    return snr


def live_eval_services(base: Path, guard: Guard) -> tuple[Services, GnaniClient, object]:
    counters = {"stt": 0, "stt_cache_hits": 0, "llm": 0, "tts": 0}
    client = GnaniClient(api_key=load_api_key(ENV, "GNANI_API_KEY"))
    stt = _gnani_transcriber(client, base / "stt_cache", counters)

    def transcribe(path: Path, language: str, start_s: float, end_s: float) -> dict:
        rec = stt(path, language, start_s, end_s)
        guard.check("STT", rec["http_status"], rec["ok"])
        return rec

    backend = make_backend(False, ENV, read_timeout_s=60.0)
    return Services(transcribe, backend, None, base / "llm_cache", counters), client, backend


def cmd_run(args) -> None:
    audio.require_ffmpeg()
    scenarios, meta = load_set()
    base = out_dir(args.mock)
    base.mkdir(parents=True, exist_ok=True)
    guard, counts = Guard(), Counter()
    if args.mock:
        client, backend = GnaniClient(api_key="mock-key", session=MockSession(), backoff_base_s=0.0), None
    else:
        services, client, backend = live_eval_services(base, guard)
    tts_cache = ROOT / "cache" / ("tts_v3_mock" if args.mock else "tts_v3")
    calls, aborted, t0, mock_counts = [], None, time.monotonic(), Counter()
    try:
        for n, sc in enumerate(scenarios):
            cdir = base / "audio" / sc["id"]
            offsets = synthesize_call(sc, client, guard, tts_cache, cdir, counts)
            snr = degrade(cdir, offsets, seed=3000 + n)
            lang = APP_LANGUAGE[sc["tts_language"]]
            ref = [{"turn": i, "speaker": t["speaker"], "text": t["text"], "start_s": round(a, 3), "end_s": round(b, 3)}
                   for i, (t, (a, b)) in enumerate(zip(sc["turns"], offsets))]
            if args.mock:
                services = mock_services(ref)
            out = {"id": sc["id"], "language": lang, "turns": ref, "snr": snr, "variants": {}}
            for v in VARIANTS:
                if not args.mock:
                    n_llm_before, events_before = backend.calls, len(backend.events)
                res = analyse(cdir / f"{v}.wav", lang, services, make_summary_audio=False, llm_timeout_s=240.0)
                res.pop("summary", None)
                status = res["detectors"]["llm_status"]
                if not args.mock:
                    new_events = backend.events[events_before:]
                    if any("HTTP 403" in e for e in new_events):
                        raise Abort("Gemini returned 403")
                    guard.check("LLM", None if status != "ok" else 200, status == "ok")
                    res["llm_requests"] = backend.calls - n_llm_before
                    res["llm_events"] = new_events
                out["variants"][v] = res
                print(f"  {sc['id']} {v:<13} segs {len(res['segments'])} "
                      f"flagged {sum(s['flagged'] for s in res['segments'])} recovered "
                      f"{sum(s['recovered'] for s in res['segments'])} flags {len(res['flags'])} llm {status}", flush=True)
            calls.append(out)
            if args.mock:
                mock_counts.update(services.counters)
    except Abort as exc:
        aborted = str(exc)
        print(f"ABORTED: {aborted}", flush=True)
    counters = mock_counts if args.mock else services.counters
    api = {"tts_calls": counts["tts_calls"], "tts_cached": counts["tts_cached"],
           "stt_calls": counters.get("stt", 0), "stt_cache_hits": counters.get("stt_cache_hits", 0),
           "llm_requests_incl_retries": getattr(backend, "calls", None),
           "llm_non_200_attempts": list(getattr(backend, "events", [])), "guard_events": guard.events}
    (base / "results.json").write_text(json.dumps(
        {"aborted": aborted, "mock": args.mock, "run_seconds": round(time.monotonic() - t0, 1),
         "llm_backend": getattr(backend, "name", "mock"), "api": api, "calls": calls},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"run: {len(calls)}/{len(scenarios)} calls; API {json.dumps(api, ensure_ascii=False)}")
    if aborted:
        sys.exit(1)


# --------------------------------------------------------------------------- report


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def mode_types(res: dict, mode: str) -> set[str]:
    return {f["concern_type"] for f in res["flags"] if f["detector"] in MODES[mode]}


def expected(m: dict) -> set[str]:
    return (set(m["concern_types"]) & set(CONDUCT)) | ({AUTHORITY} if m["authority_mention"] else set())


def omission_rows(call: dict, res: dict) -> list[dict]:
    rows = []
    for t in call["turns"]:
        ts, te = t["start_s"], t["end_s"]
        over = [g for g in res["segments"] if overlap(ts, te, g["start_s"], g["end_s"]) > 0]
        before = " ".join(g["transcript_first"] or "" for g in over)
        after = " ".join(r["text"] for r in res["transcript"] if overlap(ts, te, r["start_s"], r["end_s"]) > 0)
        d0, r0 = turn_missing(t["text"], before)
        d1, r1 = turn_missing(t["text"], after)
        flagged = sum(overlap(ts, te, g["start_s"], g["end_s"]) for g in over if g["flagged"]) / (te - ts) >= 0.5
        surfaced = any(overlap(ts, te, w["start_s"], w["end_s"]) > 0 for w in res["coverage_warnings"]
                       if w["kind"] == "untranscribed")
        rows.append({"turn": t["turn"], "speaker": t["speaker"], "omitted": d0, "detected": d0 and flagged,
                     "recovered": d0 and not d1, "missing_after": d1, "surfaced_after": d1 and surfaced,
                     "ratio_before": round(r0, 2), "ratio_after": round(r1, 2), "text": t["text"]})
    return rows


def cmd_report(args) -> None:
    data = json.loads((out_dir(args.mock) / "results.json").read_text(encoding="utf-8"))
    _, meta = load_set()
    calls = data["calls"]
    L = ["# Held-out v3 — end-to-end audio evaluation (run once)", ""]
    if data["mock"]:
        L += ["> **MOCK RUN** — tone TTS, reference-text STT and a keyword LLM mock. Tests the harness only.", ""]
    if data["aborted"]:
        L += [f"> **Run aborted:** {data['aborted']}. Results below cover {len(calls)} of 18 calls.", ""]
    L += ["18 externally written calls (6 per language: 3 concerning, 3 compliant; labels checked by reading and "
          "committed in 9ff7cdc before any audio was made). Pipeline per call: Gnani Timbre v2.5 TTS per turn → "
          "concatenation → **phone_clean** (300–3400 Hz, 8 kHz, G.711 μ-law) and **phone_noisy10** (the same plus "
          "band-limited pink noise at 10 dB SNR) → the app's `src.pipeline.analyse`: energy VAD segments ≤ 12 s → "
          "Gnani Prisma STT → coverage check + re-split recovery → frozen rules v1 / Gemini tagger with quote "
          f"verification / hybrid. LLM backend: `{data['llm_backend']}`. Run time {data['run_seconds']:.0f} s.", "",
          "The six concerning/compliant situations are each written once per language (HI00k ≈ KN00k ≈ HX00k), so "
          "per-language numbers are three renderings of the same six situations, not independent samples. Kannada "
          "scripts await native-speaker review. Every number below is a count over a handful of calls.", ""]

    L += ["## Summary", "", "| variant | method | conduct TP/FP/FN | compliant false flags | hard-negative false flags "
          "| authority (verify) TP/FP/FN |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for mode in MODES:
            c, a_ = Counter(), Counter()
            for call in calls:
                got, exp = mode_types(call["variants"][v], mode), expected(meta[call["id"]])
                for cat in CATS:
                    k = a_ if cat == AUTHORITY else c
                    k["tp"] += cat in got and cat in exp
                    k["fp"] += cat in got and cat not in exp
                    k["fn"] += cat in exp and cat not in got
            comp = [x for x in calls if meta[x["id"]]["compliant"]]
            hard = [x for x in calls if meta[x["id"]]["hard_negative"]]
            hit = lambda x: bool(mode_types(x["variants"][v], mode) & set(CONDUCT))
            L.append(f"| {v} | {mode} | {c['tp']}/{c['fp']}/{c['fn']} | {sum(map(hit, comp))}/{len(comp)} | "
                     f"{sum(map(hit, hard))}/{len(hard)} | {a_['tp']}/{a_['fp']}/{a_['fn']} |")
    L.append("")

    def per_lang_table(variant: str) -> list[str]:
        out = [f"### {variant}", "", "| category | language | " + " | ".join(f"{m} TP/FP/FN" for m in MODES) + " |",
               "|---|---|" + "---|" * len(MODES)]
        tot = {m: Counter() for m in MODES}
        for cat in CATS:
            for lang in LANG_ORDER:
                cells = []
                for mode in MODES:
                    c = Counter()
                    for call in calls:
                        if call["language"] != lang:
                            continue
                        got, exp = mode_types(call["variants"][variant], mode), expected(meta[call["id"]])
                        c["tp"] += cat in got and cat in exp
                        c["fp"] += cat in got and cat not in exp
                        c["fn"] += cat in exp and cat not in got
                    if cat in CONDUCT:
                        tot[mode].update(c)
                    cells.append(f"{c['tp']}/{c['fp']}/{c['fn']}")
                label = f"`{cat}`" + (" (verify flag)" if cat == AUTHORITY else "")
                out.append(f"| {label} | {LANG_NAME[lang]} | " + " | ".join(cells) + " |")
        out.append("| **conduct total** | all | " + " | ".join(
            f"**{tot[m]['tp']}/{tot[m]['fp']}/{tot[m]['fn']}**" for m in MODES) + " |")
        return out + [""]

    L += ["## Per-category TP / FP / FN per language", "",
          "Call-level: a category counts as flagged when any segment of the call produced a flag of that type. "
          "Conduct = intimidation, abusive language, third-party contact; authority claims are a separate "
          "verify flag scored against the agent's mention of police / arrest / FIR / court / legal notice.", ""]
    for v in VARIANTS:
        L += per_lang_table(v)

    L += ["## False flags on compliant and hard-negative calls (conduct categories only)", "",
          "| variant | language | " + " | ".join(f"{m} compliant | {m} hard-neg" for m in MODES) + " |",
          "|---|---|" + "---|---|" * len(MODES)]
    ff_list = []
    for v in VARIANTS:
        for lang in LANG_ORDER + ["all"]:
            sel = [c for c in calls if lang == "all" or c["language"] == lang]
            cells = []
            for mode in MODES:
                comp = [c for c in sel if meta[c["id"]]["compliant"]]
                hard = [c for c in sel if meta[c["id"]]["hard_negative"]]
                hit = lambda c: bool(mode_types(c["variants"][v], mode) & set(CONDUCT))
                cells += [f"{sum(map(hit, comp))}/{len(comp)}", f"{sum(map(hit, hard))}/{len(hard)}"]
                if lang == "all":
                    ff_list += [(v, mode, c["id"], f) for c in comp for f in c["variants"][v]["flags"]
                                if f["detector"] in MODES[mode] and f["concern_type"] in CONDUCT]
            name = "**all**" if lang == "all" else LANG_NAME[lang]
            L.append(f"| {v} | {name} | " + " | ".join(cells) + " |")
    L.append("")
    authority_on_compliant = [(v, c["id"], f) for v in VARIANTS for c in calls if meta[c["id"]]["compliant"]
                              for f in c["variants"][v]["flags"] if f["concern_type"] == AUTHORITY]
    L += [f"Authority verify flags on compliant calls (expected where the agent mentions a legal notice; not "
          f"conduct false flags): {len(authority_on_compliant)} across both variants.", ""]

    L += ["## Quote verification", "", "| variant | LLM quotes rejected | flags dropped after merge |", "|---|---|---|"]
    rej_list = []
    for v in VARIANTS:
        rej = [(c["id"], r) for c in calls for r in c["variants"][v]["detectors"]["llm_quotes_rejected"]]
        rej_list += [(v, cid, r) for cid, r in rej]
        dropped = sum(c["variants"][v]["detectors"]["flags_dropped_by_quote_check"] for c in calls)
        n_llm = sum(1 for c in calls for f in c["variants"][v]["flags"] if f["detector"] in {"LLM", "both"})
        L.append(f"| {v} | {len(rej)} of {len(rej) + n_llm} LLM outputs | {dropped} |")
    L.append("")
    for v, cid, r in rej_list:
        L.append(f"- {v} {cid} {r.get('segment_id')} `{r.get('type')}` — {r['reason']}: “{r.get('quote')}”")
    llm_bad = [(v, c["id"], c["variants"][v]["detectors"]["llm_status"]) for v in VARIANTS for c in calls
               if c["variants"][v]["detectors"]["llm_status"] != "ok"]
    L += ["", f"LLM step not ok (rules-only fallback, no LLM flags): {len(llm_bad)} of {len(calls) * len(VARIANTS)} "
          "analyses" + (": " + "; ".join(f"{cid} {v} ({s})" for v, cid, s in llm_bad) if llm_bad else "") + ".", ""]

    L += ["## Omissions: detected and recovered", "",
          "A turn is omitted when its best word-window match against the first-pass STT text of the overlapping "
          "segments is below 0.5 (≤ 3-word turns need 0.8). Detected = ≥ 50 % of the turn lies in a segment the "
          "coverage check flagged. Recovered = present after re-split recovery. Still missing = absent from the "
          "final transcript; surfaced = it overlaps an 'untranscribed speech' warning shown to the reviewer.", "",
          "| variant | language | turns | omitted | detected | recovered | still missing (surfaced) | segments flagged |",
          "|---|---|---|---|---|---|---|---|"]
    om_detail = []
    for v in VARIANTS:
        for lang in LANG_ORDER + ["all"]:
            sel = [c for c in calls if lang == "all" or c["language"] == lang]
            rows = [(c["id"], r) for c in sel for r in omission_rows(c, c["variants"][v])]
            segs = [g for c in sel for g in c["variants"][v]["segments"]]
            o = [r for _, r in rows if r["omitted"]]
            miss = [r for _, r in rows if r["missing_after"]]
            name = "**all**" if lang == "all" else LANG_NAME[lang]
            L.append(f"| {v} | {name} | {len(rows)} | {len(o)} | {sum(r['detected'] for r in o)} | "
                     f"{sum(r['recovered'] for r in o)} | {len(miss)} ({sum(r['surfaced_after'] for r in miss)}) | "
                     f"{sum(g['flagged'] for g in segs)}/{len(segs)} |")
            if lang == "all":
                om_detail += [(v, cid, r) for cid, r in rows if r["omitted"] or r["missing_after"]]
    L.append("")
    for v, cid, r in om_detail:
        state = ("recovered" if r["recovered"] else "still missing" + (", surfaced" if r["surfaced_after"] else ", silent")
                 if r["missing_after"] else "")
        L.append(f"- {v} {cid} turn {r['turn']} ({r['speaker']}, {len(r['text'].split())} words): match "
                 f"{r['ratio_before']} → {r['ratio_after']}; {'detected' if r['detected'] else 'not detected'}; {state}")
    L.append("")

    for mode in MODES:
        L += [f"## {mode}: every miss and false flag", ""]
        any_ = False
        for v in VARIANTS:
            for c in calls:
                got, exp = mode_types(c["variants"][v], mode), expected(meta[c["id"]])
                for cat in CATS:
                    if cat in exp and cat not in got:
                        spans = [s["text"] for s in meta[c["id"]]["concern_spans"] if s["type"] == cat]
                        L.append(f"- {v} MISS **{c['id']}** `{cat}` — “{spans[0] if spans else ''}”")
                        any_ = True
        for v, m, cid, f in ff_list:
            if m == mode:
                L.append(f"- {v} FALSE FLAG **{cid}** `{f['concern_type']}`: “{f['quote']}”")
                any_ = True
        for v in VARIANTS:
            for c in calls:
                if meta[c["id"]]["authority_mention"]:
                    continue
                for f in c["variants"][v]["flags"]:
                    if f["concern_type"] == AUTHORITY and f["detector"] in MODES[mode]:
                        L.append(f"- {v} FALSE VERIFY FLAG **{c['id']}** `{AUTHORITY}` (no agent authority mention "
                                 f"in the script): “{f['quote']}”")
                        any_ = True
        if not any_:
            L.append("None.")
        L.append("")

    a = data["api"]
    L += ["## API calls used", "",
          f"- Gnani Timbre TTS: {a['tts_calls']} requests ({a['tts_cached']} turns from cache)",
          f"- Gnani Prisma STT: {a['stt_calls']} requests incl. recovery pieces ({a['stt_cache_hits']} from cache)",
          f"- Gemini: {a['llm_requests_incl_retries']} requests incl. retries for {len(calls) * len(VARIANTS)} analyses "
          "(an analysis whose transcript is identical to an earlier one is answered from the LLM cache); non-200 attempts: "
          f"{len(a['llm_non_200_attempts'])}" + (f" ({'; '.join(a['llm_non_200_attempts'])})" if a['llm_non_200_attempts'] else ""),
          f"- Failed requests after client retries: {len(a['guard_events'])}"
          + (f" ({', '.join(a['guard_events'])})" if a["guard_events"] else ""), "",
          "## Limitations", "",
          "- Synthetic TTS speech, one noise type, one seed per call; real calls have overlap, accents, crosstalk and "
          "real line noise.",
          "- 18 calls = 6 situations × 3 languages; Kannada awaits native review; Hinglish is romanised and voiced by "
          "a hi-en TTS voice.",
          "- Call-level scoring: a flag of the right type anywhere in a call counts as a TP.",
          "- No speaker separation: rules and the LLM see both speakers' words.",
          "- The coverage baseline (chars per speech-second) was measured on the spike's clean audio, not on this set."]
    target = (out_dir(True) / "held_out_v3_results.md") if args.mock else DOC
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--mock", action="store_true", help="offline: mock TTS/STT/LLM, outputs/mock/e2e_v3")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
