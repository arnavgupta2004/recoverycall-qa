#!/usr/bin/env python3
"""Silent-omission experiment: VAD segmentation -> STT -> coverage check -> recovery, swept over SNR.

    python experiment.py prepare  [--mock]   # noise sweep audio + VAD segments (no API calls)
    python experiment.py plan     [--mock]   # exact base STT call count + worst-case recovery calls
    python experiment.py run      [--mock]   # STT with coverage check and recovery (API calls)
    python experiment.py report   [--mock]   # evaluation against the reference + outputs/experiment/report.md

Runtime path (prepare/run) uses the audio and the STT output only. The reference script and
the manifest turn offsets are used in exactly two places: to mix noise at a measured SNR
(simulation, as in run_spike degrade) and in `report` for evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_spike
from run_spike import Paths, current_spans, load_manifests, load_scripts, log, stt_cache_path
from src import audio, vad
from src.coverage import FLAG_FRACTION, char_count, check_segment, density, fine_pieces  # noqa: F401
from src.gnani_client import ConfigError, MockSession, make_client
from src.metrics import evaluate_spans, turn_missing

ROOT = run_spike.ROOT
SNRS = [20, 15, 10, 5]
N_SEEDS = 3
DROPPED_BELOW, PARTIAL_BELOW = 0.5, 0.8
ABORT_STATUSES = {403, 429, 500, 503}


def exp_dir(paths: Paths) -> Path:
    return paths.outputs / "experiment"


def seed_for(scenario_index: int, k: int) -> int:
    # k=0 reproduces run_spike degrade's phone_noisy seed (1000 + n), so the 10 dB / seed-0
    # condition is byte-identical to the manifest-chunked phone_noisy audio already run.
    return 1000 * (k + 1) + scenario_index


def conditions(manifest: dict, n: int, snrs: list[int], n_seeds: int) -> list[dict]:
    out = [{"condition": "clean", "snr_target": None, "seed": None}]
    for snr in snrs:
        for k in range(n_seeds):
            out.append({"condition": f"snr{snr}_s{k}", "snr_target": snr, "seed": seed_for(n, k)})
    return out


# --------------------------------------------------------------------------- prepare


def cmd_prepare(args, paths: Paths) -> None:
    audio.require_ffmpeg()
    base = exp_dir(paths)
    plan = []
    for n, m in enumerate(load_manifests(paths)):
        sid, sdir = m["scenario"], paths.scenario_dir(m["scenario"])
        offsets = [(t["start_s"], t["end_s"]) for t in m["turns"]]
        cdir = base / "audio" / sid
        cdir.mkdir(parents=True, exist_ok=True)
        pre, psr = audio.read_wav(sdir / "phone_pre.wav") if (sdir / "phone_pre.wav").exists() else (None, None)
        if pre is None:
            sys.exit(f"{sid}: phone_pre.wav missing; run `python run_spike.py degrade` first")
        for cond in conditions(m, n, args.snrs, args.seeds):
            wav = cdir / f"{cond['condition']}.wav"
            if cond["snr_target"] is None:
                x, sr = audio.read_wav(sdir / "clean.wav")
                audio.write_wav(wav, x, sr)
                cond["snr"] = None
            else:
                # Noise mixing uses the manifest offsets to measure SNR over speech: simulation only.
                noisy, snr = audio.mix_at_snr(pre, psr, offsets, cond["snr_target"], seed=cond["seed"])
                tmp = cdir / f"{cond['condition']}_pre.wav"
                audio.write_wav(tmp, noisy, psr)
                audio.encode_ulaw(tmp, cdir / f"{cond['condition']}_ulaw.wav")
                audio.decode_to_pcm(cdir / f"{cond['condition']}_ulaw.wav", wav)
                tmp.unlink()
                (cdir / f"{cond['condition']}_ulaw.wav").unlink()
                cond["snr"] = snr
                x, sr = audio.read_wav(wav)
            # Product path from here on: audio only.
            segs, diag = vad.segment(x, sr, vad.COARSE)
            seg_dir = cdir / cond["condition"]
            seg_dir.mkdir(exist_ok=True)
            for old in seg_dir.glob("*.wav"):
                old.unlink()
            for i, s in enumerate(segs):
                f = seg_dir / f"seg{i:02d}.wav"
                audio.write_wav(f, vad.cut_padded(x, sr, s["start_s"], s["end_s"], vad.COARSE.pad_s), sr)
                s.update({"segment": i, "path": str(f.relative_to(ROOT)),
                          "fine_pieces": len(fine_pieces(x, sr, s, diag["threshold_db"]))})
            plan.append({"scenario": sid, "language_code": m["primary_stt_language_code"],
                         "audio_path": str(wav.relative_to(ROOT)), "sample_rate": sr,
                         "duration_s": round(len(x) / sr, 3), "vad": diag, "segments": segs, **cond})
        log(f"  {sid}: {len(conditions(m, n, args.snrs, args.seeds))} conditions prepared")
    (base / "plan.json").write_text(json.dumps({"vad_coarse": vad.COARSE.to_dict(), "vad_fine": vad.FINE.to_dict(),
                                                "conditions": plan}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"prepare: wrote {(base / 'plan.json').relative_to(ROOT)}")


def load_plan(paths: Paths) -> dict:
    f = exp_dir(paths) / "plan.json"
    if not f.exists():
        sys.exit(f"{f.relative_to(ROOT)} not found; run `python experiment.py prepare` first")
    return json.loads(f.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- baseline


def cmd_baseline(paths: Paths) -> dict:
    """Chars per VAD speech-second per language, from the archived clean manifest-chunk runs.

    Uses only STT transcripts and audio (no reference text). Archived manifests give the
    chunk boundaries each archived run used.
    """
    totals = defaultdict(lambda: [0, 0.0, 0])
    sources = []
    archived = [(paths.outputs / "runs" / run / "results.json", paths.outputs / "runs" / run)
                for run in ("chunk28s", "chunk12s") if (paths.outputs / "runs" / run / "results.json").exists()]
    # Without archived runs (e.g. --mock), fall back to the current run_spike results and manifests.
    for rfile, mdir in archived or [(paths.results, None)]:
        if not rfile.exists():
            continue
        records = json.loads(rfile.read_text(encoding="utf-8"))
        manifests = {}
        for r in records:
            if r["variant"] != "clean" or r["format"] != "verbatim" or r["transcript"] is None or r["script_mismatch"]:
                continue
            sid = r["scenario"]
            if sid not in manifests:
                mf = (mdir / sid / "manifest.json") if mdir else (paths.scenario_dir(sid) / "manifest.json")
                manifests[sid] = json.loads(mf.read_text(encoding="utf-8"))
            ch = next(c for c in manifests[sid]["chunks"] if c["chunk"] == r["chunk"])
            if ch["turns"] != r["turns"]:
                sys.exit(f"{mf}: chunk {r['chunk']} turns {ch['turns']} do not match {rfile} record turns {r['turns']}")
            x, sr = audio.read_wav(paths.scenario_dir(sid) / "clean.wav")
            sub = x[int(round(ch["start_s"] * sr)): int(round(ch["end_s"] * sr))]
            regions, _ = vad.speech_regions(sub, sr, vad.COARSE)
            t = totals[r["language_code"]]
            t[0] += char_count(r["transcript"])
            t[1] += sum(e - s for s, e in regions)
            t[2] += 1
        sources.append(str(rfile.relative_to(ROOT)))
    if not totals:
        sys.exit("no archived clean runs under outputs/runs/; cannot compute the density baseline")
    baseline = {lang: {"chars_per_speech_s": round(c / s, 3), "chars": c, "speech_s": round(s, 2), "chunks": k}
                for lang, (c, s, k) in totals.items()}
    out = {"sources": sources, "method": "sum(normalized transcript chars excluding spaces) / sum(coarse VAD speech "
           "seconds) over clean verbatim chunks of the source runs", "baseline": baseline}
    (exp_dir(paths) / "baseline.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- plan


def cmd_plan(args, paths: Paths) -> None:
    plan = load_plan(paths)
    base = cached = worst = 0
    by_lang = defaultdict(lambda: [0, 0])
    for c in plan["conditions"]:
        for s in c["segments"]:
            f, _ = stt_cache_path(paths.stt_cache, ROOT / s["path"], c["language_code"], "verbatim")
            base += 1
            cached += f.exists()
            worst += s["fine_pieces"]
            by_lang[c["language_code"]][0] += 1
            by_lang[c["language_code"]][1] += s["fine_pieces"]
    n_cond = len(plan["conditions"])
    log(f"plan: {n_cond} conditions (3 scenarios x [clean + {len(args.snrs)} SNRs x {args.seeds} seeds])")
    log(f"  base segment requests: {base} ({cached} already cached, {base - cached} to send)")
    for lang, (b, w) in sorted(by_lang.items()):
        log(f"    {lang}: {b} segments, worst-case {w} recovery pieces")
    log(f"  recovery requests: depends on how many segments get flagged; worst case (every segment flagged) {worst}")
    log(f"  => between {base - cached} and {base - cached + worst} STT calls (plus retries)")


# --------------------------------------------------------------------------- run


class Abort(RuntimeError):
    pass


def mock_reference(manifest: dict, start_s: float, end_s: float, condition: str, key: str, piece: bool) -> str:
    """Mock only: text of turns overlapping [start, end], with a simulated silent omission."""
    turns = [t for t in manifest["turns"] if min(end_s, t["end_s"]) - max(start_s, t["start_s"]) > 0.3 * (t["end_s"] - t["start_s"])]
    rng = random.Random(f"mock-omit:{key}")
    p_omit = {"clean": 0.05}.get(condition, {20: 0.1, 15: 0.15, 10: 0.3, 5: 0.5}.get(
        int(condition[3:].split("_")[0]) if condition.startswith("snr") else 0, 0.1))
    if piece:
        p_omit /= 3
    if len(turns) > 1 and rng.random() < p_omit:
        keep = rng.randrange(0, len(turns) - 1) + 1
        turns = turns[:keep]  # drop the tail turns silently
    elif len(turns) == 1 and rng.random() < p_omit / 2:
        return ""
    return " ".join(t["text"] for t in turns)


def transcribe_cached(client, paths: Paths, path: Path, lang: str, force: bool, counters: dict,
                      mock_text: str | None) -> dict:
    cache_file, digest = stt_cache_path(paths.stt_cache, path, lang, "verbatim")
    if cache_file.exists() and not force:
        counters["cached"] += 1
        api = json.loads(cache_file.read_text(encoding="utf-8"))
        api["from_cache"] = True
        return api
    if isinstance(client.session, MockSession):
        client.session.register_reference(path, mock_text or "", 0.03)
    res = client.transcribe(path, lang, "verbatim")
    counters["calls"] += 1
    api = {**res.to_dict(), "from_cache": False, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if res.ok or res.http_status == 400:
        cache_file.write_text(json.dumps(api, ensure_ascii=False, indent=2), encoding="utf-8")
    if res.http_status in ABORT_STATUSES or (res.http_status is None and not res.ok):
        raise Abort(f"HTTP {res.http_status} after {res.attempts} attempt(s): {json.dumps(res.error, ensure_ascii=False)}")
    return api


def cmd_run(args, paths: Paths, client) -> None:
    plan = load_plan(paths)
    baseline = cmd_baseline(paths)["baseline"]
    manifests = {m["scenario"]: m for m in load_manifests(paths)}
    paths.stt_cache.mkdir(parents=True, exist_ok=True)
    counters = {"calls": 0, "cached": 0, "recovery_calls": 0}
    results, aborted = [], None
    try:
        for c in plan["conditions"]:
            lang, m = c["language_code"], manifests[c["scenario"]]
            thr = args.flag_fraction * baseline[lang]["chars_per_speech_s"]
            x, sr = audio.read_wav(ROOT / c["audio_path"])
            seg_out = []
            for s in c["segments"]:
                key = f"{c['scenario']}:{c['condition']}:{s['segment']}"
                mock_text = mock_reference(m, s["start_s"], s["end_s"], c["condition"], key, False) if args.mock else None
                api = transcribe_cached(client, paths, ROOT / s["path"], lang, args.force, counters, mock_text)
                first = api["transcript"]

                def transcribe_piece(j, p, samples, s=s, m=m, c=c, lang=lang):
                    f = ROOT / Path(s["path"]).parent / f"seg{s['segment']:02d}_pieces" / f"piece{j:02d}.wav"
                    f.parent.mkdir(exist_ok=True)
                    audio.write_wav(f, samples, sr)
                    key = f"{c['scenario']}:{c['condition']}:{s['segment']}:{j}"
                    mt = mock_reference(m, p["start_s"], p["end_s"], c["condition"], key, True) if args.mock else None
                    before = counters["calls"]
                    papi = transcribe_cached(client, paths, f, lang, args.force, counters, mt)
                    counters["recovery_calls"] += counters["calls"] - before
                    return papi["transcript"], {"path": str(f.relative_to(ROOT)), "http_status": papi["http_status"]}

                rec = {**{k: s[k] for k in ("segment", "start_s", "end_s", "speech_s", "path")},
                       "http_status": api["http_status"], "request_id": api["request_id"], "error": api["error"],
                       **check_segment(x, sr, s, first, thr, c["vad"]["threshold_db"], transcribe_piece,
                                       stt_error=f"STT error {api['http_status']}")}
                d0 = rec["density_first"]
                seg_out.append(rec)
                fmt = lambda v: "–" if v is None else f"{v:.1f}"
                log(f"  {c['scenario'][:1]} {c['condition']:<10} seg{s['segment']:02d} {s['start_s']:5.1f}-{s['end_s']:5.1f}s "
                    f"HTTP {rec['http_status']} density {fmt(d0)} / thr {thr:.1f}"
                    + (f"  FLAGGED -> {fmt(rec['density_final'])}"
                       f"{' kept' if rec['recovery']['kept'] else ' not kept'}"
                       f"{' UNTRANSCRIBED ' + str(len(rec['untranscribed'])) if rec['untranscribed'] else ''}"
                       if rec["flagged"] else "")
                    + (" (cached)" if api.get("from_cache") else ""))
            results.append({**{k: v for k, v in c.items() if k != "segments"}, "segments": seg_out})
    except Abort as exc:
        aborted = str(exc)
        log(f"ABORTED: {aborted}")
    out = {"aborted": aborted, "flag_fraction": args.flag_fraction, "baseline": baseline,
           "counters": counters, "conditions": results}
    (exp_dir(paths) / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"run: {counters['calls']} STT calls made ({counters['recovery_calls']} for recovery), "
        f"{counters['cached']} from cache -> {(exp_dir(paths) / 'results.json').relative_to(ROOT)}")
    if aborted:
        sys.exit(1)


# --------------------------------------------------------------------------- report (evaluation only)


def group_of(cond: dict) -> str:
    return "clean" if cond["snr_target"] is None else f"{cond['snr_target']} dB"


GROUP_ORDER = ["clean"] + [f"{s} dB" for s in SNRS]


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def evaluate(results: dict, manifests: dict) -> list[dict]:
    """Per-turn evaluation rows. Uses the reference (turn text, offsets): evaluation only."""
    rows = []
    for c in results["conditions"]:
        m = manifests[c["scenario"]]
        segs = c["segments"]
        for t in m["turns"]:
            ts, te, dur = t["start_s"], t["end_s"], t["end_s"] - t["start_s"]
            over = [g for g in segs if overlap(ts, te, g["start_s"], g["end_s"]) > 0]
            before = " ".join(g["transcript_first"] or "" for g in over)
            after = " ".join(g["transcript_final"] or "" for g in over)
            (dropped0, r0), (dropped1, r1) = turn_missing(t["text"], before), turn_missing(t["text"], after)
            flagged_cov = sum(overlap(ts, te, g["start_s"], g["end_s"]) for g in over if g["flagged"]) / dur
            untr_cov = sum(overlap(ts, te, u["start_s"], u["end_s"]) for g in over for u in g["untranscribed"]) / dur
            spans0 = evaluate_spans(current_spans(c["scenario"], t["turn"]), before)
            spans1 = evaluate_spans(current_spans(c["scenario"], t["turn"]), after)
            rows.append({
                "scenario": c["scenario"], "language_code": c["language_code"], "condition": c["condition"],
                "group": group_of(c), "turn": t["turn"], "ratio_before": round(r0, 3), "ratio_after": round(r1, 3),
                "dropped_before": dropped0, "partial_before": not dropped0 and r0 < PARTIAL_BELOW,
                "flagged": flagged_cov >= 0.5, "dropped_after": dropped1,
                "recovered": dropped0 and not dropped1,
                "surfaced": dropped1 and untr_cov >= 0.3, "silent_after": dropped1 and untr_cov < 0.3,
                "spans_garbled_before": sum(x["status"] == "garbled" for x in spans0),
                "spans_garbled_after": sum(x["status"] == "garbled" for x in spans1),
                "neg_lost_before": sum(bool(x.get("negation_missing")) for x in spans0),
                "neg_lost_after": sum(bool(x.get("negation_missing")) for x in spans1),
            })
    return rows


def seg_rows(results: dict, manifests: dict, turn_rows: list[dict]) -> list[dict]:
    dropped = {(r["scenario"], r["condition"], r["turn"]): r for r in turn_rows}
    out = []
    for c in results["conditions"]:
        m = manifests[c["scenario"]]
        for g in c["segments"]:
            inside = [t for t in m["turns"]
                      if overlap(t["start_s"], t["end_s"], g["start_s"], g["end_s"]) >= 0.5 * (t["end_s"] - t["start_s"])]
            bad = any(dropped[(c["scenario"], c["condition"], t["turn"])]["dropped_before"] or
                      dropped[(c["scenario"], c["condition"], t["turn"])]["partial_before"] for t in inside)
            out.append({"language_code": c["language_code"], "group": group_of(c), "flagged": g["flagged"],
                        "has_omission": bad, "pieces": len(g["recovery"]["pieces"]) if g["recovery"] else 0,
                        "kept": bool(g["recovery"] and g["recovery"]["kept"]),
                        "untranscribed": len(g["untranscribed"])})
    return out


def pct(a: int, b: int) -> str:
    return "–" if b == 0 else f"{100 * a / b:.0f}% ({a}/{b})"


def chart(turn_rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    langs = sorted({r["language_code"] for r in turn_rows})
    names = {"en-IN": "Hinglish (en-IN)", "kn-IN": "Kannada (kn-IN)", "hi-IN": "Hindi (hi-IN)"}
    series = [("dropped_before", "Missing before recovery", "#2a78d6", "o", "-"),
              ("dropped_after", "Missing after recovery", "#eb6834", "s", "--"),
              ("silent_after", "Missing and not surfaced (silent)", "#1baf7a", "^", ":")]
    fig, axes = plt.subplots(1, len(langs), figsize=(4.2 * len(langs), 3.6), sharey=True, facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    for ax, lang in zip(axes, langs):
        ax.set_facecolor("#fcfcfb")
        groups = [g for g in GROUP_ORDER if any(r["group"] == g and r["language_code"] == lang for r in turn_rows)]
        xs = np.arange(len(groups))
        for key, label, color, marker, ls in series:
            ys = []
            for g in groups:
                rs = [r for r in turn_rows if r["group"] == g and r["language_code"] == lang]
                ys.append(100 * sum(r[key] for r in rs) / len(rs))
            ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2, linestyle=ls, label=label)
        ax.set_xticks(xs, groups)
        ax.set_title(names.get(lang, lang), fontsize=11, color="#1a1a19", loc="left")
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b5b3a9")
        ax.tick_params(colors="#5f5e58", labelsize=9)
    axes[0].set_ylabel("Turns with omitted speech (%)", color="#5f5e58", fontsize=9)
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Omitted turns vs SNR (3 synthetic calls, 3 noise seeds; smoke test)", fontsize=11, x=0.01,
                 ha="left", color="#1a1a19")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def manifest_chunk_comparison(paths: Paths, manifests: dict, plan: dict) -> list[str]:
    """Omitted turns in the archived 12 s manifest-chunk run vs the VAD path, on byte-identical audio."""
    rfile = paths.outputs / "runs" / "chunk12s" / "results.json"
    if not rfile.exists():
        return []
    import hashlib
    records = json.loads(rfile.read_text(encoding="utf-8"))
    L = ["| language | audio | manifest 12 s chunks: turns missing | VAD segments before recovery | after recovery | audio identical |",
         "|---|---|---|---|---|---|"]
    exp = json.loads((exp_dir(paths) / "eval_turns.json").read_text(encoding="utf-8"))
    for sid, m in manifests.items():
        lang = m["primary_stt_language_code"]
        for variant, cond in (("clean", "clean"), ("phone_noisy", "snr10_s0")):
            recs = [r for r in records if r["scenario"] == sid and r["variant"] == variant and r["format"] == "verbatim"
                    and r["language_code"] == lang and r["transcript"] is not None]
            missing = 0
            for r in recs:
                for t in r["turns"]:
                    missing += turn_missing(m["turns"][t]["text"], r["transcript"])[0]
            ev = [e for e in exp if e["scenario"] == sid and e["condition"] == cond]
            if not ev:
                continue
            c = next(c for c in plan["conditions"] if c["scenario"] == sid and c["condition"] == cond)
            src = ROOT / c["audio_path"]
            ref = paths.scenario_dir(sid) / ("clean.wav" if variant == "clean" else "phone_noisy.wav")
            same = ref.exists() and hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(ref.read_bytes()).digest()
            L.append(f"| {lang} | {'clean' if variant == 'clean' else '10 dB, seed 0'} | {missing}/{len(m['turns'])} | "
                     f"{sum(e['dropped_before'] for e in ev)}/{len(ev)} | {sum(e['dropped_after'] for e in ev)}/{len(ev)} | "
                     f"{'yes' if same else 'no'} |")
    return L


def cmd_report(args, paths: Paths) -> None:
    rfile = exp_dir(paths) / "results.json"
    if not rfile.exists():
        sys.exit(f"{rfile.relative_to(ROOT)} not found; run `python experiment.py run` first")
    results = json.loads(rfile.read_text(encoding="utf-8"))
    plan = load_plan(paths)
    manifests = {m["scenario"]: m for m in load_manifests(paths)}
    turns = evaluate(results, manifests)
    (exp_dir(paths) / "eval_turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    segs = seg_rows(results, manifests, turns)
    langs = sorted({r["language_code"] for r in turns})
    names = {"en-IN": "Hinglish (en-IN)", "kn-IN": "Kannada (kn-IN)", "hi-IN": "Hindi (hi-IN)"}

    L = ["# Silent-omission experiment (VAD segmentation + coverage check + recovery)\n"]
    if args.mock:
        L.append("> **MOCK RUN.** Both APIs were faked; omissions were injected by the mock. Numbers test the "
                 "pipeline only and say nothing about Gnani's models.\n")
    if results["aborted"]:
        L.append(f"> **Run aborted:** {results['aborted']}. Results cover only the conditions completed before it.\n")
    L.append("**This is a smoke test on synthetic data:** 3 TTS-generated calls (21 turns), one noise type (band-limited "
             "pink noise), 3 noise seeds per SNR. Each call has 7 turns, so a noisy cell below is 21 turns (7 x 3 seeds) "
             "and one turn moves it by about 5 percentage points; a clean cell is 7 turns. Treat differences as "
             "directional only.\n")
    L.append("Runtime path uses audio and STT output only: energy VAD -> segments of at most "
             f"{vad.COARSE.max_segment_s:.0f} s cut at pause midpoints (whole timeline kept) -> STT -> flag a segment when "
             f"transcript characters per VAD speech-second fall below {results['flag_fraction']} x the language baseline -> "
             "re-split flagged segments at shorter pauses and keep the pieces' transcript if density improves -> anything "
             "still below threshold is reported as untranscribed speech with times. The reference script is used only "
             "for evaluation below (and to place noise at a measured SNR when simulating).\n")
    L.append("A turn counts as **missing** when its reference text matches the transcript of the segments it overlaps "
             f"with ratio < {DROPPED_BELOW} (same word-window matcher as the span check; turns of 3 words or fewer "
             "need >= 0.8, since a 2-word turn can reach 0.5 against an unrelated word). **Flagged** = at least half the "
             "turn lies in flagged segments. **Silent** = missing after recovery and not covered by an untranscribed-"
             "speech record.\n")

    L.append("## Density baseline\n")
    L.append("| language | chars per speech-second | from | threshold used |")
    L.append("|---|---|---|---|")
    for lang, b in sorted(results["baseline"].items()):
        L.append(f"| {lang} | {b['chars_per_speech_s']} | {b['chunks']} clean chunks, {b['speech_s']} s speech | "
                 f"{results['flag_fraction'] * b['chars_per_speech_s']:.2f} |")
    L.append("\nBaseline = normalized transcript characters (spaces excluded) / VAD speech seconds, over the clean "
             "verbatim chunks of the archived 28 s and 12 s runs. Those runs contained a few omissions themselves, so the "
             "baseline is slightly low, which makes flagging slightly less sensitive.\n")

    L.append("## Omitted turns vs SNR\n")
    L.append("| language | audio | turns | missing before recovery | missing after recovery | silent after recovery | "
             "partial before | garbled spans before → after | negation lost before → after |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lang in langs:
        for g in GROUP_ORDER:
            rs = [r for r in turns if r["language_code"] == lang and r["group"] == g]
            if not rs:
                continue
            seeds = sorted({r["condition"] for r in rs})
            per_seed = [sum(r["dropped_before"] for r in rs if r["condition"] == sd) for sd in seeds]
            spread = f" (per seed {'/'.join(map(str, per_seed))})" if len(seeds) > 1 else ""
            L.append(f"| {names.get(lang, lang)} | {g} | {len(rs)} | {pct(sum(r['dropped_before'] for r in rs), len(rs))}"
                     f"{spread} | {pct(sum(r['dropped_after'] for r in rs), len(rs))} | "
                     f"{pct(sum(r['silent_after'] for r in rs), len(rs))} | {sum(r['partial_before'] for r in rs)} | "
                     f"{sum(r['spans_garbled_before'] for r in rs)} → {sum(r['spans_garbled_after'] for r in rs)} | "
                     f"{sum(r['neg_lost_before'] for r in rs)} → {sum(r['neg_lost_after'] for r in rs)} |")
    png = exp_dir(paths) / "omission_vs_snr.png"
    if chart(turns, png):
        L.append(f"\n![Omitted turns vs SNR]({png.name})\n")
        L.append("The table above is the accessible view of this chart.\n")

    L.append("## Detection and recovery\n")
    L.append("| language | audio | missing turns | detected (flagged) | recovered | segments | flagged segments | "
             "false flags (no missing/partial turn) | untranscribed records |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lang in langs:
        for g in GROUP_ORDER:
            rs = [r for r in turns if r["language_code"] == lang and r["group"] == g]
            ss = [s for s in segs if s["language_code"] == lang and s["group"] == g]
            if not rs:
                continue
            miss = [r for r in rs if r["dropped_before"]]
            fl = [s for s in ss if s["flagged"]]
            L.append(f"| {names.get(lang, lang)} | {g} | {len(miss)} | {pct(sum(r['flagged'] for r in miss), len(miss))} | "
                     f"{pct(sum(r['recovered'] for r in miss), len(miss))} | {len(ss)} | {len(fl)} | "
                     f"{pct(sum(not s['has_omission'] for s in fl), len(ss))} | {sum(s['untranscribed'] for s in ss)} |")
    all_miss = [r for r in turns if r["dropped_before"]]
    clean_segs = [s for s in segs if s["group"] == "clean"]
    L.append(f"\n**Overall:** omission-detection recall {pct(sum(r['flagged'] for r in all_miss), len(all_miss))}, "
             f"recovery rate {pct(sum(r['recovered'] for r in all_miss), len(all_miss))}, false flags on clean audio "
             f"{pct(sum(s['flagged'] and not s['has_omission'] for s in clean_segs), len(clean_segs))} of clean segments, "
             f"silent omissions after recovery {pct(sum(r['silent_after'] for r in turns), len(turns))} of all turns.\n")

    L.append("### Where the missing turns come from\n")
    L.append("Conditions share the same speech, so a turn that one segment drops tends to be dropped in many "
             "conditions. Counts below are conditions (out of 13 per call) in which each turn was missing before "
             "recovery; a few turns dropped repeatedly are one failure pattern, not independent events.\n")
    L.append("| scenario | turn | text | missing before (conditions) | detected | recovered |")
    L.append("|---|---|---|---|---|---|")
    by_turn = defaultdict(list)
    for r in turns:
        if r["dropped_before"]:
            by_turn[(r["scenario"], r["turn"])].append(r)
    for (sid, t), rs in sorted(by_turn.items()):
        text = manifests[sid]["turns"][t]["text"]
        L.append(f"| {sid} | {t} | {text[:60]}{'…' if len(text) > 60 else ''} | {len(rs)} | "
                 f"{sum(r['flagged'] for r in rs)} | {sum(r['recovered'] for r in rs)} |")
    if not by_turn:
        L.append("| – | – | none | – | – | – |")
    L.append("")

    cmp_rows = manifest_chunk_comparison(paths, manifests, plan)
    if cmp_rows:
        L.append("## Manifest-chunk (12 s) vs VAD segments on the same audio\n")
        L.append("The 10 dB seed-0 condition reuses run_spike's phone_noisy noise seed; the last column checks the audio "
                 "is byte-identical, so the only difference is segmentation (no extra STT calls for the 12 s column).\n")
        L.extend(cmp_rows)
        L.append("")

    c = results["counters"]
    n_segs, n_pieces = len(segs), sum(s["pieces"] for s in segs)
    L.append("## Cost\n")
    L.append(f"- STT requests in this experiment: **{n_segs + n_pieces}** = {n_segs} base segments + {n_pieces} "
             f"recovery pieces (recovery added {pct(n_pieces, n_segs)} on top of the base).")
    L.append(f"- Sent to the API by the last `run` invocation: {c['calls']} ({c['recovery_calls']} recovery); "
             f"{c['cached']} came from the STT cache.")
    L.append("- TTS calls: 0 (cached turns reused).\n")

    untr = [(cc["scenario"], cc["condition"], u) for cc in results["conditions"] for g in cc["segments"] for u in g["untranscribed"]]
    L.append("## Untranscribed speech records\n")
    L.append("Speech the pipeline could not get a plausible transcript for after recovery. In a product these would be "
             "surfaced to a reviewer instead of silently disappearing.\n")
    if untr:
        L.append("| scenario | condition | start s | end s | speech s | reason |")
        L.append("|---|---|---|---|---|---|")
        for sid, cond, u in untr:
            L.append(f"| {sid} | {cond} | {u['start_s']:.2f} | {u['end_s']:.2f} | {u['speech_s']:.2f} | "
                     f"{u.get('reason', 'density below threshold')} |")
    else:
        L.append("None.")
    L.append("\n## Limits\n")
    L.append("- Synthetic TTS speech, 3 calls, one language per call, one noise type; real calls have overlap, "
             "music, crosstalk and codec loss that this does not model.")
    L.append("- The energy VAD under-detects speech at low SNR (it misses quiet consonants), which lowers the "
             "speech-seconds denominator and raises density, so low-SNR omissions are harder to flag.")
    L.append("- The density check catches omissions that remove characters; it cannot catch a fluent hallucination of "
             "the right length.")
    L.append("- 'Missing' uses a fuzzy text match against the reference; spelling variants can push a correct turn "
             "below the threshold and vice versa.\n")
    (exp_dir(paths) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: wrote {(exp_dir(paths) / 'report.md').relative_to(ROOT)}")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "plan", "run", "report"])
    ap.add_argument("--mock", action="store_true", help="fake both APIs; writes under outputs/mock/experiment")
    ap.add_argument("--force", action="store_true", help="ignore the STT result cache")
    ap.add_argument("--snrs", type=lambda v: [int(x) for x in v.split(",")], default=SNRS, help="e.g. 20,15,10,5")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="noise seeds per SNR")
    ap.add_argument("--flag-fraction", type=float, default=FLAG_FRACTION,
                    help="flag a segment below this fraction of the language's chars-per-speech-second baseline")
    args = ap.parse_args()
    paths = Paths(args.mock)
    exp_dir(paths).mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        cmd_prepare(args, paths)
    elif args.command == "plan":
        cmd_plan(args, paths)
    elif args.command == "run":
        try:
            client = make_client(args.mock, ROOT / ".env")
        except ConfigError as exc:
            sys.exit(f"error: {exc}")
        cmd_run(args, paths, client)
    else:
        cmd_report(args, paths)


if __name__ == "__main__":
    main()
