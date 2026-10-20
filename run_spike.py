#!/usr/bin/env python3
"""RecoveryCall QA API spike: Timbre TTS -> phone-channel simulation -> Prisma STT -> metrics.

    python run_spike.py synthesize | degrade | transcribe | report | all  [--mock]

STT result cache notes
----------------------
* Keys are SHA-256(audio bytes) + language_code + format, so regenerated audio at the
  same path always produces a different key.
* Successful results and definitive 400s are cached.  403s are NOT cached (the key may
  be temporarily revoked or misconfigured; always retry).
* A cached non-OK (400) result prints a "[WARN cached-failure]" line.
* --rerun-failures deletes only the non-OK cache entries so a selective retry is possible
  without --force (which would discard all cached successes too).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from src import audio
from src.gnani_client import ConfigError, GnaniClient, MockSession, make_client
from src.metrics import error_rates, evaluate_spans, normalize, script_check

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "data" / "scripts.json"
VARIANTS = ["clean", "phone", "phone_noisy"]
TARGET_SNR_DB = 10.0
MOCK_CORRUPTION = {"clean": 0.05, "phone": 0.12, "phone_noisy": 0.25, "phone_ulaw_container": 0.12}


class Paths:
    def __init__(self, mock: bool):
        self.outputs = ROOT / "outputs" / "mock" if mock else ROOT / "outputs"
        self.tts_cache = ROOT / "cache" / ("tts_mock" if mock else "tts")
        self.stt_cache = self.outputs / "stt_cache"
        self.results = self.outputs / "results.json"
        self.report = self.outputs / "report.md"
        self.tts_log = self.outputs / "tts_log.json"

    def scenario_dir(self, sid: str) -> Path:
        return self.outputs / "audio" / sid


def load_scripts() -> dict:
    return json.loads(SCRIPTS.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- synthesize


def tts_cache_path(paths: Paths, sid: str, idx: int, payload: dict) -> Path:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
    return paths.tts_cache / sid / f"turn{idx:02d}_{digest}.wav"


def cmd_synthesize(args, paths: Paths, client: GnaniClient) -> None:
    audio.require_ffmpeg()
    scripts = load_scripts()
    model, speed = scripts["tts_model"], scripts.get("tts_speed", 1.0)
    tts_log, made, cached, failed = [], 0, 0, 0
    for sc in scripts["scenarios"]:
        sid, sdir = sc["id"], paths.scenario_dir(sc["id"])
        turn_wavs, turn_meta, ok = [], [], True
        for i, turn in enumerate(sc["turns"]):
            voice = turn.get("voice") or sc["voices"][turn["speaker"]]
            language = turn.get("tts_language") or sc["tts_language"]
            payload = {"text": turn["text"], "voice": voice, "language": language, "model": model, "speed": speed}
            raw = tts_cache_path(paths, sid, i, payload)
            entry = {"scenario": sid, "turn": i, "voice": voice, "language": language, "model": model,
                     "cache_path": str(raw.relative_to(ROOT))}
            if raw.exists():
                cached += 1
                entry["status"] = "cached"
            else:
                res = client.synthesize(turn["text"], voice=voice, language=language, model=model, speed=speed)
                entry.update({"http_status": res.http_status, "latency_ms": res.latency_ms,
                              "attempts": res.attempts, "content_type": res.content_type, "error": res.error})
                if not res.ok:
                    failed += 1
                    ok = False
                    entry["status"] = "failed"
                    log(f"  TTS FAILED {sid} turn {i} ({voice}/{language}): {res.http_status} {res.error}")
                    tts_log.append(entry)
                    continue
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(res.audio)
                made += 1
                entry["status"] = "synthesized"
                log(f"  TTS {sid} turn {i} {voice}/{language}: {len(res.audio)} bytes, {res.latency_ms:.0f} ms")
            norm = sdir / "turns" / f"turn{i:02d}.wav"
            audio.normalize_turn(raw, norm)
            entry["normalized_duration_s"] = round(audio.wav_duration(norm), 3)
            tts_log.append(entry)
            turn_wavs.append(norm)
            turn_meta.append({"turn": i, "speaker": turn["speaker"], "voice": voice, "tts_language": language,
                              "text": turn["text"], "critical_spans": turn.get("critical_spans", [])})
        if not ok:
            log(f"  {sid}: skipped concatenation because at least one turn failed")
            continue
        offsets = audio.concatenate(turn_wavs, sdir / "clean.wav")
        for meta, (start, end) in zip(turn_meta, offsets):
            meta["start_s"], meta["end_s"] = round(start, 4), round(end, 4)
        total = audio.wav_duration(sdir / "clean.wav")
        manifest = {
            "scenario": sid, "title": sc["title"], "stt_language_codes": sc["stt_language_codes"],
            "primary_stt_language_code": sc.get("primary_stt_language_code", sc["stt_language_codes"][0]),
            "sample_rate": audio.CLEAN_SR, "gap_s": audio.GAP_S, "duration_s": round(total, 3),
            "clean_wav": str((sdir / "clean.wav").relative_to(ROOT)), "turns": turn_meta,
            "max_chunk_s": args.max_chunk_s,
            "chunks": audio.plan_chunks(offsets, total, max_s=args.max_chunk_s),
        }
        (sdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  {sid}: {len(turn_meta)} turns, {total:.1f}s, {len(manifest['chunks'])} upload chunk(s)")
    paths.outputs.mkdir(parents=True, exist_ok=True)
    paths.tts_log.write_text(json.dumps(tts_log, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"synthesize: {made} TTS calls made, {cached} cached, {failed} failed")
    if failed:
        sys.exit(1)


# --------------------------------------------------------------------------- degrade


def load_manifests(paths: Paths) -> list[dict]:
    out = []
    for sc in load_scripts()["scenarios"]:
        mpath = paths.scenario_dir(sc["id"]) / "manifest.json"
        if not mpath.exists():
            log(f"  {sc['id']}: no manifest, run synthesize first")
            continue
        out.append(json.loads(mpath.read_text(encoding="utf-8")))
    return out


def cmd_degrade(args, paths: Paths, client=None) -> None:
    audio.require_ffmpeg()
    for n, m in enumerate(load_manifests(paths)):
        sdir = paths.scenario_dir(m["scenario"])
        offsets = [(t["start_s"], t["end_s"]) for t in m["turns"]]
        # Re-plan chunks so --max-chunk-s takes effect without re-synthesizing.
        m["max_chunk_s"] = args.max_chunk_s
        m["chunks"] = audio.plan_chunks(offsets, m["duration_s"], max_s=args.max_chunk_s)
        shutil.rmtree(sdir / "chunks", ignore_errors=True)
        # phone: band-limit + 8 kHz -> mu-law -> decode back to PCM for upload
        audio.bandlimit_8k(sdir / "clean.wav", sdir / "phone_pre.wav")
        audio.encode_ulaw(sdir / "phone_pre.wav", sdir / "phone_ulaw.wav")
        audio.decode_to_pcm(sdir / "phone_ulaw.wav", sdir / "phone.wav")
        # phone_noisy: band-limit + 8 kHz -> + pink noise @ ~10 dB -> mu-law -> decode
        pre, sr = audio.read_wav(sdir / "phone_pre.wav")
        noisy, snr = audio.mix_at_snr(pre, sr, offsets, TARGET_SNR_DB, seed=1000 + n)
        audio.write_wav(sdir / "phone_noisy_pre.wav", noisy, sr)
        audio.encode_ulaw(sdir / "phone_noisy_pre.wav", sdir / "phone_noisy_ulaw.wav")
        audio.decode_to_pcm(sdir / "phone_noisy_ulaw.wav", sdir / "phone_noisy.wav")
        snr_note = "pre-codec; post-codec SNR is lower"
        m["variants"] = {
            "clean": {"path": str((sdir / "clean.wav").relative_to(ROOT)), "sample_rate": 16000, "snr": None},
            "phone": {"path": str((sdir / "phone.wav").relative_to(ROOT)), "sample_rate": 8000, "snr": None,
                      "chain": "highpass 300 Hz, lowpass 3400 Hz, 8 kHz, G.711 mu-law encode, decode to s16le"},
            "phone_noisy": {"path": str((sdir / "phone_noisy.wav").relative_to(ROOT)), "sample_rate": 8000,
                            "snr": snr, "snr_measurement_note": snr_note,
                            "chain": "as phone, plus band-limited pink noise mixed before mu-law"},
            "phone_ulaw_container": {"path": str((sdir / "phone_ulaw.wav").relative_to(ROOT)), "sample_rate": 8000,
                                     "snr": None, "chain": "phone signal uploaded as mu-law WAV (no decode)"},
        }
        # Cut upload chunks (<= MAX_CHUNK_S) at turn boundaries for every variant.
        for ch in m["chunks"]:
            ch["files"] = {}
            for v in VARIANTS:
                dst = sdir / "chunks" / f"{v}_c{ch['chunk']}.wav"
                audio.cut(sdir / f"{v}.wav", dst, ch["start_s"], ch["end_s"])
                ch["files"][v] = str(dst.relative_to(ROOT))
        # phone_ulaw_container: cut only chunk 0 (one probe per scenario).
        # Dependency: first["files"]["phone"] must be written above before this encode runs.
        first = m["chunks"][0]
        ulaw_dst = sdir / "chunks" / f"phone_ulaw_container_c{first['chunk']}.wav"
        audio.encode_ulaw(ROOT / first["files"]["phone"], ulaw_dst)
        first["files"]["phone_ulaw_container"] = str(ulaw_dst.relative_to(ROOT))
        (sdir / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  {m['scenario']}: phone + phone_noisy (measured SNR {snr['measured_snr_db']} dB, "
            f"{snr['clipped_samples']} clipped samples), {len(m['chunks'])} chunk(s) per variant")


# --------------------------------------------------------------------------- transcribe


def plan_jobs(manifests: list[dict]) -> list[dict]:
    jobs = []
    for m in manifests:
        if "variants" not in m:
            log(f"  {m['scenario']}: not degraded yet, run degrade first")
            continue
        codes, primary = m["stt_language_codes"], m["primary_stt_language_code"]
        for ch in m["chunks"]:
            for v in VARIANTS:
                for code in codes:
                    jobs.append({"manifest": m, "chunk": ch, "variant": v, "language_code": code, "format": "verbatim"})
            # ITN view of the amounts: once per clean sample, primary code. Excluded from WER.
            jobs.append({"manifest": m, "chunk": ch, "variant": "clean", "language_code": primary,
                         "format": "transcribe"})
        first = m["chunks"][0]
        jobs.append({"manifest": m, "chunk": first, "variant": "phone_ulaw_container", "language_code": primary,
                     "format": "verbatim"})
    return jobs


def current_spans(scenario: str, turn: int) -> list[dict]:
    """Critical spans for one turn from the current data/scripts.json.

    Manifests keep a snapshot of the spans from synthesis time; span metadata (e.g.
    negation_tokens) can change later without new audio, so scoring always reads it here.
    """
    sc = next(s for s in load_scripts()["scenarios"] if s["id"] == scenario)
    return sc["turns"][turn].get("critical_spans", [])


def chunk_reference(m: dict, ch: dict) -> tuple[str, list[dict]]:
    turns = [m["turns"][i] for i in ch["turns"]]
    spans = [dict(sp, turn=t["turn"]) for t in turns for sp in current_spans(m["scenario"], t["turn"])]
    return " ".join(t["text"] for t in turns), spans


def score(rec: dict) -> dict:
    """Fill metric fields on a result record (in place) from its transcript and reference."""
    rec.update({"wer": None, "cer": None, "sub": None, "del": None, "ins": None, "ref_words": None,
                "char_sub": None, "char_del": None, "char_ins": None,
                "script_mismatch": None, "script": None, "spans": [], "in_wer_aggregate": False})
    if rec["transcript"] is None:
        return rec
    rates = error_rates(rec["reference"], rec["transcript"])
    rec.update({k: rates[k] for k in ("wer", "cer", "sub", "del", "ins", "ref_words",
                                       "char_sub", "char_del", "char_ins")})
    sc = script_check(rec["reference"], rec["transcript"])
    rec["script"] = sc
    rec["script_mismatch"] = sc["script_mismatch"]
    rec["spans"] = evaluate_spans(rec["critical_spans"], rec["transcript"])
    rec["in_wer_aggregate"] = rec["format"] == "verbatim" and not rec["script_mismatch"]
    return rec


def stt_cache_path(cache_dir: Path, audio_path: Path, language_code: str, fmt: str) -> tuple[Path, str]:
    """Cache file for one STT request, keyed on the audio's content (not its path), language and format.

    Returns (cache file, full SHA-256 hex digest of the audio bytes).
    """
    digest = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
    return cache_dir / f"{digest[:16]}_{language_code}_{fmt}.json", digest


def cmd_transcribe(args, paths: Paths, client: GnaniClient) -> None:
    jobs = plan_jobs(load_manifests(paths))
    paths.stt_cache.mkdir(parents=True, exist_ok=True)
    results, calls, cached = [], 0, 0
    log(f"transcribe: {len(jobs)} STT requests planned")
    for j, job in enumerate(jobs, 1):
        m, ch = job["manifest"], job["chunk"]
        rel = ch["files"][job["variant"]]
        apath = ROOT / rel
        reference, spans = chunk_reference(m, ch)
        cache_file, digest = stt_cache_path(paths.stt_cache, apath, job["language_code"], job["format"])
        snr = m["variants"][job["variant"]]["snr"]
        rec = {
            "scenario": m["scenario"], "chunk": ch["chunk"], "turns": ch["turns"], "variant": job["variant"],
            "language_code": job["language_code"], "format": job["format"], "audio_path": rel,
            "audio_sha256": digest, "duration_s": round(audio.wav_duration(apath), 3)
            if job["variant"] != "phone_ulaw_container" else round(ch["end_s"] - ch["start_s"], 3),
            "measured_snr_db": snr["measured_snr_db"] if snr else None,
            "max_chunk_s": m.get("max_chunk_s", audio.MAX_CHUNK_S),
            "reference": reference, "critical_spans": spans,
        }
        if cache_file.exists() and not args.force:
            api = json.loads(cache_file.read_text(encoding="utf-8"))
            cached += 1
            api["from_cache"] = True
            if not api.get("ok"):
                label = f"{rec['scenario']} c{rec['chunk']} {rec['variant']} {rec['language_code']} {rec['format']}"
                log(f"[WARN cached-failure] {label}: HTTP {api.get('http_status')} {(api.get('error') or {}).get('type')}")
        else:
            if isinstance(client.session, MockSession):
                client.session.register_reference(apath, reference, MOCK_CORRUPTION[job["variant"]])
            res = client.transcribe(apath, job["language_code"], job["format"])
            calls += 1
            api = {**res.to_dict(), "from_cache": False, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            # Cache successes and definitive 400s (invalid request, not retryable).
            # Do NOT cache 403s: the key may be temporarily revoked; always retry.
            if res.ok or res.http_status == 400:
                cache_file.write_text(json.dumps(api, ensure_ascii=False, indent=2), encoding="utf-8")
        rec.update({"http_status": api["http_status"], "latency_ms": api["latency_ms"],
                    "attempts": api["attempts"], "request_id": api["request_id"],
                    "transcript": api["transcript"], "error": api["error"], "from_cache": api["from_cache"]})
        score(rec)
        results.append(rec)
        status = rec["http_status"] if rec["error"] is None else f"{rec['http_status']} {rec['error'].get('type')}"
        wer = f"WER {rec['wer']:.3f}" if rec["wer"] is not None else "no WER"
        log(f"  [{j}/{len(jobs)}] {rec['scenario']} c{rec['chunk']} {rec['variant']:<20} {rec['language_code']} "
            f"{rec['format']:<10} {status} {wer}{' SCRIPT-MISMATCH' if rec['script_mismatch'] else ''}"
            f"{' (cached)' if rec['from_cache'] else ''}")
    paths.results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"transcribe: {calls} STT calls made, {cached} served from cache -> {paths.results.relative_to(ROOT)}")


# --------------------------------------------------------------------------- report


def _fmt(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _md_cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def aggregate(recs: list[dict]) -> dict:
    words = sum(r["ref_words"] for r in recs)
    edits = sum(r["sub"] + r["del"] + r["ins"] for r in recs)
    chars = sum(len(normalize(r["reference"])) for r in recs)
    # Sum raw character edit counts stored per record (avoids rounding error from cer*ref_chars).
    cedits = sum(r["char_sub"] + r["char_del"] + r["char_ins"] for r in recs)
    spans = [s for r in recs for s in r["spans"]]
    return {
        "runs": len(recs), "ref_words": words,
        "wer": edits / words if words else None, "cer": cedits / chars if chars else None,
        "sub": sum(r["sub"] for r in recs), "del": sum(r["del"] for r in recs), "ins": sum(r["ins"] for r in recs),
        "spans_exact": sum(s["status"] == "exact" for s in spans), "spans_total": len(spans),
        "negation_missing": sum(bool(s.get("negation_missing")) for s in spans),
    }


def cmd_report(args, paths: Paths, client=None) -> None:
    if not paths.results.exists():
        sys.exit(f"{paths.results.relative_to(ROOT)} not found; run transcribe first")
    results = json.loads(paths.results.read_text(encoding="utf-8"))
    for rec in results:  # re-score so metric and span changes apply without new API calls
        rec["critical_spans"] = [dict(sp, turn=t) for t in rec["turns"] for sp in current_spans(rec["scenario"], t)]
        score(rec)
    paths.results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    tts_log = json.loads(paths.tts_log.read_text(encoding="utf-8")) if paths.tts_log.exists() else []

    L = []
    L.append("# RecoveryCall QA: Gnani API spike report\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from `{paths.results.relative_to(ROOT)}` "
             f"({len(results)} STT requests).\n")
    if args.mock:
        L.append("> **MOCK RUN.** Both APIs were faked offline (tone audio, lightly corrupted reference text). "
                 "Every number below is synthetic and says nothing about Gnani's models.\n")
    L.append("WER/CER use the normalization in README.md (no number or word conversion). Aggregates are "
             "micro-averaged (total edits / total reference words) over `format=verbatim` runs that returned a "
             "transcript and passed the script check. `format=transcribe` (ITN) runs are excluded from WER.\n")

    # Aggregates
    ok_verbatim = [r for r in results if r["in_wer_aggregate"]]
    L.append("## Aggregates by variant\n")
    L.append("| variant | runs | ref words | WER | CER | S | D | I | spans exact | negation missing |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS + ["phone_ulaw_container"]:
        recs = [r for r in ok_verbatim if r["variant"] == v]
        if not recs:
            L.append(f"| {v} | 0 | – | – | – | – | – | – | – | – |")
            continue
        a = aggregate(recs)
        L.append(f"| {v} | {a['runs']} | {a['ref_words']} | {_fmt(a['wer'])} | {_fmt(a['cer'])} | {a['sub']} | "
                 f"{a['del']} | {a['ins']} | {a['spans_exact']}/{a['spans_total']} | {a['negation_missing']} |")
    L.append("\nThe variant rows mix scenarios and language codes. Use the breakdown below for like-for-like "
             "comparisons. `phone_ulaw_container` covers only the first chunk of each scenario.\n")
    L.append("> **Cross-chunk-length comparison note:** `phone_ulaw_container` is excluded from any comparison "
             "of chunk lengths (e.g. 28 s vs 12 s). At different chunk lengths the boundary of chunk 0 shifts, "
             "so the `phone_ulaw_container` run covers a different set of turns and therefore a different "
             "reference string. Comparing its WER across chunk-length settings is not like-for-like.\n")

    L.append("### By scenario, language code and variant\n")
    L.append("| scenario | lang | variant | runs | WER | CER | S/D/I | excluded (script mismatch / failed) |")
    L.append("|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in results:
        if r["format"] == "verbatim":
            groups[(r["scenario"], r["language_code"], r["variant"])].append(r)
    for (sid, code, v), recs in sorted(groups.items()):
        good = [r for r in recs if r["in_wer_aggregate"]]
        mism = sum(bool(r["script_mismatch"]) for r in recs)
        fail = sum(r["transcript"] is None for r in recs)
        a = aggregate(good) if good else None
        sdi = f"{a['sub']}/{a['del']}/{a['ins']}" if a else "–"
        L.append(f"| {sid} | {code} | {v} | {len(recs)} | {_fmt(a['wer']) if a else '–'} | "
                 f"{_fmt(a['cer']) if a else '–'} | {sdi} | {mism} / {fail} |")

    mismatches = [r for r in results if r["script_mismatch"]]
    if mismatches:
        L.append("\n### Script mismatches (excluded from aggregates)\n")
        for r in mismatches:
            s = r["script"]
            L.append(f"- {r['scenario']} c{r['chunk']} {r['variant']} {r['language_code']} {r['format']}: reference is "
                     f"{s['reference_script']}, hypothesis is mostly {s['hypothesis_script']} "
                     f"({_fmt(s['hypothesis_share_in_reference_script'])} of letters in the reference script).")

    # Per-run table
    L.append("\n## Per-run results\n")
    L.append("| scenario | chunk | variant | lang | format | dur s | SNR dB | HTTP | tries | latency ms | WER | CER | "
             "S/D/I | script mismatch | spans E/N/G | request_id |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        sp = r["spans"]
        egn = (f"{sum(s['status'] == 'exact' for s in sp)}/{sum(s['status'] == 'near' for s in sp)}/"
               f"{sum(s['status'] == 'garbled' for s in sp)}") if sp else "–"
        sdi = f"{r['sub']}/{r['del']}/{r['ins']}" if r["sub"] is not None else "–"
        L.append(f"| {r['scenario']} | {r['chunk']} | {r['variant']} | {r['language_code']} | {r['format']} | "
                 f"{_fmt(r['duration_s'], 1)} | {_fmt(r['measured_snr_db'], 1)} | {_fmt(r['http_status'])} | "
                 f"{r['attempts']} | {_fmt(r['latency_ms'], 0)} | {_fmt(r['wer'])} | {_fmt(r['cer'])} | {sdi} | "
                 f"{'yes' if r['script_mismatch'] else ('no' if r['script_mismatch'] is False else '–')} | {egn} | "
                 f"{_md_cell(r['request_id'] or '–')} |")

    # Critical errors
    L.append("\n## Critical errors\n")
    L.append("Every amount, negation, threat and family span that was not an exact match, in `format=verbatim` "
             "runs that returned a transcript. Runs excluded for script mismatch are listed separately above, "
             "since their spans cannot match by construction. **NEGATION LOST** marks a negation span whose "
             "negation marker is missing from the best-matching transcript text, the most dangerous error type.\n")
    L.append("| scenario | chunk | variant | lang | type | status | ratio | reference span | best match in transcript |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    n_crit = 0
    for r in results:
        if r["format"] != "verbatim" or r["transcript"] is None or r["script_mismatch"]:
            continue
        for s in r["spans"]:
            if s["status"] == "exact":
                continue
            n_crit += 1
            flag = " **NEGATION LOST**" if s.get("negation_missing") else ""
            L.append(f"| {r['scenario']} | {r['chunk']} | {r['variant']} | {r['language_code']} | {s['type']}{flag} | "
                     f"{s['status']} | {s['ratio']:.2f} | {_md_cell(s['span'])} | {_md_cell(s['best_match']) or '(nothing)'} |")
    if not n_crit:
        L.append("| – | – | – | – | – | – | – | none | – |")
    neg_lost = [(r, s) for r in results if r["format"] == "verbatim" and r["transcript"] is not None
                and not r["script_mismatch"] for s in r["spans"] if s.get("negation_missing")]
    L.append(f"\n{n_crit} non-exact critical spans; {len(neg_lost)} with the negation marker missing.\n")

    # ITN
    itn = [r for r in results if r["format"] == "transcribe"]
    if itn:
        L.append("## ITN (`format=transcribe`) on clean audio\n")
        L.append("Excluded from WER. Shown so you can see how amounts are rendered.\n")
        for r in itn:
            L.append(f"**{r['scenario']} chunk {r['chunk']} ({r['language_code']})**, HTTP {_fmt(r['http_status'])}\n")
            if r["transcript"] is not None:
                L.append(f"- transcript: {r['transcript']}")
                for s in r["spans"]:
                    if s["type"] == "amount":
                        L.append(f"- amount span `{s['span']}` → closest text `{s['best_match']}`")
            else:
                L.append(f"- error: {r['error']}")
            L.append("")

    # Unsupported / failed
    L.append("## Unsupported or failed features (as observed)\n")
    observed = []
    for e in tts_log:
        if e.get("status") == "failed":
            observed.append(f"- TTS {e['scenario']} turn {e['turn']} voice={e['voice']} language={e['language']} "
                            f"model={e['model']}: HTTP {e.get('http_status')} {json.dumps(e.get('error'), ensure_ascii=False)}")
    for r in results:
        if r["transcript"] is None:
            observed.append(f"- STT {r['scenario']} c{r['chunk']} {r['variant']} {r['language_code']} {r['format']}: "
                            f"HTTP {r['http_status']} after {r['attempts']} attempt(s) "
                            f"{json.dumps(r['error'], ensure_ascii=False)}")
    retried = [r for r in results if r["attempts"] and r["attempts"] > 1 and not r.get("from_cache")]
    for r in retried:
        observed.append(f"- STT {r['scenario']} c{r['chunk']} {r['variant']} {r['language_code']} needed "
                        f"{r['attempts']} attempts (retryable 429/5xx or transport error) and ended with HTTP {r['http_status']}")
    L.extend(observed or ["- No request failed in this run."])
    L.append("")
    L.append("**mu-law WAV container upload (`phone_ulaw_container`):**\n")
    for r in [r for r in results if r["variant"] == "phone_ulaw_container"]:
        outcome = ("accepted, transcript returned" if r["transcript"] is not None
                   else f"rejected: HTTP {r['http_status']} {json.dumps(r['error'], ensure_ascii=False)}")
        L.append(f"- {r['scenario']} ({r['language_code']}): {outcome}")
    L.append("")
    if mismatches:
        L.append(f"**Script mismatch:** {len(mismatches)} run(s) returned a different script than the reference "
                 "(see the list above). For the Roman-script Hinglish scenario this is the expected hi-IN behaviour "
                 "if the model writes Hindi in Devanagari.\n")
    L.append("Not tested here, per the docs: word timestamps (not offered by the REST STT response) and a dedicated "
             "Hinglish STT language code (none exists; hi-IN and en-IN were tried instead).\n")

    # Embed the silent-omission experiment report (experiment.py report) when it exists.
    exp_report = paths.outputs / "experiment" / "report.md"
    if exp_report.exists():
        body = exp_report.read_text(encoding="utf-8").split("\n", 1)[1]
        body = body.replace("\n## ", "\n### ").replace("](omission_vs_snr.png)", "](experiment/omission_vs_snr.png)")
        L.append("\n## Silent-omission experiment (VAD segmentation, coverage check, recovery)\n")
        L.append(f"From `{exp_report.relative_to(ROOT)}`; regenerate with `python experiment.py report`.\n")
        L.append(body)
    paths.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: wrote {paths.report.relative_to(ROOT)} and re-scored {paths.results.relative_to(ROOT)}")


# --------------------------------------------------------------------------- rerun-failures


def cmd_rerun_failures(paths: Paths) -> None:
    """Delete only the non-OK (failed) entries from the STT cache.

    This lets a selective retry proceed without --force, which would also discard all
    the successful cached results.
    """
    if not paths.stt_cache.exists():
        log("rerun-failures: STT cache directory does not exist, nothing to do")
        return
    removed = 0
    for cf in sorted(paths.stt_cache.iterdir()):
        if cf.suffix != ".json":
            continue
        try:
            entry = json.loads(cf.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not entry.get("ok"):
            cf.unlink()
            removed += 1
            log(f"  removed cached failure: {cf.name}")
    log(f"rerun-failures: removed {removed} failed cache entr{'y' if removed == 1 else 'ies'}")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["synthesize", "degrade", "transcribe", "report", "all",
                                        "rerun-failures"])
    ap.add_argument("--mock", action="store_true", help="fake both APIs offline; writes to outputs/mock and cache/tts_mock")
    ap.add_argument("--force", action="store_true", help="ignore the STT result cache (TTS cache is always used)")
    ap.add_argument("--rerun-failures", dest="rerun_failures_flag", action="store_true",
                    help="delete only the non-OK STT cache entries, then run transcribe (or use the "
                         "'rerun-failures' command alone to just clear them)")
    ap.add_argument("--max-chunk-s", type=float, default=audio.MAX_CHUNK_S,
                    help=f"max upload chunk length in seconds, split at turn boundaries (default {audio.MAX_CHUNK_S}, "
                         "max 30); applied by synthesize and degrade")
    args = ap.parse_args()
    if not 1.0 <= args.max_chunk_s <= 30.0:
        ap.error("--max-chunk-s must be between 1 and 30")
    paths = Paths(args.mock)
    # --rerun-failures: clear non-OK cache entries, then continue with the requested command.
    if args.rerun_failures_flag or args.command == "rerun-failures":
        cmd_rerun_failures(paths)
        if args.command == "rerun-failures":
            return
    needs_client = args.command in ("synthesize", "transcribe", "all")
    try:
        if needs_client:
            audio.require_ffmpeg()
        client = make_client(args.mock, ROOT / ".env") if needs_client else None
    except (ConfigError, audio.FFmpegMissing) as exc:
        sys.exit(f"error: {exc}")
    steps = {"synthesize": [cmd_synthesize], "degrade": [cmd_degrade], "transcribe": [cmd_transcribe],
             "report": [cmd_report], "all": [cmd_synthesize, cmd_degrade, cmd_transcribe, cmd_report]}
    try:
        for step in steps[args.command]:
            log(f"== {step.__name__.removeprefix('cmd_')}{' (mock)' if args.mock else ''}")
            step(args, paths, client)
    except audio.FFmpegMissing as exc:
        sys.exit(f"error: {exc}")
    if args.mock and client is not None:
        log(f"mock API calls: {client.session.calls}")


if __name__ == "__main__":
    main()
