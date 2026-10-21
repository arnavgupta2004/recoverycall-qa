"""End-to-end analysis of one call: audio -> VAD -> Prisma STT -> coverage check/recovery -> hybrid flags.

    from src.pipeline import analyse, live_services, mock_services, audit_json
    result = analyse("call.wav", "hi-IN", live_services(Path(".env")))

Reuses src.vad (segments <= 12 s), src.coverage (density check + re-split recovery), src.classifier
(frozen rules), src.llm_tagger (verified Gemini flags) and src.hybrid (merge). The LLM step has a
hard time cap; on timeout or error the result falls back to rule-based flags and says so.

Every temporary file lives in a per-call TemporaryDirectory that is deleted before returning.
No result contains audio bytes except result["summary"]["audio_wav"], which audit_json() drops.
Wording: flags carry the classifier's labels; nothing here produces a score or a legal finding.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src import audio, vad
from src.classifier import classify, verify_quote
from src.concern_rules import ABUSIVE, AUTHORITY, CATEGORY_LABELS, INTIMIDATION, REGULATORY_NOTE, THIRD_PARTY
from src.coverage import FLAG_FRACTION, check_segment, threshold_for
from src.gnani_client import GnaniClient, MockSession, load_api_key
from src.hybrid import merge_flags
from src.llm_tagger import PROMPT_VERSION, MockBackend, make_backend, tag
from src.stt_cache import stt_cache_path

APP_VERSION = "t4-2026-10-21"
LANGUAGES = {"hi-IN": "Hindi", "kn-IN": "Kannada", "en-IN": "Hinglish / English"}
SUMMARY_VOICES = {"hi-IN": "Hemraj", "kn-IN": "Kavin", "en-IN": "Pranav"}
LLM_TIMEOUT_S = 20.0
LLM_UNAVAILABLE_NOTICE = "Advanced check unavailable — showing rule-based flags only"
RULES_VERSION = "rules v1, frozen 2026-10-21 (commit 12cb3b1)"
DETECTOR_NAMES = {"rules": "rules", "llm": "LLM", "rules+llm": "both"}
CONDUCT = [INTIMIDATION, ABUSIVE, THIRD_PARTY]
WINDOW = (dt.time(8, 0), dt.time(19, 0))   # permitted calling hours (IST), to verify against the RBI circular


class AnalysisError(ValueError):
    """Input problem the user can fix (e.g. audio too long); message is safe to show."""


# --------------------------------------------------------------------------- services


@dataclass
class Services:
    """External calls used by analyse(); swap in mocks for tests and offline runs."""
    transcribe: Callable[[Path, str, float, float], dict]   # (wav, language, start_s, end_s) -> STT record
    llm_backend: Any | None                                 # None = advanced check disabled
    synthesize: Callable[[str, str, str], bytes] | None     # (text, voice, language) -> WAV bytes
    llm_cache_dir: Path | None = None
    counters: dict = field(default_factory=lambda: {"stt": 0, "stt_cache_hits": 0, "llm": 0, "tts": 0})


def _gnani_transcriber(client: GnaniClient, cache_dir: Path | None, counters: dict):
    def transcribe(path: Path, language: str, start_s: float, end_s: float) -> dict:
        cache_file = stt_cache_path(cache_dir, path, language, "verbatim")[0] if cache_dir else None
        if cache_file and cache_file.exists():
            counters["stt_cache_hits"] += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))
        res = client.transcribe(path, language, "verbatim")
        counters["stt"] += 1
        rec = {"transcript": res.transcript, "http_status": res.http_status, "error": res.error,
               "request_id": res.request_id, "latency_ms": res.latency_ms, "attempts": res.attempts,
               "ok": res.ok, "from_cache": False}
        if cache_file and res.ok:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return rec
    return transcribe


def _gnani_synthesizer(client: GnaniClient, counters: dict):
    def synthesize(text: str, voice: str, language: str) -> bytes:
        res = client.synthesize(text, voice=voice, language=language)
        counters["tts"] += 1
        if not res.ok:
            raise RuntimeError(f"TTS HTTP {res.http_status} {(res.error or {}).get('type')}")
        return res.audio
    return synthesize


def live_services(env_file: Path | None, stt_cache_dir: Path | None = None, llm_cache_dir: Path | None = None,
                  llm_timeout_s: float = LLM_TIMEOUT_S) -> Services:
    """Real Gnani + Gemini. Pass cache dirs only for bundled samples (uploads must not be stored)."""
    counters = {"stt": 0, "stt_cache_hits": 0, "llm": 0, "tts": 0}
    client = GnaniClient(api_key=load_api_key(env_file, "GNANI_API_KEY"))
    try:
        backend = make_backend(False, env_file, read_timeout_s=llm_timeout_s, max_tries=1, min_interval_s=0)
    except Exception:  # no GEMINI_API_KEY: rules only, with the notice
        backend = None
    return Services(_gnani_transcriber(client, stt_cache_dir, counters), backend,
                    _gnani_synthesizer(client, counters), llm_cache_dir, counters)


MOCK_UPLOAD_TEXT = {
    "hi-IN": "नमस्ते, आपके बारह हज़ार रुपये बाकी हैं। आज नहीं दिए तो हमारे आदमी आपके घर आएँगे।",
    "kn-IN": "ನಮಸ್ಕಾರ, ನಿಮ್ಮ ಹನ್ನೆರಡು ಸಾವಿರ ರೂಪಾಯಿ ಬಾಕಿ ಇದೆ. ಇಂದು ಕಟ್ಟದಿದ್ದರೆ ನಮ್ಮ ಜನ ನಿಮ್ಮ ಮನೆಗೆ ಬರುತ್ತಾರೆ.",
    "en-IN": "Hello, aapke barah hazaar rupaye baaki hain. Aaj nahi diya toh hamare log aapke ghar aayenge.",
}


def mock_services(reference: list[dict] | None = None, llm_backend: Any = "mock") -> Services:
    """Offline services. `reference` = [{start_s, end_s, text}] turns: the mock STT returns the text of
    turns overlapping each request's time range (so recovery and coverage logic run for real). Turns
    marked "mock_omit_in_long_requests" are dropped from requests longer than 8 s, reproducing the
    silent omission seen with real STT on sample D; they come back when recovery re-splits."""
    counters = {"stt": 0, "stt_cache_hits": 0, "llm": 0, "tts": 0}

    def transcribe(path: Path, language: str, start_s: float, end_s: float) -> dict:
        counters["stt"] += 1
        if reference is None:
            text = MOCK_UPLOAD_TEXT[language] if start_s < 1.0 else ""
        else:
            long_request = end_s - start_s > 8.0
            text = " ".join(t["text"] for t in reference
                            if min(end_s, t["end_s"]) - max(start_s, t["start_s"]) > 0.5 * (t["end_s"] - t["start_s"])
                            and not (long_request and t.get("mock_omit_in_long_requests")))
        return {"transcript": text, "http_status": 200, "error": None, "request_id": "mock", "ok": True,
                "from_cache": False, "latency_ms": 0.0, "attempts": 1}

    client = GnaniClient(api_key="mock-key", session=MockSession(), backoff_base_s=0.0)
    backend = MockBackend() if llm_backend == "mock" else llm_backend
    return Services(transcribe, backend, _gnani_synthesizer(client, counters), None, counters)


# --------------------------------------------------------------------------- helpers


def fmt_time(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


def out_of_hours(filename: str | None) -> dict:
    """Check a YYYYMMDD_HHMMSS timestamp in the file name against 08:00–19:00 (assumed IST)."""
    m = re.search(r"(\d{8})_(\d{6})", filename or "")
    if not m:
        return {"checked": False, "reason": "no YYYYMMDD_HHMMSS timestamp in the file name"}
    try:
        when = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return {"checked": False, "reason": f"'{m.group(0)}' is not a valid date and time"}
    outside = not (WINDOW[0] <= when.time() < WINDOW[1])
    res = {"checked": True, "call_time": when.strftime("%Y-%m-%d %H:%M:%S"), "timezone_assumed": "IST",
           "window": "08:00–19:00", "outside_window": outside}
    if outside:
        res["note"] = (f"Call time {when.strftime('%H:%M')} (from the file name) is outside 08:00–19:00 — "
                       f"potential conduct concern — human review required; {REGULATORY_NOTE}")
    return res


# --------------------------------------------------------------------------- spoken summary

NUM_WORDS = {
    "en-IN": "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen twenty".split(),
    "hi-IN": "शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस".split(),
    "kn-IN": ("ಸೊನ್ನೆ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು ಎಂಟು ಒಂಬತ್ತು ಹತ್ತು ಹನ್ನೊಂದು ಹನ್ನೆರಡು ಹದಿಮೂರು ಹದಿನಾಲ್ಕು "
              "ಹದಿನೈದು ಹದಿನಾರು ಹದಿನೇಳು ಹದಿನೆಂಟು ಹತ್ತೊಂಬತ್ತು ಇಪ್ಪತ್ತು").split(),
}
MORE_THAN_20 = {"en-IN": "more than twenty", "hi-IN": "बीस से अधिक", "kn-IN": "ಇಪ್ಪತ್ತಕ್ಕಿಂತ ಹೆಚ್ಚು"}

# Kannada wording is provisional pending native-speaker review, like the Kannada scripts.
TEMPLATES = {
    "en-IN": {
        "open": "RecoveryCall QA summary, for human review.",
        "names": {INTIMIDATION: "possible intimidation or threats", ABUSIVE: "possible abusive language",
                  THIRD_PARTY: "a possible threat to contact family, employer or neighbours"},
        "conduct": ("One passage was flagged as {name}.", "{n} passages were flagged as {name}."),
        "authority": ("One claim of police, arrest or court action needs to be verified.",
                      "{n} claims of police, arrest or court action need to be verified."),
        "none": "No potential conduct concerns were flagged.",
        "coverage": ("One part of the audio may not be fully transcribed; please listen and review.",
                     "{n} parts of the audio may not be fully transcribed; please listen and review."),
        "hours": "The call time in the file name is outside eight a.m. to seven p.m.; please verify.",
        "close": "These are potential concerns only, for a human reviewer to check against the RBI circular.",
    },
    "hi-IN": {
        "open": "रिकवरी कॉल क्यूए सारांश, मानवीय समीक्षा के लिए।",
        "names": {INTIMIDATION: "संभावित धमकी या डराने", ABUSIVE: "संभावित अपमानजनक भाषा",
                  THIRD_PARTY: "परिवार, दफ़्तर या पड़ोसियों से संपर्क की संभावित धमकी"},
        "conduct": ("{name} का एक अंश चिह्नित किया गया है।", "{name} के {n} अंश चिह्नित किए गए हैं।"),
        "authority": ("पुलिस, गिरफ़्तारी या अदालती कार्रवाई के एक दावे की जाँच ज़रूरी है।",
                      "पुलिस, गिरफ़्तारी या अदालती कार्रवाई के {n} दावों की जाँच ज़रूरी है।"),
        "none": "कोई संभावित आचरण संबंधी चिंता चिह्नित नहीं हुई।",
        "coverage": ("ऑडियो के एक हिस्से का ट्रांसक्रिप्ट अधूरा हो सकता है, कृपया सुनकर समीक्षा करें।",
                     "ऑडियो के {n} हिस्सों का ट्रांसक्रिप्ट अधूरा हो सकता है, कृपया सुनकर समीक्षा करें।"),
        "hours": "फ़ाइल नाम में दिया गया कॉल का समय सुबह आठ से शाम सात बजे के बाहर है, कृपया जाँच करें।",
        "close": "ये केवल संभावित चिंताएँ हैं, जिन्हें मानवीय समीक्षक आरबीआई परिपत्र के अनुसार जाँचेंगे।",
    },
    "kn-IN": {
        "open": "ರಿಕವರಿ ಕಾಲ್ ಕ್ಯೂಎ ಸಾರಾಂಶ, ಮಾನವ ಪರಿಶೀಲನೆಗಾಗಿ.",
        "names": {INTIMIDATION: "ಸಂಭಾವ್ಯ ಬೆದರಿಕೆ", ABUSIVE: "ಸಂಭಾವ್ಯ ಅವಮಾನಕರ ಭಾಷೆ",
                  THIRD_PARTY: "ಕುಟುಂಬ, ಕಚೇರಿ ಅಥವಾ ನೆರೆಹೊರೆಯವರನ್ನು ಸಂಪರ್ಕಿಸುವ ಸಂಭಾವ್ಯ ಬೆದರಿಕೆ"},
        "conduct": ("{name} ಎಂದು ಒಂದು ಭಾಗವನ್ನು ಗುರುತಿಸಲಾಗಿದೆ.", "{name} ಎಂದು {n} ಭಾಗಗಳನ್ನು ಗುರುತಿಸಲಾಗಿದೆ."),
        "authority": ("ಪೊಲೀಸ್, ಬಂಧನ ಅಥವಾ ನ್ಯಾಯಾಲಯ ಕ್ರಮದ ಒಂದು ಹೇಳಿಕೆಯನ್ನು ಪರಿಶೀಲಿಸಬೇಕಾಗಿದೆ.",
                      "ಪೊಲೀಸ್, ಬಂಧನ ಅಥವಾ ನ್ಯಾಯಾಲಯ ಕ್ರಮದ {n} ಹೇಳಿಕೆಗಳನ್ನು ಪರಿಶೀಲಿಸಬೇಕಾಗಿದೆ."),
        "none": "ಯಾವುದೇ ಸಂಭಾವ್ಯ ನಡವಳಿಕೆ ಕಾಳಜಿ ಗುರುತಿಸಲಾಗಿಲ್ಲ.",
        "coverage": ("ಆಡಿಯೊದ ಒಂದು ಭಾಗದ ಪ್ರತಿಲಿಪಿ ಅಪೂರ್ಣವಾಗಿರಬಹುದು, ದಯವಿಟ್ಟು ಕೇಳಿ ಪರಿಶೀಲಿಸಿ.",
                     "ಆಡಿಯೊದ {n} ಭಾಗಗಳ ಪ್ರತಿಲಿಪಿ ಅಪೂರ್ಣವಾಗಿರಬಹುದು, ದಯವಿಟ್ಟು ಕೇಳಿ ಪರಿಶೀಲಿಸಿ."),
        "hours": "ಫೈಲ್ ಹೆಸರಿನಲ್ಲಿರುವ ಕರೆ ಸಮಯ ಬೆಳಿಗ್ಗೆ ಎಂಟರಿಂದ ಸಂಜೆ ಏಳರ ಹೊರಗಿದೆ, ದಯವಿಟ್ಟು ಪರಿಶೀಲಿಸಿ.",
        "close": "ಇವು ಕೇವಲ ಸಂಭಾವ್ಯ ಕಾಳಜಿಗಳು; ಮಾನವ ಪರಿಶೀಲಕರು ಆರ್‌ಬಿಐ ಸುತ್ತೋಲೆಯೊಂದಿಗೆ ಪರಿಶೀಲಿಸಬೇಕು.",
    },
}


def _num(n: int, language: str) -> str:
    words = NUM_WORDS[language]
    return words[n] if n < len(words) else MORE_THAN_20[language]


def summary_text(flags: list[dict], warnings: list[dict], hours: dict, language: str) -> str:
    """Per-language template filled from the flags; numbers are spoken words (Timbre guidance)."""
    t = TEMPLATES[language]
    parts = [t["open"]]
    for cat in CONDUCT:
        n = sum(f["concern_type"] == cat for f in flags)
        if n:
            parts.append((t["conduct"][0] if n == 1 else t["conduct"][1]).format(name=t["names"][cat],
                                                                                 n=_num(n, language)))
    if not any(f["concern_type"] in CONDUCT for f in flags):
        parts.append(t["none"])
    n_auth = sum(f["concern_type"] == AUTHORITY for f in flags)
    if n_auth:
        parts.append((t["authority"][0] if n_auth == 1 else t["authority"][1]).format(n=_num(n_auth, language)))
    if warnings:
        n = len(warnings)
        parts.append((t["coverage"][0] if n == 1 else t["coverage"][1]).format(n=_num(n, language)))
    if hours.get("outside_window"):
        parts.append(t["hours"])
    parts.append(t["close"])
    text = " ".join(parts)
    return text[0].upper() + text[1:] if language == "en-IN" else text


# --------------------------------------------------------------------------- analyse


def _run_llm(services: Services, rows: list[dict], language: str, timeout_s: float) -> tuple[list, list, str]:
    """Returns (flags, rejected, status); status is 'ok' or why the advanced check is unavailable."""
    if services.llm_backend is None:
        return [], [], "disabled (no GEMINI_API_KEY)"
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(tag, rows, language, services.llm_backend, services.llm_cache_dir)
    try:
        res = fut.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        services.counters["llm"] += 1
        return [], [], f"timeout after {timeout_s:.0f} s"
    except Exception as exc:  # never let the advanced check break the analysis
        services.counters["llm"] += 1
        return [], [], f"error: {type(exc).__name__}"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if not res["from_cache"]:
        services.counters["llm"] += 1
    if res["error"]:
        return [], res["rejected"], f"error: {res['error'][:200]}"
    return res["flags"], res["rejected"], "ok"


def _highlights(text: str, flags: list[dict]) -> None:
    """Attach char offsets of each flag's quote within its row text (first occurrence)."""
    for f in flags:
        i = text.find(f["quote"])
        f["char_start"], f["char_end"] = (i, i + len(f["quote"])) if i >= 0 else (None, None)


def analyse(audio_path: str | Path, language: str, services: Services, *, filename: str | None = None,
            max_duration_s: float | None = None, llm_timeout_s: float = LLM_TIMEOUT_S,
            make_summary_audio: bool = True, flag_fraction: float = FLAG_FRACTION) -> dict:
    """Analyse one call. Returns a JSON-serialisable dict except result['summary']['audio_wav'] (bytes)."""
    if language not in LANGUAGES:
        raise AnalysisError(f"unsupported language {language!r}; choose one of {', '.join(LANGUAGES)}")
    t_all = time.perf_counter()
    latency: dict[str, float] = {}

    def lap(stage: str, t0: float) -> None:
        latency[stage] = round((time.perf_counter() - t0) * 1000, 1)

    with tempfile.TemporaryDirectory(prefix="rcqa_") as tmp_name:
        tmp = Path(tmp_name)
        t0 = time.perf_counter()
        wav = tmp / "call.wav"
        try:
            audio.normalize_turn(Path(audio_path), wav)
        except RuntimeError as exc:
            raise AnalysisError("could not read the audio file (WAV or MP3 expected)") from exc
        x, sr = audio.read_wav(wav)
        duration = len(x) / sr
        lap("decode", t0)
        if max_duration_s and duration > max_duration_s + 0.5:
            raise AnalysisError(f"audio is {fmt_time(duration)} long; the limit is {fmt_time(max_duration_s)}")
        if duration < 0.5:
            raise AnalysisError("audio is shorter than half a second")

        t0 = time.perf_counter()
        segs, diag = vad.segment(x, sr, vad.COARSE)
        lap("vad", t0)

        thr = threshold_for(language, flag_fraction)
        seg_records, rows = [], []
        t_stt = t_cov = 0.0
        for i, s in enumerate(segs):
            t0 = time.perf_counter()
            f = tmp / f"seg{i:02d}.wav"
            audio.write_wav(f, vad.cut_padded(x, sr, s["start_s"], s["end_s"], vad.COARSE.pad_s), sr)
            api = services.transcribe(f, language, s["start_s"], s["end_s"])
            f.unlink()
            t_stt += time.perf_counter() - t0

            t0 = time.perf_counter()

            def transcribe_piece(j, p, samples, i=i):
                pf = tmp / f"seg{i:02d}_piece{j:02d}.wav"
                audio.write_wav(pf, samples, sr)
                papi = services.transcribe(pf, language, p["start_s"], p["end_s"])
                pf.unlink()
                return papi["transcript"], {"http_status": papi["http_status"]}

            rec = check_segment(x, sr, s, api["transcript"], thr, diag["threshold_db"], transcribe_piece,
                                stt_error=f"STT error {api['http_status']}")
            t_cov += time.perf_counter() - t0
            seg_records.append({"segment": i, "start_s": s["start_s"], "end_s": s["end_s"], "speech_s": s["speech_s"],
                                "http_status": api["http_status"], "stt_error": api.get("error"),
                                **{k: rec[k] for k in ("flagged", "density_first", "density_final", "threshold",
                                                       "untranscribed")},
                                "recovered": bool(rec["recovery"] and rec["recovery"]["kept"]),
                                "transcript_first": rec["transcript_first"]})
            if rec["recovery"] and rec["recovery"]["kept"]:
                for p in rec["recovery"]["pieces"]:
                    rows.append({"start_s": p["start_s"], "end_s": p["end_s"], "text": p["transcript"],
                                 "origin": "re-split piece", "segment": i})
            else:
                rows.append({"start_s": s["start_s"], "end_s": s["end_s"], "text": rec["transcript_final"] or "",
                             "origin": "segment", "segment": i})
        latency["stt"] = round(t_stt * 1000, 1)
        latency["coverage_recovery"] = round(t_cov * 1000, 1)
        for k, r in enumerate(rows):
            r["segment_id"] = f"r{k:02d}"

        t0 = time.perf_counter()
        rule_flags = classify(rows, language)
        lap("rules", t0)
        t0 = time.perf_counter()
        llm_flags, rejected, llm_status = _run_llm(services, rows, language, llm_timeout_s)
        lap("llm", t0)
        merged = merge_flags(rule_flags, llm_flags, rows) if llm_status == "ok" else \
            [{**f, "source": "rules"} for f in rule_flags]
        by_id = {r["segment_id"]: r for r in rows}
        flags, dropped = [], 0
        for f in merged:
            row = by_id[f["segment_id"]]
            if not verify_quote(f["quote"], row["text"]):
                dropped += 1
                continue
            flags.append({"concern_type": f["concern_type"], "quote": f["quote"], "segment_id": f["segment_id"],
                          "start_s": row["start_s"], "end_s": row["end_s"], "time": fmt_time(row["start_s"]),
                          "label": CATEGORY_LABELS[f["concern_type"]], "regulatory_note": REGULATORY_NOTE,
                          "detector": DETECTOR_NAMES[f["source"]]})
        flags.sort(key=lambda f: (f["start_s"], f["concern_type"]))
        for r in rows:
            _highlights(r["text"], [f for f in flags if f["segment_id"] == r["segment_id"]])

        warnings = []
        for s in seg_records:
            for u in s["untranscribed"]:
                warnings.append({"kind": "untranscribed", "start_s": u["start_s"], "end_s": u["end_s"],
                                 "text": f"untranscribed speech at {fmt_time(u['start_s'])}–{fmt_time(u['end_s'])} "
                                         "— listen and review"})
            if s["recovered"]:
                warnings.append({"kind": "recovered", "start_s": s["start_s"], "end_s": s["end_s"],
                                 "text": f"low transcript coverage at {fmt_time(s['start_s'])}–{fmt_time(s['end_s'])}: "
                                         "re-transcribed in shorter pieces — listen and review"})
        warnings.sort(key=lambda w: w["start_s"])
        hours = out_of_hours(filename)

        t0 = time.perf_counter()
        text = summary_text(flags, warnings, hours, language)
        summary = {"text": text, "voice": SUMMARY_VOICES[language], "language": language, "audio_wav": None,
                   "status": "not requested"}
        if make_summary_audio and services.synthesize is not None:
            try:
                summary["audio_wav"] = services.synthesize(text, SUMMARY_VOICES[language], language)
                summary["status"] = "ok"
            except Exception as exc:
                summary["status"] = f"unavailable ({type(exc).__name__})"
        lap("summary_tts", t0)
    temp_removed = not Path(tmp_name).exists()
    latency["total"] = round((time.perf_counter() - t_all) * 1000, 1)

    notices = [] if llm_status == "ok" else [LLM_UNAVAILABLE_NOTICE]
    return {
        "app_version": APP_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "language": language, "language_name": LANGUAGES[language], "filename": Path(filename).name if filename else None,
        "audio_sha256": hashlib.sha256(Path(audio_path).read_bytes()).hexdigest(),
        "duration_s": round(duration, 2),
        "detectors": {"rules": RULES_VERSION, "llm": getattr(services.llm_backend, "name", None),
                      "llm_prompt_version": PROMPT_VERSION, "llm_status": llm_status,
                      "llm_quotes_rejected": rejected, "flags_dropped_by_quote_check": dropped},
        "notices": notices,
        "segments": seg_records, "transcript": rows, "flags": flags, "coverage_warnings": warnings,
        "out_of_hours": hours, "summary": summary,
        "api_calls": dict(services.counters), "latency_ms": latency,
        "temp_files_removed": temp_removed,
        "wording": "Flags are potential conduct concerns for human review; authority claims are to verify. "
                   "No score or legal conclusion is produced.",
    }


def audit_json(result: dict) -> str:
    """The downloadable audit record: everything except audio bytes."""
    r = json.loads(json.dumps({k: v for k, v in result.items() if k != "summary"}, ensure_ascii=False))
    r["summary"] = {k: v for k, v in result["summary"].items() if k != "audio_wav"}
    return json.dumps(r, ensure_ascii=False, indent=2)
